import os
import time
import pickle
import hashlib
import sqlite3

import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import numpy as np

# =====================================================
# 基本設定
# =====================================================
st.set_page_config(
    page_title="技銷金技出差經驗分享",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 技銷金技出差經驗分享｜企業知識庫 V3 Lite")

# 自動偵測 data / DATA
if os.path.exists("data"):
    DATA_DIR = "data"
elif os.path.exists("DATA"):
    DATA_DIR = "DATA"
else:
    DATA_DIR = "data"
    os.makedirs(DATA_DIR, exist_ok=True)

CACHE_DIR = "index_cache"
DB_PATH = "answer_cache.db"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

EMBEDDING_PATH = os.path.join(CACHE_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(CACHE_DIR, "metadata.pkl")

# =====================================================
# API KEY
# =====================================================
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("請先到 Streamlit Secrets 設定 GEMINI_API_KEY")
    st.stop()

genai.configure(api_key=API_KEY)

# =====================================================
# 取得可用生成模型
# =====================================================
@st.cache_resource
def get_available_models():
    try:
        models = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        return models
    except Exception as e:
        st.sidebar.error(f"取得模型失敗：{e}")
        return []

available_models = get_available_models()

MODEL_PRIORITY = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-flash-lite-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it"
]

fallback_models = [m for m in MODEL_PRIORITY if m in available_models]

for m in available_models:
    if m not in fallback_models:
        fallback_models.append(m)

if not fallback_models:
    st.error("目前沒有可用 Gemini 模型")
    st.stop()

# =====================================================
# 取得可用 Embedding 模型
# =====================================================
@st.cache_resource
def get_embedding_model():
    try:
        models = list(genai.list_models())

        usable = []
        for m in models:
            name = m.name
            methods = getattr(m, "supported_generation_methods", [])

            if "embedContent" in methods:
                usable.append(name)

        priority = [
            "models/embedding-001",
            "models/text-embedding-004"
        ]

        for p in priority:
            if p in usable:
                return p

        if usable:
            return usable[0]

    except Exception as e:
        st.sidebar.error(f"取得 Embedding 模型失敗：{e}")

    return "models/embedding-001"

EMBEDDING_MODEL = get_embedding_model()

# =====================================================
# SQLite 問答快取
# =====================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS answer_cache (
            key TEXT PRIMARY KEY,
            question TEXT,
            answer TEXT,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def get_cache(question):
    key = hashlib.md5(question.strip().encode("utf-8")).hexdigest()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT answer, model FROM answer_cache WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()

    if row:
        return row[0], row[1]

    return None, None

def save_cache(question, answer, model):
    key = hashlib.md5(question.strip().encode("utf-8")).hexdigest()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO answer_cache (key, question, answer, model)
        VALUES (?, ?, ?, ?)
    """, (key, question, answer, model))
    conn.commit()
    conn.close()

init_db()

# =====================================================
# PDF 讀取
# =====================================================
def extract_pdf_pages(file_path):
    pages = []

    try:
        reader = PdfReader(file_path)

        for i, page in enumerate(reader.pages):
            text = page.extract_text()

            if text and text.strip():
                pages.append({
                    "file": os.path.basename(file_path),
                    "page": i + 1,
                    "text": text.strip()
                })

    except Exception as e:
        st.error(f"PDF 讀取錯誤：{file_path}｜{e}")

    return pages

# =====================================================
# 切 Chunk
# =====================================================
def split_text_by_page(page_item, chunk_size=1000, overlap=200):
    text = page_item["text"]
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append({
                "file": page_item["file"],
                "page": page_item["page"],
                "text": chunk_text.strip()
            })

        start += chunk_size - overlap

    return chunks

# =====================================================
# Embedding
# =====================================================
def embed_document(text):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document"
    )
    return np.array(result["embedding"], dtype="float32")

def embed_question(text):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return np.array(result["embedding"], dtype="float32")

# =====================================================
# 建立索引
# =====================================================
def build_index(chunk_size=1000, overlap=200):
    pdf_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        st.warning(f"{DATA_DIR} 資料夾內沒有 PDF")
        return [], None

    all_chunks = []

    progress = st.progress(0)
    status = st.empty()

    for i, pdf in enumerate(pdf_files):
        status.write(f"正在讀取 PDF：{os.path.basename(pdf)}")

        pages = extract_pdf_pages(pdf)

        for page in pages:
            chunks = split_text_by_page(
                page,
                chunk_size=chunk_size,
                overlap=overlap
            )
            all_chunks.extend(chunks)

        progress.progress((i + 1) / len(pdf_files))

    if not all_chunks:
        st.warning("PDF 沒有讀取到文字，可能是掃描圖片型 PDF")
        return [], None

    vectors = []
    embedded_chunks = []

    for i, chunk in enumerate(all_chunks):
        status.write(f"正在建立 Embedding：{i + 1}/{len(all_chunks)}")

        try:
            vec = embed_document(chunk["text"])
            vectors.append(vec)
            embedded_chunks.append(chunk)
            time.sleep(0.08)

        except Exception as e:
            st.warning(f"第 {i + 1} 段 Embedding 失敗，已略過：{e}")
            continue

    if not vectors:
        st.error("沒有成功建立任何 Embedding")
        return [], None

    embeddings = np.vstack(vectors).astype("float32")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)

    np.save(EMBEDDING_PATH, embeddings)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(embedded_chunks, f)

    progress.empty()
    status.success("索引建立完成")

    return embedded_chunks, embeddings

# =====================================================
# 載入索引
# =====================================================
@st.cache_resource
def load_index():
    if not os.path.exists(EMBEDDING_PATH) or not os.path.exists(METADATA_PATH):
        return [], None

    embeddings = np.load(EMBEDDING_PATH)

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    return metadata, embeddings

# =====================================================
# 搜尋 Chunk
# =====================================================
def search_chunks(question, metadata, embeddings, top_k=50):
    q_vec = embed_question(question).astype("float32")
    q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-10)

    scores = embeddings @ q_vec

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:
        item = metadata[idx]

        results.append({
            "score": float(scores[idx]),
            "file": item["file"],
            "page": item["page"],
            "text": item["text"]
        })

    return results

# =====================================================
# Gemini 自動跳模型
# =====================================================
def generate_with_fallback(prompt_text, selected_model="AUTO"):
    if selected_model != "AUTO":
        models_to_try = [selected_model]
    else:
        models_to_try = fallback_models

    errors = []

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt_text)

            if hasattr(response, "text") and response.text:
                return response.text, model_name

            answer = ""

            if hasattr(response, "candidates"):
                for c in response.candidates:
                    if c.content:
                        for p in c.content.parts:
                            if hasattr(p, "text"):
                                answer += p.text

            if answer.strip():
                return answer, model_name

            errors.append(f"{model_name}：沒有回傳內容")

        except Exception as e:
            err = str(e)
            errors.append(f"{model_name}：{err}")

            if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
                time.sleep(2)
                continue

            continue

    return "所有模型都無法使用。\n\n錯誤紀錄：\n" + "\n\n".join(errors), None

# =====================================================
# 建立 Prompt
# =====================================================
def build_prompt(question, search_results):
    context = ""

    used_sources = []

    for i, r in enumerate(search_results, 1):
        source = f'{r["file"]}｜第 {r["page"]} 頁'

        if source not in used_sources:
            used_sources.append(source)

        context += f"""
【資料 {i}】
來源：{r["file"]}｜第 {r["page"]} 頁
相關分數：{r["score"]:.4f}

內容：
{r["text"]}
"""

    source_text = "\n".join([f"- {s}" for s in used_sources[:20]])

    prompt = f"""
你是一位技術服務專家，熟悉技銷、金技、鋁陽極染色、客戶拜訪、出差經驗整理與技術文件分析。

以下是從公司 PDF 出差文件中搜尋出的相關內容：

==============================
{context}
==============================

使用者問題：
{question}

請依照以下規則回答：

1. 優先根據 PDF 內容回答。
2. 如果 PDF 內容不足，請先寫：「目前PDF資料中沒有完整資訊。」
3. 不可以亂編 PDF 沒有的客戶名稱、公司名稱、數據、年份。
4. 若需要補充，可以用「依照一般技術服務經驗補充」的方式說明。
5. 回答請使用繁體中文。
6. 回答要完整、清楚、條列整理。
7. 如果問題與出差注意事項有關，請分成：
   - 出差前準備
   - 拜訪客戶時注意事項
   - 技術交流注意事項
   - 當地文化與溝通注意事項
   - 出差後紀錄與追蹤
8. 最後請列出「參考來源」，只能使用以下來源：
{source_text}
"""

    return prompt

# =====================================================
# Sidebar
# =====================================================
with st.sidebar:
    st.header("⚙️ 模型設定")

    model_mode = st.radio(
        "模型模式",
        ["自動跳模型", "手動指定模型"]
    )

    if model_mode == "手動指定模型":
        selected_model = st.selectbox(
            "選擇模型",
            available_models,
            index=0
        )
    else:
        selected_model = "AUTO"

    st.write("目前可用模型：")
    st.write(available_models)

    st.write("目前 Embedding 模型：")
    st.code(EMBEDDING_MODEL)

    st.divider()

    st.header("📁 文件中心")

    uploaded_files = st.file_uploader(
        "上傳 PDF",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            save_path = os.path.join(DATA_DIR, file.name)

            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

        st.success("PDF 已上傳，請重新建立索引")

    st.divider()

    chunk_size = st.slider(
        "Chunk 文字長度",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100
    )

    overlap = st.slider(
        "Chunk 重疊字數",
        min_value=50,
        max_value=500,
        value=200,
        step=50
    )

    top_k = st.slider(
        "搜尋段落數量 Top K",
        min_value=10,
        max_value=120,
        value=50,
        step=10
    )

    use_cache = st.checkbox("啟用相同問題快取", value=True)

    if st.button("🔄 重新建立索引"):
        if os.path.exists(EMBEDDING_PATH):
            os.remove(EMBEDDING_PATH)

        if os.path.exists(METADATA_PATH):
            os.remove(METADATA_PATH)

        st.cache_resource.clear()

        with st.spinner("正在重新建立索引，第一次會比較久..."):
            build_index(
                chunk_size=chunk_size,
                overlap=overlap
            )

        st.success("索引已重新建立，請重新整理頁面")

    st.divider()

    pdf_count = len([
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".pdf")
    ])

    st.write(f"PDF 資料夾：{DATA_DIR}")
    st.write(f"PDF 數量：{pdf_count}")

# =====================================================
# 載入索引
# =====================================================
metadata, embeddings = load_index()

if embeddings is None or not metadata:
    st.warning("目前尚未建立索引。請先確認 PDF 已在 data 或 DATA 資料夾，然後按左側「重新建立索引」。")
else:
    st.success(f"知識庫已載入，共 {len(metadata)} 個段落")

# =====================================================
# Session
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================================
# 顯示聊天紀錄
# =====================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =====================================================
# Chat
# =====================================================
question = st.chat_input("請輸入問題...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("AI 分析中..."):

            if embeddings is None or not metadata:
                answer = "目前尚未建立 PDF 知識庫索引，請先確認 PDF 已在 data 或 DATA 資料夾，並重新建立索引。"
                used_model = None

            else:
                if use_cache:
                    cached_answer, cached_model = get_cache(question)

                    if cached_answer:
                        answer = cached_answer
                        used_model = f"{cached_model}｜快取"
                    else:
                        results = search_chunks(
                            question,
                            metadata,
                            embeddings,
                            top_k=top_k
                        )

                        full_prompt = build_prompt(question, results)

                        answer, used_model = generate_with_fallback(
                            full_prompt,
                            selected_model=selected_model
                        )

                        if answer and used_model:
                            save_cache(question, answer, used_model)

                else:
                    results = search_chunks(
                        question,
                        metadata,
                        embeddings,
                        top_k=top_k
                    )

                    full_prompt = build_prompt(question, results)

                    answer, used_model = generate_with_fallback(
                        full_prompt,
                        selected_model=selected_model
                    )

            st.markdown(answer)

            if used_model:
                st.caption(f"本次使用模型：{used_model}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
