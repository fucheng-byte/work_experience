import os
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import numpy as np

# ==========================================
# 1. 網頁基本設定
# ==========================================
st.set_page_config(
    page_title="技銷金技出差經驗分享", 
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 技銷金技出差經驗分享 (RAG 高階智慧檢索版)")

# ==========================================
# 2. 初始化 Gemini AI 金鑰
# ==========================================
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("錯誤：未在 Streamlit 後台設定 GEMINI_API_KEY 金鑰！")

# ==========================================
# 3. 核心工具函式：PDF 讀取、文字切片、向量計算
# ==========================================
def extract_text_from_pdf(pdf_file):
    """讀取單一 PDF 檔案的純文字"""
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    except Exception as e:
        pass
    return text

def split_text(text, chunk_size=800, chunk_overlap=150):
    """將長文本切成適當大小的文字區塊，避免斷句問題"""
    chunks = []
    if not text:
        return chunks
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += chunk_size - chunk_overlap
    return chunks

def cosine_similarity(v1, v2):
    """計算兩個向量之間的餘弦相似度 (比對關聯性)"""
    dot_prod = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_prod / (norm1 * norm2)

def get_dir_mod_time(data_dir):
    """獲取資料夾內檔案的最後修改時間，用來判斷是否需要更新快取"""
    if not os.path.exists(data_dir):
        return 0
    files = os.listdir(data_dir)
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    if not pdf_files:
        return 0
    return max(os.path.getmtime(os.path.join(data_dir, f)) for f in pdf_files)

# ==========================================
# 4. 智慧快取機制：為數百份檔案建立向量索引
# ==========================================
@st.cache_data(show_spinner=False)
def build_knowledge_base(data_dir, mod_time):
    """掃描 data 資料夾，將所有 PDF 切片並轉換成 AI 向量資料庫"""
    database = []
    if not os.path.exists(data_dir):
        return database
    
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        return database
    
    all_chunks = []
    # 逐一讀取 PDF 檔案並切碎
    for filename in pdf_files:
        file_path = os.path.join(data_dir, filename)
        text = extract_text_from_pdf(file_path)
        chunks = split_text(text)
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": filename})
    
    if not all_chunks:
        return database
    
    # 分批將文字丟給 Google 轉換成向量 (Embedding)
    texts_to_embed = [c["text"] for c in all_chunks]
    batch_size = 50
    embeddings = []
    
    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i+batch_size]
        try:
            res = genai.embed_content(model="models/text-embedding-004", content=batch)
            embeddings.extend(res['embedding'])
        except Exception as e:
            pass
            
    # 將向量與文字重新打包
    for chunk, emb in zip(all_chunks, embeddings):
        chunk["embedding"] = emb
        database.append(chunk)
        
    return database

# ==========================================
# 5. 初始化與載入資料庫
# ==========================================
data_dir = "data"
current_mod_time = get_dir_mod_time(data_dir)

# 確保只在網頁首次啟動或資料夾有變動時才顯示大載入畫面
if "db_initialized" not in st.session_state:
    with st.spinner("🔄 正在初始化企業級知識庫（正在為 200+ 份文件建立高階檢索索引，請稍候...）"):
        st.session_state.vector_db = build_knowledge_base(data_dir, current_mod_time)
        st.session_state.db_initialized = True
else:
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = build_knowledge_base(data_dir, current_mod_time)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 6. 側邊欄設計 (文件中心)
# ==========================================
with st.sidebar:
    st.subheader("📁 文件中心")
    file_count = len([f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]) if os.path.exists(data_dir) else 0
    st.success(f"✅ 系統已自動索引 data 資料夾內 {file_count} 份出差報告")
    
    st.subheader("📥 臨時額外追加檔案")
    uploaded_files = st.file_uploader("上傳新的 PDF（僅供本次對話查詢）", accept_multiple_files=True, type=['pdf'])
    
    if uploaded_files:
        if "uploaded_files_processed" not in st.session_state:
            st.session_state.uploaded_files_processed = set()
        
        new_chunks = []
        for f in uploaded_files:
            if f.name not in st.session_state.uploaded_files_processed:
                with st.spinner(f"正在即時索引新文件: {f.name}..."):
                    text = extract_text_from_pdf(f)
                    chunks = split_text(text)
                    for chunk in chunks:
                        new_chunks.append({"text": chunk, "source": f.name})
                    st.session_state.uploaded_files_processed.add(f.name)
        
        if new_chunks:
            texts_to_embed = [c["text"] for c in new_chunks]
            try:
                res = genai.embed_content(model="models/text-embedding-004", content=texts_to_embed)
                for chunk, emb in zip(new_chunks, res['embedding']):
                    chunk["embedding"] = emb
                    st.session_state.vector_db.append(chunk)
                st.sidebar.success(f"成功額外載入 {len(uploaded_files)} 個臨時檔案！")
            except Exception as e:
                st.sidebar.error(f"臨時檔案建立索引失敗: {e}")

# ==========================================
# 7. 聊天對話介面與 RAG 搜尋核心
# ==========================================
# 顯示歷史對話紀錄
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 底部聊天輸入框
if prompt := st.chat_input("請輸入關於出差經驗或客戶的問題..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🔍 正在海量檔案中進行智慧檢索與分析..."):
            try:
                # 🛠️ RAG 步驟 1: 將使用者的問題轉換成向量
                q_res = genai.embed_content(model="models/text-embedding-004", content=[prompt])
                q_emb = q_res['embedding'][0]
                
                # 🛠️ RAG 步驟 2: 計算問題與資料庫中所有文字塊的相似度
                scored_chunks = []
                for chunk in st.session_state.vector_db:
                    score = cosine_similarity(q_emb, chunk["embedding"])
                    scored_chunks.append((score, chunk))
                
                # 依相似度由高到低排序
                scored_chunks.sort(key=lambda x: x[0], reverse=True)
                
                # 🛠️ RAG 步驟 3: 篩選出最相關的前 6 個文字區塊 (Top 6)
                top_k = scored_chunks[:6]
                
                context_str = ""
                sources = set()
                for score, chunk in top_k:
                    # 相似度大於 0.3 的才納入（避免不相關的雜訊）
                    if score > 0.3:
                        context_str += f"[來源檔案: {chunk['source']}]\n{chunk['text']}\n\n"
                        sources.add(chunk['source'])
                
                # 🛠️ RAG 步驟 4: 打包精華內容，交給最適合閱讀長文的 gemini-1.5-flash
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                if context_str:
                    full_prompt = (
                        "你是一個專門分析公司內部『技銷金技出差經驗與廠商資料』的高階 AI 智能助理。\n"
                        "請根據以下從數百份檔案中精確檢索出來的【相關知識庫段落】內容，全面、客觀且準確地回答使用者的問題。\n"
                        "在回答時，請務必主動提及你是從哪些來源檔案（例如：130131 江富誠...報告）中找到這些資訊的，以便使用者核對。\n"
                        "如果檢索出的內容與問題無關，或者在裡面完全找不到答案，請回答：'抱歉，從目前的出差報告中找不到直接相關的紀錄。'，並根據你的通用知識給予專業建議。\n\n"
                        f"【精確檢索的知識庫段落】:\n{context_str}\n"
                        f"【使用者問題】: {prompt}"
                    )
                else:
                    full_prompt = (
                        "你是一個專門分析公司內部『技銷金技出差經驗與廠商資料』的高階 AI 智能助理。\n"
                        "目前系統在數百份檔案中沒有檢索到與使用者問題直接相關的段落。\n"
                        "請客氣地回答：'抱歉，目前的出差經驗知識庫中沒有檢索到直接相關的資料。'，並根據你的通用知識給予相關的商務或出差基本建議。\n\n"
                        f"【使用者問題】: {prompt}"
                    )
                    
                response = model.generate_content(full_prompt)
                ai_reply = response.text
                
                # 🛠️ RAG 步驟 5: 如果有找到資料，在回答最下方附上精美的參考文獻標註
                if sources:
                    ai_reply += "\n\n---\n**🔍 本次回答參考的原始檔案：**\n" + "\n".join([f"- 📄 {s}" for s in sources])
                
                st.markdown(ai_reply)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                
            except Exception as e:
                st.error(f"系統檢索或呼叫 AI 時發生錯誤，請稍後重試。錯誤訊息: {e}")
