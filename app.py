import os
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import numpy as np

st.set_page_config(page_title="技銷金技出差經驗分享", page_icon="🚀", layout="wide")
st.title("🚀 技銷金技出差經驗分享 (RAG 高階智慧檢索版)")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("錯誤：未在 Streamlit 後台設定 GEMINI_API_KEY 金鑰！")
    st.stop()

@st.cache_resource
def get_best_models():
    """動態偵測 API 金鑰支援的模型，徹底解決所有 404 錯誤"""
    embed_model = "models/embedding-001" 
    chat_model = "gemini-pro" # 最保守的安全牌對話模型
    try:
        available = [m for m in genai.list_models()]
        
        # 1. 尋找支援的向量檢索模型
        embed_models = [m.name for m in available if 'embedContent' in m.supported_generation_methods]
        if "models/text-embedding-004" in embed_models:
            embed_model = "models/text-embedding-004"
        elif "models/embedding-001" in embed_models:
            embed_model = "models/embedding-001"
        elif embed_models:
            embed_model = embed_models[0]
            
        # 2. 尋找支援的對話生成模型
        chat_models = [m.name for m in available if 'generateContent' in m.supported_generation_methods]
        if "models/gemini-1.5-flash" in chat_models:
            chat_model = "gemini-1.5-flash"
        elif "models/gemini-1.5-pro" in chat_models:
            chat_model = "gemini-1.5-pro"
        elif "models/gemini-pro" in chat_models:
            chat_model = "gemini-pro"
        elif chat_models:
            chat_model = chat_models[0].replace("models/", "")
    except Exception:
        pass
    return embed_model, chat_model

EMBED_MODEL, CHAT_MODEL = get_best_models()

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    except Exception:
        pass
    return text

def split_text(text, chunk_size=800, chunk_overlap=150):
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
    dot_prod = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_prod / (norm1 * norm2)

def get_dir_mod_time(data_dir):
    if not os.path.exists(data_dir):
        return 0
    files = os.listdir(data_dir)
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    if not pdf_files:
        return 0
    return max(os.path.getmtime(os.path.join(data_dir, f)) for f in pdf_files)

@st.cache_data(show_spinner=False)
def build_knowledge_base(data_dir, mod_time):
    database = []
    if not os.path.exists(data_dir):
        return database
    
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        return database
    
    all_chunks = []
    for filename in pdf_files:
        file_path = os.path.join(data_dir, filename)
        text = extract_text_from_pdf(file_path)
        chunks = split_text(text)
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": filename})
    
    if not all_chunks:
        return database
    
    texts_to_embed = [c["text"] for c in all_chunks]
    batch_size = 50
    embeddings = []
    
    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i+batch_size]
        try:
            res = genai.embed_content(model=EMBED_MODEL, content=batch)
            embeddings.extend(res['embedding'])
        except Exception:
            for _ in batch:
                embeddings.append([0.0]*768)
            
    for chunk, emb in zip(all_chunks, embeddings):
        if any(v != 0.0 for v in emb): 
            chunk["embedding"] = emb
            database.append(chunk)
            
    return database

data_dir = "data"
current_mod_time = get_dir_mod_time(data_dir)

if "db_initialized" not in st.session_state:
    with st.spinner("🔄 正在初始化企業級知識庫（正在讀取 197 份文件，這可能需要幾分鐘...）"):
        st.session_state.vector_db = build_knowledge_base(data_dir, current_mod_time)
        st.session_state.db_initialized = True
else:
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = build_knowledge_base(data_dir, current_mod_time)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("📁 文件中心")
    file_count = len([f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]) if os.path.exists(data_dir) else 0
    
    if "vector_db" in st.session_state and len(st.session_state.vector_db) > 0:
        st.success(f"✅ 系統已自動索引 {file_count} 份報告\n(對話 AI: {CHAT_MODEL})")
    else:
        st.warning(f"⚠️ 找到 {file_count} 份報告，但索引建立失敗或還在處理中。")
        
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
                res = genai.embed_content(model=EMBED_MODEL, content=texts_to_embed)
                for chunk, emb in zip(new_chunks, res['embedding']):
                    chunk["embedding"] = emb
                    st.session_state.vector_db.append(chunk)
                st.sidebar.success(f"成功額外載入 {len(uploaded_files)} 個臨時檔案！")
            except Exception as e:
                st.sidebar.error(f"臨時檔案建立索引失敗: {e}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("請輸入關於出差經驗或客戶的問題..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if not st.session_state.vector_db:
            st.error("系統資料庫目前為空或初始化失敗，請嘗試清除快取。")
            st.stop()
            
        with st.spinner(f"🔍 正在海量檔案中進行智慧檢索... (使用 {CHAT_MODEL})"):
            try:
                q_res = genai.embed_content(model=EMBED_MODEL, content=[prompt])
                q_emb = q_res['embedding'][0]
                
                scored_chunks = []
                for chunk in st.session_state.vector_db:
                    score = cosine_similarity(q_emb, chunk["embedding"])
                    scored_chunks.append((score, chunk))
                
                scored_chunks.sort(key=lambda x: x[0], reverse=True)
                top_k = scored_chunks[:6]
                
                context_str = ""
                sources = set()
                for score, chunk in top_k:
                    if score > 0.3:
                        context_str += f"[來源檔案: {chunk['source']}]\n{chunk['text']}\n\n"
                        sources.add(chunk['source'])
                
                model = genai.GenerativeModel(CHAT_MODEL)
                
                if context_str:
                    full_prompt = (
                        "你是一個專門分析公司內部『技銷金技出差經驗與廠商資料』的高階 AI 智能助理。\n"
                        "請根據以下從數百份檔案中精確檢索出來的【相關知識庫段落】內容，全面、客觀且準確地回答使用者的問題。\n"
                        "在回答時，請務必主動提及你是從哪些來源檔案中找到這些資訊的，以便使用者核對。\n"
                        "如果檢索出的內容與問題無關，或者在裡面完全找不到答案，請回答：'抱歉，從目前的出差報告中找不到直接相關的紀錄。'\n\n"
                        f"【精確檢索的知識庫段落】:\n{context_str}\n"
                        f"【使用者問題】: {prompt}"
                    )
                else:
                    full_prompt = (
                        "你是一個專門分析公司內部『技銷金技出差經驗與廠商資料』的高階 AI 智能助理。\n"
                        "目前系統在數百份檔案中沒有檢索到與使用者問題直接相關的段落。\n"
                        "請客氣地回答：'抱歉，目前的出差經驗知識庫中沒有檢索到直接相關的資料。'\n\n"
                        f"【使用者問題】: {prompt}"
                    )
                    
                response = model.generate_content(full_prompt)
                ai_reply = response.text
                
                if sources:
                    ai_reply += "\n\n---\n**🔍 本次回答參考的原始檔案：**\n" + "\n".join([f"- 📄 {s}" for s in sources])
                
                st.markdown(ai_reply)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                
            except Exception as e:
                st.error(f"系統檢索或呼叫 AI 時發生錯誤，請稍後重試。錯誤訊息: {e}")
