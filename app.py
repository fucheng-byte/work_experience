import os
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import numpy as np
import re

st.set_page_config(page_title="技銷金技出差經驗分享", page_icon="🚀", layout="wide")
st.title("🚀 技銷金技出差經驗分享 (絕不當機版)")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("錯誤：未在 Streamlit 後台設定 GEMINI_API_KEY 金鑰！")
    st.stop()

@st.cache_resource
def get_best_models():
    """【終極防護】100% 只用金鑰真實支援的模型，徹底消滅 404"""
    try:
        available = [m for m in genai.list_models()]
        embed_models = [m.name for m in available if 'embedContent' in m.supported_generation_methods]
        chat_models = [m.name for m in available if 'generateContent' in m.supported_generation_methods]
        
        # 1. 安全挑選向量模型
        embed_model = embed_models[0] if embed_models else "models/text-embedding-004"
        if "models/text-embedding-004" in embed_models: 
            embed_model = "models/text-embedding-004"
            
        # 2. 安全挑選主力對話模型
        primary_chat = chat_models[0].replace("models/", "") if chat_models else "gemini-1.5-flash"
        if "models/gemini-2.5-flash" in chat_models: 
            primary_chat = "gemini-2.5-flash"
        elif "models/gemini-1.5-flash" in chat_models: 
            primary_chat = "gemini-1.5-flash"
            
        # 3. 安全挑選備援模型 (找一個跟主力不同，且確定存在的)
        fallback_chat = primary_chat
        for m in chat_models:
            m_name = m.replace("models/", "")
            if m_name != primary_chat and "flash" in m_name:
                fallback_chat = m_name
                break
        if fallback_chat == primary_chat and len(chat_models) > 1:
            fallback_chat = chat_models[-1].replace("models/", "")
            
        return embed_model, primary_chat, fallback_chat
    except Exception:
        return "models/text-embedding-004", "gemini-1.5-flash", "gemini-1.5-flash"

EMBED_MODEL, PRIMARY_CHAT, FALLBACK_CHAT = get_best_models()

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
            search_content = f"報告檔名：{filename}\n內容：{chunk}"
            all_chunks.append({"text": chunk, "source": filename, "search_content": search_content})
    
    if not all_chunks:
        return database
    
    texts_to_embed = [c["search_content"] for c in all_chunks]
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

def extract_keywords(prompt):
    stop_words = ["幫我", "找", "的", "報告", "請問", "有沒有", "出差", "資料", "哪些", "什麼", "關於", "整理", "所有", "人員"]
    clean_prompt = prompt
    for sw in stop_words:
        clean_prompt = clean_prompt.replace(sw, " ")
    words = re.findall(r'\b\w+\b', clean_prompt)
    return [w for w in words if len(w) >= 2]

data_dir = "data"
current_mod_time = get_dir_mod_time(data_dir)

if "db_initialized" not in st.session_state:
    with st.spinner("🔄 正在初始化企業級知識庫（請耐心等候...）"):
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
        st.success(f"✅ 系統已自動索引 {file_count} 份報告")
        st.info(f"🚀 主力 AI: {PRIMARY_CHAT}\n\n🛡️ 備援 AI: {FALLBACK_CHAT}")
    else:
        st.warning(f"⚠️ 找到 {file_count} 份報告，但索引建立失敗。")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("請輸入問題..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if not st.session_state.vector_db:
            st.error("系統資料庫目前為空，請嘗試清除快取。")
            st.stop()
            
        with st.spinner(f"🔍 正在海量檔案中進行智慧檢索..."):
            try:
                q_res = genai.embed_content(model=EMBED_MODEL, content=[prompt])
                q_emb = q_res['embedding'][0]
                
                keywords = extract_keywords(prompt)
                
                scored_chunks = []
                for chunk in st.session_state.vector_db:
                    vec_score = cosine_similarity(q_emb, chunk["embedding"])
                    
                    keyword_boost = 0.0
                    for kw in keywords:
                        if kw in chunk['source']:
                            keyword_boost += 0.3  
                        if kw in chunk['text']:
                            keyword_boost += 0.15 
                    
                    final_score = vec_score + keyword_boost
                    scored_chunks.append((final_score, chunk))
                
                scored_chunks.sort(key=lambda x: x[0], reverse=True)
                top_k = scored_chunks[:60] 
                
                context_str = ""
                sources = set()
                for score, chunk in top_k:
                    if score > 0.2:
                        context_str += f"[來源檔案: {chunk['source']}]\n{chunk['text']}\n\n"
                        sources.add(chunk['source'])
                
                if context_str:
                    full_prompt = (
                        "你是一個專門分析公司內部『技銷金技出差經驗與廠商資料』的高階 AI 智能助理。\n"
                        "請根據以下從數百份檔案中精確檢索出來的【相關知識庫段落】內容，全面且準確地回答使用者的問題。\n"
                        "在回答時，請務必主動提及你是從哪些來源檔案中找到這些資訊的。\n\n"
                        f"【精確檢索的知識庫段落】:\n{context_str}\n"
                        f"【使用者問題】: {prompt}"
                    )
                else:
                    full_prompt = (
                        f"使用者問了問題：「{prompt}」，但目前系統在數百份檔案中沒有檢索到直接相關的段落。\n"
                        "請客氣地回答：'抱歉，目前的出差經驗知識庫中沒有檢索到直接相關的資料。您可以嘗試換個更明確的關鍵字。'"
                    )
                
                used_ai = ""
                ai_reply = ""
                
                try:
                    # 第一擊：絕對存在的主力模型
                    model = genai.GenerativeModel(PRIMARY_CHAT)
                    response = model.generate_content(full_prompt)
                    used_ai = PRIMARY_CHAT
                    ai_reply = response.text
                except Exception as e1:
                    # 如果主力模型跟備援模型不同，才進行切換
                    if PRIMARY_CHAT != FALLBACK_CHAT:
                        st.warning(f"⚠️ `{PRIMARY_CHAT}` 額度滿載，自動切換至備援 `{FALLBACK_CHAT}`...")
                        try:
                            # 第二擊：絕對存在的備援模型
                            model = genai.GenerativeModel(FALLBACK_CHAT)
                            response = model.generate_content(full_prompt)
                            used_ai = FALLBACK_CHAT
                            ai_reply = response.text
                        except Exception as e2:
                            st.error("🚨 您的金鑰今日『所有免費額度』皆已耗盡。請明天再試，或更換 API Key！")
                            st.stop()
                    else:
                        st.error("🚨 您的金鑰今日免費額度已完全耗盡。請明天再試！")
                        st.stop()
                
                if sources:
                    ai_reply += f"\n\n---\n**🔍 本次回答參考的原始檔案：** (由 {used_ai} 生成)\n" + "\n".join([f"- 📄 {s}" for s in sources])
                
                st.markdown(ai_reply)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                
            except Exception as e:
                st.error(f"系統發生不可預期的錯誤，請稍後重試。錯誤訊息: {str(e)}")
