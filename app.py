import os
import time
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

st.set_page_config(page_title="技銷金技出差經驗分享", page_icon="🚀", layout="wide")
st.title("🚀 技銷金技出差經驗分享 (全文深度精讀版)")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("錯誤：未在 Streamlit 後台設定 GEMINI_API_KEY 金鑰！")
    st.stop()

@st.cache_resource
def get_best_models():
    """自動偵測最適合處理超長全文的對話模型"""
    try:
        available = [m for m in genai.list_models()]
        chat_models = [m.name for m in available if 'generateContent' in m.supported_generation_methods]
        
        primary_chat = "gemini-2.5-flash"
        if "models/gemini-2.5-flash" in chat_models: 
            primary_chat = "gemini-2.5-flash"
        elif "models/gemini-1.5-flash" in chat_models: 
            primary_chat = "gemini-1.5-flash"
        else:
            primary_chat = chat_models[0].replace("models/", "") if chat_models else "gemini-1.5-flash"
            
        fallback_chat = "gemini-1.5-flash" if "models/gemini-1.5-flash" in chat_models else primary_chat
        return primary_chat, fallback_chat
    except Exception:
        return "gemini-2.5-flash", "gemini-1.5-flash"

PRIMARY_CHAT, FALLBACK_CHAT = get_best_models()

def extract_text_from_pdf(pdf_file):
    """讀取單一 PDF 檔案的文字"""
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    except Exception as e:
        st.error(f"讀取檔案時發生錯誤: {e}")
    return text

# ==========================================
# 4. 初始化知識庫與對話紀錄 (Session State)
# ==========================================
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 自動載入 GitHub 中 data 資料夾的 PDF 檔案 ---
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if not st.session_state.data_loaded:
    data_dir = "data"
    auto_text = ""
    if os.path.exists(data_dir):
        pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
        if pdf_files:
            for filename in pdf_files:
                file_path = os.path.join(data_dir, filename)
                auto_text += extract_text_from_pdf(file_path)
            st.session_state.knowledge_base += auto_text
    st.session_state.data_loaded = True

# ==========================================
# 5. 側邊欄設計 (文件中心)
# ==========================================
with st.sidebar:
    st.subheader("文件中心")
    st.info("已自動載入 data 資料夾中的檔案")
    
    st.subheader("手動額外上傳資料")
    uploaded_files = st.file_uploader("Upload", accept_multiple_files=True, type=['pdf'])
    
    if uploaded_files:
        with st.spinner("正在讀取上傳的文件..."):
            uploaded_text = ""
            for f in uploaded_files:
                uploaded_text += extract_text_from_pdf(f)
            # 將新上傳的文字追加到知識庫中
            st.session_state.knowledge_base += uploaded_text
            st.success("額外文件載入成功！")

# ==========================================
# 6. 聊天對話介面
# ==========================================
# 顯示歷史對話紀錄
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 底部聊天輸入框
if prompt := st.chat_input("請輸入問題..."):
    # 顯示使用者的問題
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 呼叫 Gemini 產生回答
    with st.chat_message("assistant"):
        with st.spinner("AI 正在思考中..."):
            try:
                # 建立最新的 Gemini 模型 (使用適合大上下文的 1.5-flash)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # 組合 Prompt，強制 AI 必須根據知識庫回答
                full_prompt = (
                    "你是一個專門分析『技銷金技出差經驗』的 AI 智能助理。\n"
                    "請嚴格根據以下提供的【出差經驗知識庫】內容來回答使用者的問題。\n"
                    "如果問題在知識庫中完全找不到線索，請客氣地回答：'抱歉，目前的出差經驗知識庫中沒有記載相關資料。'，並根據你的通用知識給予基本建議。\n\n"
                    f"【出差經驗知識庫內容】:\n{st.session_state.knowledge_base}\n\n"
                    f"【使用者問題】: {prompt}"
                )
                
                response = model.generate_content(full_prompt)
                ai_reply = response.text
                
                st.markdown(ai_reply)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                
            except Exception as e:
                st.error(f"系統呼叫 AI 時發生錯誤，請檢查您的 API Key 是否正確。錯誤訊息: {e}")
