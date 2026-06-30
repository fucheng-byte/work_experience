import os
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

st.set_page_config(page_title="技銷金技出差經驗分享", page_icon="🚀", layout="wide")
st.title("🚀 技銷金技出差經驗分享 (暴力全資料搜索版)")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("錯誤：未在 Streamlit 後台設定 GEMINI_API_KEY 金鑰！")
    st.stop()

# 選擇支援超大上下文的模型 (Gemini 1.5 Pro / Flash 支援高達 100萬~200萬 Token)
CHAT_MODEL = "gemini-1.5-pro" 

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

def get_dir_mod_time(data_dir):
    if not os.path.exists(data_dir):
        return 0
    files = os.listdir(data_dir)
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    if not pdf_files:
        return 0
    return max(os.path.getmtime(os.path.join(data_dir, f)) for f in pdf_files)

@st.cache_data(show_spinner=False)
def load_all_documents(data_dir, mod_time):
    """【全資料搜索核心】：不切碎、不轉換，直接把所有檔案讀成一篇超級無敵長的文字檔"""
    all_text_content = ""
    file_count = 0
    
    if not os.path.exists(data_dir):
        return all_text_content, file_count
    
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    file_count = len(pdf_files)
    
    if not pdf_files:
        return all_text_content, file_count
    
    for filename in pdf_files:
        file_path = os.path.join(data_dir, filename)
        text = extract_text_from_pdf(file_path)
        # 把檔名跟內容完整拼接
        all_text_content += f"\n\n============================\n"
        all_text_content += f"【來源檔案】：{filename}\n"
        all_text_content += f"============================\n{text}\n"
            
    return all_text_content, file_count

data_dir = "data"
current_mod_time = get_dir_mod_time(data_dir)

if "full_database" not in st.session_state:
    with st.spinner("🔄 正在讀取所有文件內容（正在將 100% 的資料載入記憶體，請稍候...）"):
        all_text, count = load_all_documents(data_dir, current_mod_time)
        st.session_state.full_database = all_text
        st.session_state.file_count = count
        st.session_state.db_initialized = True
else:
    if "full_database" not in st.session_state:
        all_text, count = load_all_documents(data_dir, current_mod_time)
        st.session_state.full_database = all_text
        st.session_state.file_count = count

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("📁 文件中心")
    if st.session_state.file_count > 0:
        st.success(f"✅ 系統已將 {st.session_state.file_count} 份報告的「完整內容」準備完畢，每次將進行 100% 全文比對。")
    else:
        st.warning("⚠️ 找不到報告。")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("請輸入問題... AI 將會閱讀全部 197 份報告來回答您！"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if not st.session_state.full_database:
            st.error("系統內沒有任何資料可以讀取。")
            st.stop()
            
        with st.spinner(f"🔍 AI 正在從頭到尾閱讀所有資料... (如果字數過多可能會跳出 429 錯誤)"):
            try:
                # 建立 AI 模型 (使用最能處理長文的 1.5-pro)
                model = genai.GenerativeModel(CHAT_MODEL)
                
                # 【全資料搜索】：直接把幾十萬字全部塞進提示詞
                full_prompt = (
                    "你是一個專門分析公司內部『技銷金技出差經驗與廠商資料』的高階 AI 智能助理。\n"
                    "我將提供公司【所有的出差報告完整內容】給你。請你務必「從頭到尾」仔細閱讀所有的內容，絕對不可以遺漏任何一份檔案。\n"
                    "請根據這些內容，全面、精準且毫無遺漏地回答使用者的問題。回答時，請務必標示資料是來自哪一份檔案（例如：130131 江富誠...報告）。\n\n"
                    f"【公司所有出差報告完整內容】:\n{st.session_state.full_database}\n\n"
                    f"【使用者問題】: {prompt}"
                )
                
                response = model.generate_content(full_prompt)
                ai_reply = response.text
                
                st.markdown(ai_reply)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                
            except Exception as e:
                # 攔截 429 錯誤並給予白話文提示
                error_msg = str(e)
                if "429" in error_msg:
                    st.error("🚨 發生 429 錯誤！這代表您的 197 份報告總字數「超過了 Google 免費版金鑰的極限」。您必須刪減 data 資料夾內的檔案數量，或升級付費版 Google API。")
                else:
                    st.error(f"系統呼叫 AI 時發生錯誤: {error_msg}")
