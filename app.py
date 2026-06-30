import os
import time
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

st.set_page_config(page_title="技銷金技出差經驗分享", page_icon="🚀", layout="wide")
st.title("🚀 技銷金技出差經驗分享 (分段讀取全搜版)")

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("錯誤：未在 Streamlit 後台設定 GEMINI_API_KEY 金鑰！")
    st.stop()

@st.cache_resource
def get_best_chat_model():
    """動態偵測支援的對話模型，徹底解決 404 錯誤"""
    chat_model = "gemini-pro" # 最安全底線
    try:
        available = [m for m in genai.list_models()]
        chat_models = [m.name for m in available if 'generateContent' in m.supported_generation_methods]
        
        # 優先順序：3.1 -> 2.5 -> 1.5 -> pro
        if "models/gemini-3.1-flash" in chat_models:
            chat_model = "gemini-3.1-flash"
        elif "models/gemini-2.5-flash" in chat_models:
            chat_model = "gemini-2.5-flash"
        elif "models/gemini-1.5-flash" in chat_models:
            chat_model = "gemini-1.5-flash"
        elif "models/gemini-pro" in chat_models:
            chat_model = "gemini-pro"
        elif chat_models:
            chat_model = chat_models[0].replace("models/", "")
    except Exception:
        pass
    return chat_model

CHAT_MODEL = get_best_chat_model()

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
def prepare_text_batches(data_dir, mod_time):
    """將所有 PDF 讀取並分批，確保每批不會超過 API 字數限制"""
    text_batches = []
    file_count = 0
    
    if not os.path.exists(data_dir):
        return text_batches, file_count
    
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    file_count = len(pdf_files)
    
    if not pdf_files:
        return text_batches, file_count
    
    current_batch = ""
    # 設定每批大約 80,000 個字元
    BATCH_LIMIT = 80000 
    
    for filename in pdf_files:
        file_path = os.path.join(data_dir, filename)
        text = extract_text_from_pdf(file_path)
        doc_str = f"\n\n---【來源檔案: {filename}】---\n{text}\n"
        
        if len(current_batch) + len(doc_str) > BATCH_LIMIT and current_batch != "":
            text_batches.append(current_batch)
            current_batch = doc_str
        else:
            current_batch += doc_str
            
    if current_batch:
        text_batches.append(current_batch)
            
    return text_batches, file_count

data_dir = "data"
current_mod_time = get_dir_mod_time(data_dir)

if "batches_initialized" not in st.session_state:
    with st.spinner("🔄 系統首次啟動：正在將所有報告整理並進行『安全分裝』，請稍候..."):
        batches, count = prepare_text_batches(data_dir, current_mod_time)
        st.session_state.text_batches = batches
        st.session_state.file_count = count
        st.session_state.batches_initialized = True
else:
    if "text_batches" not in st.session_state:
        batches, count = prepare_text_batches(data_dir, current_mod_time)
        st.session_state.text_batches = batches
        st.session_state.file_count = count

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("📁 文件中心")
    if st.session_state.file_count > 0:
        batch_num = len(st.session_state.text_batches)
        st.success(f"✅ 已讀取 {st.session_state.file_count} 份報告。\n\n系統已自動分成 **{batch_num} 批次** (使用模型: {CHAT_MODEL})")
    else:
        st.warning("⚠️ 找不到報告。")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("請輸入問題... (系統將分批閱讀所有資料，這可能需要 1~2 分鐘，請耐心等待)"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if not st.session_state.text_batches:
            st.error("系統內沒有資料。")
            st.stop()
            
        model = genai.GenerativeModel(CHAT_MODEL)
        intermediate_results = ""
        
        progress_text = f"🔍 啟動全資料分批搜尋... (使用 {CHAT_MODEL})"
        my_bar = st.progress(0, text=progress_text)
        
        total_batches = len(st.session_state.text_batches)
        
        try:
            # 第一階段：分批萃取資料
            for i, batch_text in enumerate(st.session_state.text_batches):
                my_bar.progress((i) / total_batches, text=f"🔍 正在閱讀第 {i+1} / {total_batches} 批資料... (請耐心等候)")
                
                extract_prompt = (
                    f"你是一個資料萃取助理。請在以下的【報告批次內容】中，尋找與使用者問題：「{prompt}」相關的資訊。\n"
                    "請擷取相關的段落與來源檔案名稱。如果這批資料中完全沒有相關資訊，請直接回答「NO_MATCH」，不要多說廢話。\n\n"
                    f"【報告批次內容】:\n{batch_text}"
                )
                
                response = model.generate_content(extract_prompt)
                
                if "NO_MATCH" not in response.text.upper():
                    intermediate_results += f"\n\n[來自第 {i+1} 批次的發現]:\n{response.text}"
                
                # 強制休息 4 秒，避免踩到限制
                if i < total_batches - 1:
                    time.sleep(4)
            
            my_bar.progress(1.0, text="✨ 所有資料閱讀完畢！正在為您總結最後答案...")
            time.sleep(2)
            
            # 第二階段：終極統整
            if intermediate_results.strip():
                final_prompt = (
                    f"你是一個專門分析公司內部『出差經驗』的高階助理。使用者問了：「{prompt}」。\n"
                    "以下是我讓系統分批搜尋近 200 份報告後，所撈出來的【所有相關重點碎片】。\n"
                    "請你根據這些碎片，幫我寫出一份完整、有條理的最終回答，並且務必保留並列出「資料來源檔案名稱」。\n\n"
                    f"【所有相關重點碎片】:\n{intermediate_results}"
                )
                final_response = model.generate_content(final_prompt)
                ai_reply = final_response.text
            else:
                ai_reply = f"抱歉，我剛剛翻遍了所有的檔案（共分成了 {total_batches} 批次閱讀），裡面完全找不到關於「{prompt}」的任何資料喔！"
            
            my_bar.empty()
            st.markdown(ai_reply)
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            
        except Exception as e:
            my_bar.empty()
            st.error(f"系統分段讀取時發生錯誤，請稍後再試。錯誤訊息: {str(e)}")
