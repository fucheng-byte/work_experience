import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

# 1. 設定 Gemini API 金鑰（這要在 Streamlit Cloud 的 Secrets 設定）
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# 2. 建立一個讀取 PDF 文字的 function
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# 3. 網頁側邊欄上傳
with st.sidebar:
    st.subheader("文件中心")
    uploaded_files = st.file_uploader("上傳知識庫 PDF", accept_multiple_files=True, type=['pdf'])
    
    if uploaded_files:
        # 讀取 PDF 內容
        with st.spinner("正在讀取雲端/上傳的文件..."):
            raw_text = get_pdf_text(uploaded_files)
            st.success("知識庫載入成功！")

# 4. 當使用者輸入問題時，把 raw_text (知識庫) 跟問題一起餵給 Gemini
if prompt := st.chat_input("請輸入問題..."):
    # 這裡組合 Prompt（提示詞強化）
    full_prompt = f"請根據以下知識庫內容回答問題：\n\n【知識庫內容】:\n{raw_text}\n\n【使用者問題】: {prompt}"
    
    # 呼叫 Gemini 模型的程式碼...