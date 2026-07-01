import os
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

# -----------------------------
# Streamlit 設定
# -----------------------------
st.set_page_config(
    page_title="技銷金技出差經驗分享",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 技銷金技出差經驗分享（全文深度精讀版）")

# -----------------------------
# API KEY
# -----------------------------
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("請先到 Streamlit Secrets 設定 GEMINI_API_KEY")
    st.stop()

genai.configure(api_key=API_KEY)

# -----------------------------
# 自動取得可用模型
# -----------------------------
@st.cache_resource
def get_model():
    try:
        models = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]

        st.sidebar.write("可用模型：")
        st.sidebar.write(models)

        priority = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]

        for p in priority:
            if p in models:
                return p

        if models:
            return models[0]

    except Exception as e:
        st.sidebar.error(f"取得模型失敗：{e}")

    return None


MODEL_NAME = get_model()

if MODEL_NAME is None:
    st.error("沒有找到可用 Gemini 模型")
    st.stop()

st.sidebar.success(f"目前使用模型：{MODEL_NAME}")

# -----------------------------
# PDF 讀取
# -----------------------------
def extract_text(file):
    text = ""

    try:
        reader = PdfReader(file)

        for page in reader.pages:
            t = page.extract_text()

            if t:
                text += t + "\n"

    except Exception as e:
        st.error(f"PDF 讀取錯誤：{e}")

    return text

# -----------------------------
# Session 初始化
# -----------------------------
if "knowledge" not in st.session_state:
    st.session_state.knowledge = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "loaded" not in st.session_state:
    st.session_state.loaded = False

# -----------------------------
# 自動載入 data 資料夾 PDF
# -----------------------------
if not st.session_state.loaded:

    if os.path.exists("data"):

        pdfs = [
            f for f in os.listdir("data")
            if f.lower().endswith(".pdf")
        ]

        for pdf in pdfs:
            path = os.path.join("data", pdf)
            text = extract_text(path)

            if text.strip():
                st.session_state.knowledge += "\n\n"
                st.session_state.knowledge += f"===== {pdf} =====\n"
                st.session_state.knowledge += text

        if pdfs:
            st.sidebar.success(f"已自動載入 {len(pdfs)} 個 PDF")
        else:
            st.sidebar.warning("data 資料夾內沒有 PDF 檔案")

    else:
        st.sidebar.warning("尚未建立 data 資料夾")

    st.session_state.loaded = True

# -----------------------------
# Sidebar 文件中心
# -----------------------------
with st.sidebar:

    st.header("文件中心")

    files = st.file_uploader(
        "額外上傳 PDF",
        type=["pdf"],
        accept_multiple_files=True
    )

    if files:

        for f in files:
            uploaded_text = extract_text(f)

            if uploaded_text.strip():
                st.session_state.knowledge += "\n\n"
                st.session_state.knowledge += f"===== {f.name} =====\n"
                st.session_state.knowledge += uploaded_text

        st.success("新增成功")

    st.divider()

    st.write("目前知識庫字數：")
    st.write(len(st.session_state.knowledge))

# -----------------------------
# 顯示聊天紀錄
# -----------------------------
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Chat
# -----------------------------
prompt = st.chat_input("請輸入問題...")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("AI 分析中..."):

            MAX_CHARS = 800000
            current_knowledge = st.session_state.knowledge[-MAX_CHARS:]

            system_prompt = f"""
你是一位技術服務專家。
以下是公司所有技術出差文件。

==============================
{current_knowledge}
==============================

請依照以上內容回答。

回答規則：
1. 如果文件中有答案，請優先根據 PDF 內容回答。
2. 如果文件中沒有答案，請先回答：「目前PDF資料中沒有相關資訊。」
3. 若需要補充，請再依照你的專業知識補充。
4. 回答請使用繁體中文。
5. 回答請盡量完整，並使用條列整理。
"""

            model = genai.GenerativeModel(MODEL_NAME)

            try:
                response = model.generate_content(
                    contents=system_prompt + "\n\n使用者問題：\n" + prompt
                )

                if hasattr(response, "text"):
                    answer = response.text
                elif hasattr(response, "candidates"):
                    answer = ""

                    for c in response.candidates:
                        if c.content:
                            for p in c.content.parts:
                                if hasattr(p, "text"):
                                    answer += p.text
                else:
                    answer = "AI 沒有回傳任何內容。"

            except Exception as e:
                answer = f"AI 發生錯誤：{str(e)}"

            st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )
