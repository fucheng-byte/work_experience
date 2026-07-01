import os
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

# -----------------------------
# Streamlit設定
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
except:
    st.error("請先到 Streamlit Secrets 設定 GEMINI_API_KEY")
    st.stop()

genai.configure(api_key=API_KEY)

# -----------------------------
# 自動取得可用模型
# -----------------------------
@st.cache_resource
def get_model():

    try:

        models = []

        for m in genai.list_models():

            if "generateContent" in m.supported_generation_methods:

                models.append(m.name)

        st.sidebar.success("成功取得模型")

        st.sidebar.write(models)

        priority = [

            "models/gemini-2.5-flash",

            "models/gemini-2.5-pro",

            "models/gemini-2.0-flash",

            "models/gemini-1.5-flash"

        ]

        for p in priority:

            if p in models:

                return p.replace("models/","")

        if len(models)>0:

            return models[0].replace("models/","")

    except Exception as e:

        st.error(e)

    return None

MODEL_NAME = get_model()

if MODEL_NAME is None:

    st.error("沒有找到可用 Gemini 模型")

    st.stop()

st.sidebar.success(f"目前使用模型：{MODEL_NAME}")

model = genai.GenerativeModel(MODEL_NAME)

# -----------------------------
# PDF讀取
# -----------------------------
def extract_text(file):

    text=""

    try:

        reader=PdfReader(file)

        for page in reader.pages:

            t=page.extract_text()

            if t:

                text+=t+"\n"

    except Exception as e:

        st.error(e)

    return text

# -----------------------------
# Session
# -----------------------------
if "knowledge" not in st.session_state:

    st.session_state.knowledge=""

if "messages" not in st.session_state:

    st.session_state.messages=[]

# -----------------------------
# 自動載入data資料夾
# -----------------------------
if "loaded" not in st.session_state:

    st.session_state.loaded=False

if not st.session_state.loaded:

    if os.path.exists("data"):

        pdfs=[

            f for f in os.listdir("data")

            if f.lower().endswith(".pdf")

        ]

        for pdf in pdfs:

            path=os.path.join("data",pdf)

            st.session_state.knowledge+=extract_text(path)

    st.session_state.loaded=True

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("文件中心")

    st.success("已自動載入 data 資料夾")

    files=st.file_uploader(

        "額外上傳PDF",

        type=["pdf"],

        accept_multiple_files=True

    )

    if files:

        for f in files:

            st.session_state.knowledge+=extract_text(f)

        st.success("新增成功")

# -----------------------------
# 顯示聊天紀錄
# -----------------------------
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# -----------------------------
# Chat
# -----------------------------
prompt=st.chat_input("請輸入問題...")

if prompt:

    st.session_state.messages.append(

        {

            "role":"user",

            "content":prompt

        }

    )

    with st.chat_message("user"):

        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("AI分析中..."):

            system_prompt=f"""

你是一位技術服務專家。

以下是公司所有技術出差文件。

==============================

{st.session_state.knowledge}

==============================

請依照以上內容回答。

如果文件中沒有答案，

請先回答：

"目前PDF資料中沒有相關資訊。"

然後再依照你的專業知識補充。

回答請盡量完整。

使用繁體中文。

使用條列整理。

"""

            try:

                response=model.generate_content(

                    system_prompt+"\n\n問題："+prompt

                )

                answer=response.text

            except Exception as e:

                answer=f"AI發生錯誤：{e}"

            st.markdown(answer)

            st.session_state.messages.append(

                {

                    "role":"assistant",

                    "content":answer

                }

            )
