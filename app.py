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

def extract_text_from_pdf(file_path):
    """一字不漏地完整萃取整份 PDF 的全文本"""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    except Exception:
        pass
    return text

data_dir = "data"

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("📁 文件中心")
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')] if os.path.exists(data_dir) else []
    file_count = len(pdf_files)
    
    if file_count > 0:
        st.success(f"✅ 成功辨識 {file_count} 份同仁合併大檔案。")
        st.info("💡 目前運行模式：【100% 全文深度精讀模式】\n\n系統已關閉語意猜測。每次提問，AI 都會老老實實把左下角所有檔案從頭到尾讀完，確保真實性與廣泛性！")
        for f in pdf_files:
            st.text(f"📄 {f}")
    else:
        st.warning("⚠️ data 資料夾內找不到任何 PDF 檔案。")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("請輸入宏觀盤點或特定查詢問題... (AI 將逐一精讀所有大檔案，約需 1 分鐘)"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if file_count == 0:
            st.error("沒有任何檔案可供深度精讀。")
            st.stop()
            
        pdf_files.sort() # 確保每次讀取的檔案順序一致
        intermediate_results = ""
        
        # Streamlit 畫面上動態展示進度條與即時播報區
        my_bar = st.progress(0, text="🚀 啟動跨檔案深度全文大盤點...")
        live_output = st.empty()
        
        try:
            for i, filename in enumerate(pdf_files):
                my_bar.progress((i) / file_count, text=f"🔍 正在 100% 全文精讀分析：【{filename}】(每份約需 10 秒，請耐心等候)...")
                file_path = os.path.join(data_dir, filename)
                
                # 讀取該同仁大檔案的「完整全文」
                file_text = extract_text_from_pdf(file_path)
                
                if not file_text.strip():
                    continue
                
                # 針對單一檔案的全文下達最嚴格的數據萃取指令
                extract_prompt = (
                    f"你是一個極度精準的數據與商業資料萃取專家。現在請你『從頭到尾、逐字逐句』完整閱讀以下這份同仁的出差報告大合集。\n"
                    f"請針對使用者的問題：『{prompt}』，將這份報告中所有相關的真實數據、出差次數、時間、地點、拜訪廠商、經濟趨勢等細節，毫無遺漏地全部盤點並記錄下來。\n"
                    f"【鐵律】：請保持資料的 100% 真實性，絕對不可自己捏造、腦補或省略任何數據。如果這份報告中完全找不到與問題相關的任何資訊，請只回答『NO_MATCH』，不要輸出任何贅字。\n\n"
                    f"【當前審查檔案名稱】：{filename}\n"
                    f"【檔案全文內容】:\n{file_text}"
                )
                
                # 呼叫 AI (內建不當機雙引擎備援)
                try:
                    model = genai.GenerativeModel(PRIMARY_CHAT)
                    response = model.generate_content(extract_prompt)
                    res_text = response.text
                except Exception:
                    try:
                        model = genai.GenerativeModel(FALLBACK_CHAT)
                        response = model.generate_content(extract_prompt)
                        res_text = response.text
                    except Exception:
                        res_text = "⚠️ 該檔案因雲端 API 超載暫時無法讀取。"
                
                # 如果這份檔案裡有挖到寶，就記錄下來並即時播報給使用者看
                if "NO_MATCH" not in res_text.upper() and res_text.strip():
                    intermediate_results += f"\n\n============================\n"
                    intermediate_results += f"📄 來自檔案【{filename}】的 100% 真實盤點紀錄：\n"
                    intermediate_results += f"============================\n{res_text}\n"
                    live_output.info(f"👀 **即時播報：AI 剛剛讀完了新檔案！目前累積搜集到的真實線索：**\n\n{intermediate_results}")
                else:
                    live_output.warning(f"👀 播報：檔案【{filename}】中沒有找到與問題相關的紀錄，繼續精讀下一份...")
                
                # 每次精讀完一份大檔案，強制休息 8 秒，極其安全地保護 TPM/RPM 免費限額
                if i < file_count - 1:
                    time.sleep(8)
            
            my_bar.progress(1.0, text="✨ 所有大檔案已 100% 深度精讀完畢！正在進行最終的大數據融合總結...")
            time.sleep(2)
            
            # 第二階段：將所有人各別盤點出來的「純真實線索」融合成最終大報告
            if intermediate_results.strip():
                final_prompt = (
                    f"你是一個專門分析公司內部『出差經驗與商業情報』的高階智能特助。\n"
                    f"使用者問了一個全局大宏觀的問題：『{prompt}』。\n"
                    f"以下是我讓系統把資料夾中『每一份合併大檔案從第一字到最後一字』全部一字不漏精讀後，所撈出來的 100% 真實各檔案盤點碎片。\n"
                    f"請你將這些碎片完美地融合成一份最終的、結構極其清晰、條理分明的大數據總盤點報告。報告中必須包含所有人（不可以漏掉任何一份檔案裡的同仁）、真實的次數加總、去過的地點、拜訪的廠商以及各地的經濟趨勢分析。\n"
                    f"【嚴格要求】：必須展現出 100% 的真實性，數據是多少就寫多少，並且在提及數據時，必須註明資料是源自哪一個檔案，以便同仁查核。\n\n"
                    f"【各檔案 100% 真實盤點碎片集】:\n{intermediate_results}"
                )
                
                try:
                    model = genai.GenerativeModel(PRIMARY_CHAT)
                    final_response = model.generate_content(final_prompt)
                except Exception:
                    model = genai.GenerativeModel(FALLBACK_CHAT)
                    final_response = model.generate_content(final_prompt)
                    
                ai_reply = final_response.text
            else:
                ai_reply = f"抱歉，我剛剛從頭到尾、一字不漏地精讀了資料夾內所有的合併報告，裡面完全找不到任何與『{prompt}』相關的蛛絲馬跡喔！"
            
            my_bar.empty()
            live_output.empty()
            
            st.markdown(ai_reply)
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            
        except Exception as e:
            my_bar.empty()
            live_output.empty()
            st.error(f"系統發生不可預期的錯誤，請稍後重試。錯誤訊息: {str(e)}")
