import streamlit as st
import os
from openai import OpenAI
import google.generativeai as genai

# 1. 页面配置
st.set_page_config(page_title="四模型 AI 对话助手", layout="wide")
st.title("🧠 四模型 AI 聊天对比 (Gemini 官方库版)")

# 2. 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. 侧边栏
with st.sidebar:
    if st.button("🧹 清空所有对话"):
        st.session_state.messages = []
        st.rerun()
    st.write("---")
    st.info("Gemini 现已切换至官方 SDK 调用，彻底解决 404 问题。")

# 4. 在界面上渲染历史消息
for message in st.session_state.messages:
    display_content = message["content"]
    if "[参考回答]" in display_content:
        display_content = display_content.split("[参考回答]: ")[-1]
    with st.chat_message(message["role"]):
        st.markdown(display_content)

# 5. 聊天输入框
if prompt := st.chat_input("向 AI 发起提问..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    cols = st.columns(4)
    
    # --- 通用 OpenAI 调用函数 (用于 DeepSeek, GPT, Qwen) ---
    def ask_openai_style(api_key, base_url, model_name, col_obj, name):
        with col_obj:
            st.subheader(name)
            if not api_key:
                st.error("未在 Secrets 填入 Key")
                return None
            try:
                client = OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=st.session_state.messages,
                    timeout=30
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                return answer
            except Exception as e:
                st.error(f"❌ 失败: {str(e)}")
                return None

    with st.spinner("AI 正在同步思考并调取记忆..."):
        # 1. DeepSeek
        ans_ds = ask_openai_style(os.getenv("DEEPSEEK_API_KEY"), "https://api.deepseek.com/v1", "deepseek-chat", cols[0], "🤖 DeepSeek")
        
        # 2. Gemini (官方库调用方式 - 彻底修复 404)
        with cols[1]:
            st.subheader("✨ Gemini")
            try:
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # 转换记忆格式：将 OpenAI 格式转为 Google 格式
                history = []
                for m in st.session_state.messages[:-1]:
                    # Google 角色要求：user 对应 user, model 对应 assistant
                    g_role = "user" if m["role"] == "user" else "model"
                    history.append({"role": g_role, "parts": [m["content"]]})
                
                chat = model.start_chat(history=history)
                response = chat.send_message(prompt)
                ans_gemini = response.text
                st.markdown(ans_gemini)
            except Exception as e:
                st.error(f"❌ Gemini 官方库报错: {str(e)}")
                ans_gemini = None
        
        # 3. GPT-3.5 (OpenRouter)
        ans_gpt = ask_openai_style(os.getenv("OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1", "openai/gpt-3.5-turbo", cols[2], "💬 GPT-3.5")
        
        # 4. 通义千问 (阿里云)
        ans_qwen = ask_openai_style(os.getenv("QWEN_API_KEY"), "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-max", cols[3], "🌸 通义千问")

    # 7. 记忆同步：优先存入 Gemini 或 Qwen 的回答
    ref_ans = ans_gemini if ans_gemini else ans_qwen
    if ref_ans:
        st.session_state.messages.append({"role": "assistant", "content": f"[参考回答]: {ref_ans}"})
