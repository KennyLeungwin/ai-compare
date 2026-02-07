import streamlit as st
import os
from openai import OpenAI

# 1. 页面配置
st.set_page_config(page_title="四模型 AI 对话助手", layout="wide")
st.title("🧠 四模型 AI 聊天对比 (含 Gemini 免费版)")

# 2. 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. 侧边栏
with st.sidebar:
    if st.button("🧹 清空所有对话"):
        st.session_state.messages = []
        st.rerun()
    st.write("---")
    st.info("提示：Gemini 是目前最稳定的免费国际模型。")

# 4. 在界面上渲染历史消息
for message in st.session_state.messages:
    display_content = message["content"]
    if "[参考回答]" in display_content:
        display_content = display_content.split("[参考回答]: ")[-1]
    
    with st.chat_message(message["role"]):
        st.markdown(display_content)

# 5. 聊天输入框
if prompt := st.chat_input("向 AI 发起提问..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 6. 创建 4 列展示
    cols = st.columns(4)
    
    def ask_ai(api_key, base_url, model_name, col_obj, name):
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

    # 开始获取 4 个回答
    with st.spinner("AI 正在思考并调取记忆..."):
        # 1. DeepSeek
        ans_ds = ask_ai(os.getenv("DEEPSEEK_API_KEY"), "https://api.deepseek.com/v1", "deepseek-chat", cols[0], "🤖 DeepSeek")
        
        # 2. Gemini (新加入的 Google 免费接口)
        ans_gemini = ask_ai(os.getenv("GEMINI_API_KEY"), "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-1.5-flash", cols[1], "✨ Gemini")
        
        # 3. GPT-3.5 (OpenRouter - 没钱会报错)
        ans_gpt = ask_ai(os.getenv("OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1", "openai/gpt-3.5-turbo", cols[2], "💬 GPT-3.5")
        
        # 4. 通义千问 (阿里云)
        ans_qwen = ask_ai(os.getenv("QWEN_API_KEY"), "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-max", cols[3], "🌸 通义千问")

    # 7. 记忆同步：优先存入 Gemini 的回答作为后续参考
    ref_ans = ans_gemini if ans_gemini else ans_qwen
    if ref_ans:
        st.session_state.messages.append({"role": "assistant", "content": f"[参考回答]: {ref_ans}"})
