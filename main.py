import streamlit as st
import os
from openai import OpenAI

# 1. 页面配置
st.set_page_config(page_title="五模型 AI 对话助手", layout="wide")
st.title("🧠 五模型聊天对比 (V6.0 终极全通版)")

# 2. 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. 侧边栏
with st.sidebar:
    st.header("功能区")
    if st.button("🧹 清空所有对话"):
        st.session_state.messages = []
        st.rerun()
    st.write("---")
    st.success("✅ Gemini 路径已破解：当前使用 2.0 Flash 模型")

# 4. 渲染聊天记录
for message in st.session_state.messages:
    display_content = message["content"]
    if "[参考回答]" in display_content:
        display_content = display_content.split("[参考回答]: ")[-1]
    with st.chat_message(message["role"]):
        st.markdown(display_content)

# 5. 聊天输入框
if prompt := st.chat_input("向 5 个 AI 同时发起提问..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 6. 创建 5 列布局
    cols = st.columns(5)
    
    # 通用调用函数
    def ask_ai(api_key, base_url, model_name, col_obj, display_name, web_url):
        with col_obj:
            st.markdown(f"### [{display_name}]({web_url})")
            if not api_key:
                st.warning("未配置 Key")
                return None
            try:
                client = OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=st.session_state.messages,
                    timeout=60
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                return answer
            except Exception as e:
                st.error(f"❌ 失败: {str(e)}")
                return None

    with st.spinner("5 大 AI 正在同步思考并调取记忆..."):
        # --- 1. DeepSeek ---
        ans_ds = ask_ai(os.getenv("DEEPSEEK_API_KEY"), "https://api.deepseek.com/v1", "deepseek-chat", cols[0], "🤖 DeepSeek", "https://chat.deepseek.com/")
        
# --- 2. Gemini (换成配额更足的 1.5 版本) ---
        ans_gemini = ask_ai(
            os.getenv("GEMINI_API_KEY"), 
            "https://generativelanguage.googleapis.com/v1beta/openai/", 
            "gemini-1.5-flash", # 这里从 2.0 改回 1.5
            cols[1], 
            "✨ Gemini", 
            "https://gemini.google.com/"
        )
        
        # --- 3. Kimi ---
        ans_kimi = ask_ai(os.getenv("KIMI_API_KEY"), "https://api.moonshot.cn/v1", "moonshot-v1-8k", cols[2], "🌙 Kimi", "https://kimi.moonshot.cn/")
        
        # --- 4. GPT-3.5 ---
        ans_gpt = ask_ai(os.getenv("OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1", "openai/gpt-3.5-turbo", cols[3], "💬 GPT-3.5", "https://chatgpt.com/")
        
        # --- 5. 通义千问 ---
        ans_qwen = ask_ai(os.getenv("QWEN_API_KEY"), "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-max", cols[4], "🌸 通义千问", "https://tongyi.aliyun.com/")

    # 7. 记忆同步
    ref_ans = ans_gemini if ans_gemini else ans_qwen
    if ref_ans:
        st.session_state.messages.append({"role": "assistant", "content": f"[参考回答]: {ref_ans}"})
