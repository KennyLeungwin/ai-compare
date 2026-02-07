import streamlit as st
import os
from openai import OpenAI

# 1. 页面配置
st.set_page_config(page_title="多模型对比导航版", layout="wide")
st.title("🧠 五模型对比聊天 (官方对话框直达版)")

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
    st.info("💡 提示：点击模型名称可直接跳转到官方网页版对话框进行原生对话。")

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
    
    # 核心调用函数：带官方对话框链接
    def ask_ai(api_key, base_url, model_name, col_obj, display_name, web_url):
        with col_obj:
            # 渲染带链接的标题，点击即跳转
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
        # --- 1. DeepSeek 官方对话链接 ---
        ans_ds = ask_ai(
            os.getenv("DEEPSEEK_API_KEY"), 
            "https://api.deepseek.com/v1", 
            "deepseek-chat", 
            cols[0], 
            "🤖 DeepSeek", 
            "https://chat.deepseek.com/"
        )
        
# --- 2. Gemini (权限探测版) ---
        with cols[1]:
            st.subheader("✨ Gemini 权限探测")
            gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
            
            try:
                import requests
                # 这一步是直接向 Google 询问：我这个 Key 能用哪些模型？
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}"
                response = requests.get(url)
                models_data = response.json()
                
                if "models" in models_data:
                    # 获取前 3 个可用模型的名字
                    available_models = [m["name"] for m in models_data["models"]]
                    st.write("✅ 你的 Key 可用模型：")
                    for m in available_models[:5]:
                        st.code(m)
                    
                    # 尝试用列表里的第一个模型跑一下
                    test_model = available_models[0]
                    client = OpenAI(
                        api_key=gemini_key,
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                    )
                    r = client.chat.completions.create(
                        model=test_model.replace("models/", ""), # 去掉 models/ 前缀
                        messages=st.session_state.messages,
                        timeout=15
                    )
                    st.write(f"🚀 使用 {test_model} 成功回答：")
                    st.write(r.choices[0].message.content)
                    ans_gemini = r.choices[0].message.content
                else:
                    st.error("❌ 该 Key 没关联任何模型。请去 AI Studio 重新创建一个 Key，并选择 'Create API key in a new project'。")
                    ans_gemini = None
            except Exception as e:
                st.error(f"❌ 探测失败: {str(e)}")
                ans_gemini = None
        
        # --- 3. Kimi 官方对话链接 ---
        ans_kimi = ask_ai(
            os.getenv("KIMI_API_KEY"), 
            "https://api.moonshot.cn/v1", 
            "moonshot-v1-8k", 
            cols[2], 
            "🌙 Kimi", 
            "https://kimi.moonshot.cn/"
        )
        
        # --- 4. GPT-3.5 (ChatGPT) 官方对话链接 ---
        ans_gpt = ask_ai(
            os.getenv("OPENROUTER_API_KEY"), 
            "https://openrouter.ai/api/v1", 
            "openai/gpt-3.5-turbo", 
            cols[3], 
            "💬 GPT-3.5", 
            "https://chatgpt.com/"
        )
        
        # --- 5. 通义千问 官方对话链接 ---
        ans_qwen = ask_ai(
            os.getenv("QWEN_API_KEY"), 
            "https://dashscope.aliyuncs.com/compatible-mode/v1", 
            "qwen-max", 
            cols[4], 
            "🌸 通义千问", 
            "https://www.qianwen.com/"
        )

    # 7. 记忆保存
    ref_ans = ans_kimi if ans_kimi else ans_qwen
    if ref_ans:
        st.session_state.messages.append({"role": "assistant", "content": f"[参考回答]: {ref_ans}"})
