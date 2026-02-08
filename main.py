import streamlit as st
import re

# --- 1. 基础配置 (保持你原有的配置即可) ---
MODEL_CONFIG = {
    "deepseek": {"name": "DeepSeek-V3", "model": "deepseek-chat", "color": "#00E5FF"},
    "gemini": {"name": "Gemini 2.0", "model": "gemini-2.0-flash", "color": "#4285F4"},
    "mistral": {"name": "Mistral Large", "model": "mistral-large-latest", "color": "#FF7000"},
    # ... 其他模型配置
}

# --- 2. 核心函数：带隔离记忆的 AI 请求 ---
def ask_ai(model_id, col_obj):
    config = MODEL_CONFIG[model_id]
    client = config["client"] # 假设你之前已经初始化了各家 client
    
    # 【核心逻辑】构造专属该模型的“纯净记忆”
    # 这样 UI 上大家在一起，但模型心里只有自己和用户
    private_messages = []
    
    # 针对 Mistral 等易受影响模型的系统加固
    if model_id == "mistral":
        private_messages.append({
            "role": "system", 
            "content": "You are a helpful assistant. Your identity is Mistral AI."
        })

    for m in st.session_state.messages:
        # 1. 用户的消息：必须带上
        if m["role"] == "user":
            private_messages.append({"role": "user", "content": m["content"]})
        
        # 2. 系统消息：带上
        elif m["role"] == "system":
            private_messages.append(m)
            
        # 3. 助手消息：【关键】只带上属于该模型自己的历史回复
        # 我们通过 m.get("model_id") 来识别
        elif m["role"] == "assistant" and m.get("model_id") == model_id:
            private_messages.append({"role": "assistant", "content": m["content"]})

    # 处理 Thinking Mode (仅对当前这一轮的用户输入加料)
    if st.session_state.thinking_mode and private_messages:
        last_msg = private_messages[-1]["content"]
        private_messages[-1]["content"] = f"{last_msg}\n(Please think step by step)"

    # 执行请求 (保持你原有的流式输出 UI)
    with col_obj:
        with st.chat_message(model_id):
            placeholder = st.empty()
            full_response = ""
            try:
                stream = client.chat.completions.create(
                    model=config["model"],
                    messages=private_messages,
                    stream=True,
                    temperature=0.7 if model_id != "mistral" else 0.3 # 降低 Mistral 的模仿性
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
                
                # 返回结果，用于后续存入 session_state
                return full_response
            except Exception as e:
                st.error(f"{config['name']} Error: {e}")
                return None

# --- 3. UI 主逻辑 ---
st.title("🤖 Multi-LLM Arena v9.0")

# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thinking_mode" not in st.session_state:
    st.session_state.thinking_mode = False

# 侧边栏及渲染逻辑 (保持原样)
# ... 

# 渲染历史记录 (UI 依然显示所有人的对话)
for m in st.session_state.messages:
    role_icon = "👤" if m["role"] == "user" else "🤖"
    with st.chat_message(m["role"]):
        # 如果是助手，显示模型标签
        prefix = f"**[{m.get('model_id', 'AI')}]** " if m["role"] == "assistant" else ""
        st.markdown(prefix + m["content"])

# --- 4. 输入处理逻辑 ---
if prompt := st.chat_input("输入你的问题..."):
    # 1. 将用户输入存入全局消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun() # 刷新 UI 显示用户问题

# 检查是否有新消息需要 AI 回复
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    enabled_models = ["deepseek", "gemini", "mistral"] # 这里是你勾选的模型
    cols = st.columns(len(enabled_models))
    
    # 2. 并行/依次调用各模型
    for idx, m_id in enumerate(enabled_models):
        response = ask_ai(m_id, cols[idx])
        
        if response:
            # 【重写关键】存入全局消息时，必须带上 model_id 标签
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "model_id": m_id # 这个标签是实现隔离的“手术刀”
            })
