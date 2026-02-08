import streamlit as st
import os
from openai import OpenAI
from typing import Optional

# 1. 页面配置
st.set_page_config(
    page_title="Multi-LLM Arena v9.1",
    layout="wide",
    page_icon="🤖"
)
st.title("🤖 Multi-LLM Arena v9.1 (Stability Patch)")

# 2. 模型配置
MODEL_CONFIG = {
    "deepseek": {
        "name": "DeepSeek Reasoning",
        "model": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com/v1",
        "emoji": "🚀",
        "env_key": "DEEPSEEK_API_KEY",
        "params": {"temperature": 0.7, "max_tokens": 8192}
    },
    "gemini": {
        "name": "Gemini 2.5 Flash",
        "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "emoji": "✨",
        "env_key": "GEMINI_API_KEY",
        "params": {"temperature": 0.7, "max_tokens": 8192}
    },
    "kimi": {
        "name": "Kimi Moonshot",
        "model": "moonshot-v1-8k",
        "base_url": "https://api.moonshot.cn/v1",
        "emoji": "🌙",
        "env_key": "KIMI_API_KEY",
        "params": {"temperature": 1.0, "max_tokens": 4096}
    },
    "qwen": {
        "name": "通义千问 Max",
        "model": "qwen-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "emoji": "🌸",
        "env_key": "QWEN_API_KEY",
        "params": {"temperature": 0.7, "max_tokens": 4096}
    },
    "mistral": {
        "name": "Mistral AI (Mixtral-8x22B)",
        "model": "mistral-large-latest",
        "base_url": "https://api.mistral.ai/v1",
        "emoji": "🦉",
        "env_key": "MISTRAL_API_KEY",
        "params": {"temperature": 0.7, "max_tokens": 4096}
    }
}

# 3. 初始化状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thinking_mode" not in st.session_state:
    st.session_state.thinking_mode = True

# 4. 侧边栏
with st.sidebar:
    st.header("⚙️ 模型选择")
    enabled_models = {}
    for mid, cfg in MODEL_CONFIG.items():
        enabled_models[mid] = st.checkbox(f"{cfg['emoji']} {cfg['name']}", value=True)
    
    st.write("---")
    st.session_state.thinking_mode = st.toggle("启用思考模式引导", value=st.session_state.thinking_mode)
    global_temp = st.slider("全局温度 (Kimi除外)", 0.0, 1.0, 0.7, 0.1)
    
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

# 5. 核心请求函数 (彻底移除 config["client"] 依赖)
def ask_ai(model_id, col_obj, messages):
    config = MODEL_CONFIG[model_id]
    api_key = os.getenv(config['env_key'])
    
    with col_obj:
        st.subheader(f"{config['emoji']} {config['name']}")
        if not api_key:
            st.error("Missing API Key")
            return None
        
        try:
            # 关键修复：动态初始化客户端，不访问 config 字典中的 client 键
            client = OpenAI(api_key=api_key, base_url=config['base_url'])
            
            # 准备参数
            params = config['params'].copy()
            if model_id == "kimi":
                params["temperature"] = 1.0 # 强制 Kimi 温度为 1.0
            else:
                params["temperature"] = global_temp

            # 消息预处理
            prompt_msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
            if st.session_state.thinking_mode:
                prompt_msgs[-1]["content"] += "\n请详细展示你的思考过程。"

            with st.spinner("思考中..."):
                response = client.chat.completions.create(
                    model=config['model'],
                    messages=prompt_msgs,
                    **params
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                return answer
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None

# 6. 主逻辑渲染
# 先渲染历史记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 处理新输入
if prompt := st.chat_input("问问 AI 们..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# 如果最后一条是用户消息，触发 AI 回复
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    active_ids = [mid for mid, ok in enabled_models.items() if ok]
    if active_ids:
        cols = st.columns(len(active_ids))
        current_responses = {}
        
        for idx, mid in enumerate(active_ids):
            res = ask_ai(mid, cols[idx], st.session_state.messages)
            if res:
                current_responses[mid] = res
        
        # 保存第一个成功的回复到历史记录中（用于维持上下文）
        if current_responses:
            first_mid = list(current_responses.keys())[0]
            st.session_state.messages.append({
                "role": "assistant", 
                "content": current_responses[first_mid]
            })
            # 注意：此处不再强制 st.rerun()，以防在渲染时产生冲突
            # 如果需要立即同步，可以使用 st.fragment 局部刷新
