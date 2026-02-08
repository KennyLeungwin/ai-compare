import streamlit as st
import os
from openai import OpenAI

# 1. 页面配置
st.set_page_config(page_title="AI Arena v11.0", layout="wide", page_icon="🧬")
st.title("🧬 AI Arena v11.0 (局部渲染增强版)")

# 2. 模型配置
MODEL_CONFIG = {
    "deepseek": {
        "name": "DeepSeek R1", "model": "deepseek-reasoner", "emoji": "🚀",
        "base_url": "https://api.deepseek.com/v1", "env_key": "DEEPSEEK_API_KEY",
        "system": "You are DeepSeek-R1.", "params": {"max_tokens": 8192}
    },
    "gemini": {
        "name": "Gemini 2.5", "model": "gemini-2.5-flash", "emoji": "✨",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "env_key": "GEMINI_API_KEY",
        "system": "You are Gemini by Google.", "params": {"max_tokens": 8192}
    },
    "kimi": {
        "name": "Kimi K2.5", "model": "kimi-k2.5", "emoji": "🌙",
        "base_url": "https://api.moonshot.cn/v1", "env_key": "KIMI_API_KEY",
        "system": "你是 Kimi。", "params": {"max_tokens": 4096}
    },
    "qwen": {
        "name": "通义千问", "model": "qwen-max", "emoji": "🌸",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "env_key": "QWEN_API_KEY",
        "system": "你是通义千问。", "params": {"max_tokens": 4096}
    },
    "mistral": {
        "name": "Mistral Large", "model": "mistral-large-latest", "emoji": "Owl",
        "base_url": "https://api.mistral.ai/v1", "env_key": "MISTRAL_API_KEY",
        "system": "You are Mistral Large.", "params": {"max_tokens": 4096}
    }
}

# 3. 状态初始化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "temp_dict" not in st.session_state:
    st.session_state.temp_dict = {}

# 4. 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    active_ids = [mid for mid in MODEL_CONFIG if st.checkbox(f"{MODEL_CONFIG[mid]['emoji']} {MODEL_CONFIG[mid]['name']}", value=True, key=f"sel_{mid}")]
    global_temp = st.slider("温度控制", 0.0, 1.0, 0.7)
    if st.button("🗑️ 清空记录", use_container_width=True):
        st.session_state.messages = []
        st.session_state.temp_dict = {}
        st.rerun()

# 5. 核心推理组件 (使用 fragment 隔离渲染)
@st.fragment
def model_container(mid, col_obj, user_input):
    cfg = MODEL_CONFIG[mid]
    api_key = os.getenv(cfg['env_key'])
    
    with col_obj:
        st.subheader(f"{cfg['emoji']} {cfg['name']}")
        if not api_key:
            st.error("API Key Missing")
            return

        try:
            client = OpenAI(api_key=api_key, base_url=cfg['base_url'])
            
            # 记忆隔离过滤
            msgs = [{"role": "system", "content": cfg['system']}]
            for m in st.session_state.messages:
                if m["role"] == "user":
                    msgs.append({"role": "user", "content": m["content"]})
                elif m["role"] == "assistant" and m.get("model_id") == mid:
                    msgs.append({"role": "assistant", "content": m["content"]})
            
            # 加上当前这一条
            msgs.append({"role": "user", "content": user_input})

            # 参数适配
            p = cfg['params'].copy()
            p["temperature"] = 1.0 if mid == "kimi" else global_temp

            with st.spinner(f"{cfg['name']} 正在思考..."):
                resp = client.chat.completions.create(model=cfg['model'], messages=msgs, **p)
                ans = resp.choices[0].message.content
                st.markdown(ans)
                
                # 存入临时缓存，避免立即刷新导致冲突
                st.session_state.temp_dict[mid] = ans
        except Exception as e:
            st.error(f"Error: {str(e)}")

# 6. 界面渲染
# 渲染历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入处理
if prompt := st.chat_input("输入问题..."):
    # 立即渲染用户提问
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 启动多模型容器
    if active_ids:
        cols = st.columns(len(active_ids))
        for i, mid in enumerate(active_ids):
            model_container(mid, cols[i], prompt)
        
        # 此时所有模型已在各自的 fragment 中渲染完成
        # 如果需要持久化记录，可以在这里手动同步
        if st.button("💾 保存本次对话到历史"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            for mid, ans in st.session_state.temp_dict.items():
                st.session_state.messages.append({"role": "assistant", "content": ans, "model_id": mid})
            st.session_state.temp_dict = {}
            st.rerun()
