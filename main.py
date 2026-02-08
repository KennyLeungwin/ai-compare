import streamlit as st
import os
from openai import OpenAI

# 1. 页面配置
st.set_page_config(page_title="AI Arena v12.0", layout="wide", page_icon="🧬")
st.title("🧬 AI Arena v12.0 (自动持久化版)")

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
        "name": "Mistral Large", "model": "mistral-large-latest", "emoji": "🦉",
        "base_url": "https://api.mistral.ai/v1", "env_key": "MISTRAL_API_KEY",
        "system": "You are Mistral Large.", "params": {"max_tokens": 4096}
    }
}

# 3. 状态初始化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    # 使用 set 记录选中的模型
    active_ids = [mid for mid in MODEL_CONFIG if st.checkbox(f"{MODEL_CONFIG[mid]['emoji']} {MODEL_CONFIG[mid]['name']}", value=True, key=f"sel_{mid}")]
    global_temp = st.slider("温度控制 (Kimi除外)", 0.0, 1.0, 0.7)
    if st.button("🗑️ 清空所有对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 5. 辅助函数：消息隔离过滤
def get_isolated_messages(model_id, current_prompt):
    cfg = MODEL_CONFIG[model_id]
    msgs = [{"role": "system", "content": cfg['system']}]
    for m in st.session_state.messages:
        if m["role"] == "user":
            msgs.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant" and m.get("model_id") == model_id:
            # 去除显示标签
            content = m["content"].split("]: ", 1)[-1] if "]: " in m["content"] else m["content"]
            msgs.append({"role": "assistant", "content": content})
    msgs.append({"role": "user", "content": current_prompt})
    return msgs

# 6. 界面渲染逻辑
# 首先渲染历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # 仅显示纯文本给用户看
        display_content = msg["content"].split("]: ", 1)[-1] if "]: " in msg["content"] else msg["content"]
        st.markdown(display_content)

# 处理用户输入
if prompt := st.chat_input("向所有模型提问..."):
    # 1. 立即显示用户的问题
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. 准备并行显示的列
    if active_ids:
        cols = st.columns(len(active_ids))
        new_responses = [] # 用于临时存放结果
        
        for i, mid in enumerate(active_ids):
            cfg = MODEL_CONFIG[mid]
            with cols[i]:
                st.subheader(f"{cfg['emoji']} {cfg['name']}")
                api_key = os.getenv(cfg['env_key'])
                if not api_key:
                    st.error("Missing Key")
                    continue
                
                try:
                    client = OpenAI(api_key=api_key, base_url=cfg['base_url'])
                    p = cfg['params'].copy()
                    p["temperature"] = 1.0 if mid == "kimi" else global_temp
                    
                    with st.spinner("思考中..."):
                        resp = client.chat.completions.create(
                            model=cfg['model'],
                            messages=get_isolated_messages(mid, prompt),
                            **p
                        )
                        ans = resp.choices[0].message.content
                        st.markdown(ans)
                        # 记录到临时列表
                        new_responses.append({"mid": mid, "ans": ans})
                except Exception as e:
                    st.error(f"失败: {str(e)}")

        # 3. 核心改进：所有模型跑完后，一次性存入 session_state 并静默处理
        if new_responses:
            # 先存用户问题
            st.session_state.messages.append({"role": "user", "content": prompt})
            # 再存每个模型的回答
            for item in new_responses:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"[{MODEL_CONFIG[item['mid']]['name']}]: {item['ans']}",
                    "model_id": item['mid']
                })
            # 这里不使用 st.rerun()，以保持 Mistral 的渲染状态不被强行切断
            # 下一次输入时，历史记录会自动渲染出来的
