import streamlit as st
import os
from openai import OpenAI
from typing import Optional

# 1. 页面配置
st.set_page_config(
    page_title="六模型 AI 对话助手 (DeepSeek + Qwen + Gemini + Kimi + GPT + Mistral)",
    layout="wide",
    page_icon="🤖"
)
st.title("🧠 六模型聊天对比 (V13.0 - UI全回归版)")

# 2. 模型配置信息 (整合了 Web URL 和参数)
MODEL_CONFIG = {
    "deepseek": {
        "name": "DeepSeek Reasoning", "model": "deepseek-reasoner", "emoji": "🚀",
        "base_url": "https://api.deepseek.com/v1", "web_url": "https://chat.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY", "system": "You are DeepSeek-R1.",
        "params": {"temperature": 0.7, "max_tokens": 8192, "reasoning_effort": "medium"}
    },
    "gemini": {
        "name": "Gemini 2.5 Flash", "model": "gemini-2.5-flash", "emoji": "✨",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "web_url": "https://gemini.google.com/",
        "env_key": "GEMINI_API_KEY", "system": "You are Gemini by Google.",
        "params": {"temperature": 0.7, "max_tokens": 8192}
    },
    "kimi": {
        "name": "Kimi Moonshot", "model": "kimi-k2.5", "emoji": "🌙",
        "base_url": "https://api.moonshot.cn/v1", "web_url": "https://kimi.moonshot.cn",
        "env_key": "KIMI_API_KEY", "system": "你是 Kimi。",
        "params": {"temperature": 1.0, "max_tokens": 4096}
    },
    "gpt": {
        "name": "GPT-3.5 Turbo", "model": "openai/gpt-3.5-turbo", "emoji": "💬",
        "base_url": "https://openrouter.ai/api/v1", "web_url": "https://chat.openai.com",
        "env_key": "OPENROUTER_API_KEY", "system": "You are ChatGPT.",
        "params": {"temperature": 0.7, "max_tokens": 2048}
    },
    "qwen": {
        "name": "通义千问 Max", "model": "qwen-max", "emoji": "🌸",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "web_url": "https://www.qianwen.com/",
        "env_key": "QWEN_API_KEY", "system": "你是通义千问。",
        "params": {"temperature": 0.7, "max_tokens": 4096}
    },
    "mistral": {
        "name": "Mistral Large", "model": "mistral-large-latest", "emoji": "🦉",
        "base_url": "https://api.mistral.ai/v1", "web_url": "https://chat.mistral.ai/",
        "env_key": "MISTRAL_API_KEY", "system": "You are Mistral Large.",
        "params": {"temperature": 0.7, "max_tokens": 4096}
    }
}

# 3. 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "enable_qwen_search" not in st.session_state:
    st.session_state.enable_qwen_search = False
if "kimi_k25_enabled" not in st.session_state:
    st.session_state.kimi_k25_enabled = True
if "thinking_mode" not in st.session_state:
    st.session_state.thinking_mode = True

# 4. 侧边栏 UI (完整回归)
with st.sidebar:
    st.header("⚙️ 配置面板")
    
    enabled_models = {}
    for model_id, config in MODEL_CONFIG.items():
        default_val = False if model_id == "gpt" else True
        enabled_models[model_id] = st.checkbox(f"{config['emoji']} {config['name']}", value=default_val, key=f"en_{model_id}")

    st.write("---")
    st.subheader("🔑 API Key 状态")
    for model_id, config in MODEL_CONFIG.items():
        key = os.getenv(config['env_key'])
        status = "✅ 已配置" if key else "❌ 未配置"
        st.write(f"{config['emoji']} [{config['name']}]({config['web_url']}): {status}")

    st.write("---")
    with st.expander("高级设置"):
        st.subheader("🧠 思考模式")
        st.session_state.thinking_mode = st.toggle("启用思考模式", value=st.session_state.thinking_mode, help="思考模式会显式引导模型进行逻辑推理")
        
        reasoning_level = st.select_slider("DeepSeek 推理强度", options=["low", "medium", "high"], value="medium")
        MODEL_CONFIG["deepseek"]["params"]["reasoning_effort"] = reasoning_level

        global_temp = st.slider("全局温度", 0.0, 1.0, 0.7)
        for mid in MODEL_CONFIG:
            if mid != "kimi": # Kimi K2.5 通常固定为 1
                MODEL_CONFIG[mid]["params"]["temperature"] = global_temp

    if enabled_models.get("qwen"):
        with st.expander("🌸 Qwen 专属增强"):
            st.session_state.enable_qwen_search = st.checkbox("🔍 启用联网搜索", value=st.session_state.enable_qwen_search)

    if enabled_models.get("kimi"):
        with st.expander("🌙 Kimi 专属增强", expanded=True):
            st.session_state.kimi_k25_enabled = st.toggle("使用 Kimi K2.5", value=st.session_state.kimi_k25_enabled)

    st.write("---")
    if st.button("🗑️ 清空所有对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 5. 辅助函数：隔离记忆
def get_isolated_messages(model_id, current_prompt):
    cfg = MODEL_CONFIG[model_id]
    msgs = [{"role": "system", "content": cfg['system']}]
    
    # 思考模式注入
    prompt_to_send = current_prompt
    if st.session_state.thinking_mode:
        prompt_to_send += "\n\n请详细展示你的思考步骤，然后再给出最终回答。"

    for m in st.session_state.messages:
        if m["role"] == "user":
            msgs.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant" and m.get("model_id") == model_id:
            # 提取存入时的纯净内容
            content = m["content"].split("]: ", 1)[-1] if "]: " in m["content"] else m["content"]
            msgs.append({"role": "assistant", "content": content})
    
    msgs.append({"role": "user", "content": prompt_to_send})
    return msgs

# 6. 核心对话逻辑
# 渲染历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        display_content = msg["content"].split("]: ", 1)[-1] if "]: " in msg["content"] else msg["content"]
        st.markdown(display_content)

# 输入处理
if prompt := st.chat_input("向选中的 AI 模型提问..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    active_ids = [mid for mid, enabled in enabled_models.items() if enabled]
    if active_ids:
        cols = st.columns(len(active_ids))
        new_responses = []

        for i, mid in enumerate(active_ids):
            cfg = MODEL_CONFIG[mid]
            with cols[i]:
                st.markdown(f"### {cfg['emoji']} [{cfg['name']}]({cfg['web_url']})")
                api_key = os.getenv(cfg['env_key'])
                
                if not api_key:
                    st.warning(f"缺失密钥: {cfg['env_key']}")
                    continue

                try:
                    client = OpenAI(api_key=api_key, base_url=cfg['base_url'])
                    params = cfg['params'].copy()
                    if mid == "qwen" and st.session_state.enable_qwen_search:
                        params["enable_search"] = True

                    status_text = f"{cfg['emoji']} 思考中..." if st.session_state.thinking_mode else f"{cfg['emoji']} 回答中..."
                    with st.status(status_text) as status:
                        resp = client.chat.completions.create(
                            model=cfg['model'],
                            messages=get_isolated_messages(mid, prompt),
                            **params
                        )
                        ans = resp.choices[0].message.content
                        status.update(label=f"✅ {cfg['name']} 完成", state="complete")
                    
                    st.markdown(ans)
                    new_responses.append({"mid": mid, "ans": ans})
                except Exception as e:
                    st.error(f"调用失败: {str(e)[:100]}")

        # 持久化：将本次对话存入 session_state
        if new_responses:
            st.session_state.messages.append({"role": "user", "content": prompt})
            for item in new_responses:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"[{MODEL_CONFIG[item['mid']]['name']}]: {item['ans']}",
                    "model_id": item['mid']
                })
