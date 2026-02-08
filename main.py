import streamlit as st
import os
from openai import OpenAI
from typing import Optional

# 1. 页面基础配置
st.set_page_config(
    page_title="AI Arena Pro v10.0",
    layout="wide",
    page_icon="🧬"
)
st.title("🧬 AI Arena Pro v10.0 (全模型认知修复版)")

# 2. 模型核心配置
MODEL_CONFIG = {
    "deepseek": {
        "name": "DeepSeek R1",
        "model": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com/v1",
        "emoji": "🚀",
        "env_key": "DEEPSEEK_API_KEY",
        "system": "You are DeepSeek, a large language model developed by DeepSeek-AI.",
        "params": {"max_tokens": 8192}
    },
    "gemini": {
        "name": "Gemini 2.5 Flash",
        "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "emoji": "✨",
        "env_key": "GEMINI_API_KEY",
        "system": "You are Gemini, a large language model trained by Google.",
        "params": {"max_tokens": 8192}
    },
    "kimi": {
        "name": "Kimi Moonshot",
        "model": "kimi-k2.5", # 默认使用最新 K2.5
        "base_url": "https://api.moonshot.cn/v1",
        "emoji": "🌙",
        "env_key": "KIMI_API_KEY",
        "system": "你是 Kimi，由 Moonshot AI 提供的对话助手。",
        "params": {"max_tokens": 4096}
    },
    "qwen": {
        "name": "通义千问 Max",
        "model": "qwen-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "emoji": "🌸",
        "env_key": "QWEN_API_KEY",
        "system": "你是通义千问，由阿里巴巴开发的 AI 助手。",
        "params": {"max_tokens": 4096}
    },
    "mistral": {
        "name": "Mistral Large",
        "model": "mistral-large-latest",
        "base_url": "https://api.mistral.ai/v1",
        "emoji": "🦉",
        "env_key": "MISTRAL_API_KEY",
        "system": "You are Mistral AI, a large language model developed by Mistral AI.",
        "params": {"max_tokens": 4096}
    }
}

# 3. 会话状态初始化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thinking_mode" not in st.session_state:
    st.session_state.thinking_mode = True

# 4. 侧边栏交互
with st.sidebar:
    st.header("⚙️ 竞技场设置")
    enabled_models = {}
    for mid, cfg in MODEL_CONFIG.items():
        enabled_models[mid] = st.checkbox(f"{cfg['emoji']} {cfg['name']}", value=True)
    
    st.write("---")
    st.session_state.thinking_mode = st.toggle("强制推理模式引导", value=st.session_state.thinking_mode)
    global_temp = st.slider("全局生成温度 (除Kimi外)", 0.0, 1.0, 0.7, 0.1)
    
    if st.button("🗑️ 清空所有历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 5. 核心推理函数
def ask_ai(model_id, col_obj, history_messages):
    config = MODEL_CONFIG[model_id]
    api_key = os.getenv(config['env_key'])
    
    with col_obj:
        st.subheader(f"{config['emoji']} {config['name']}")
        if not api_key:
            st.warning(f"未检测到 {config['env_key']}")
            return None
        
        try:
            client = OpenAI(api_key=api_key, base_url=config['base_url'])
            
            # --- 记忆隔离逻辑 (解决认知问题) ---
            # 只提取：系统提示 + 用户历史提问 + 该模型自己的历史回答
            filtered_msgs = [{"role": "system", "content": config['system']}]
            for m in history_messages:
                if m["role"] == "user":
                    filtered_msgs.append({"role": "user", "content": m["content"]})
                elif m["role"] == "assistant" and m.get("model_id") == model_id:
                    # 剥离显示用的标签，只保留纯文本内容给 AI 参考
                    pure_content = m["content"].split("]: ", 1)[-1] if "]: " in m["content"] else m["content"]
                    filtered_msgs.append({"role": "assistant", "content": pure_content})
            
            # 推理模式引导
            if st.session_state.thinking_mode:
                filtered_msgs[-1]["content"] += "\n(请详细展示你的思考推导过程)"

            # 参数动态适配
            run_params = config['params'].copy()
            run_params["temperature"] = 1.0 if model_id == "kimi" else global_temp

            with st.status(f"{config['emoji']} 正在响应...", expanded=False) as status:
                response = client.chat.completions.create(
                    model=config['model'],
                    messages=filtered_msgs,
                    **run_params
                )
                answer = response.choices[0].message.content
                status.update(label="响应完成", state="complete")
            
            st.markdown(answer)
            return answer

        except Exception as e:
            st.error(f"调用中断: {str(e)}")
            return None

# 6. 聊天记录渲染 (历史回溯)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # 渲染时隐藏 [参考回答] 标签，保持界面简洁
        display_text = msg["content"]
        if "]: " in display_text:
            display_text = display_text.split("]: ", 1)[-1]
        st.markdown(display_text)

# 7. 用户输入与多模型联动逻辑
if prompt := st.chat_input("输入指令，所有选中的 AI 将同时回答..."):
    # 1. 存入用户消息并即时刷新
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# 触发 AI 响应
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    active_ids = [mid for mid, ok in enabled_models.items() if ok]
    
    if active_ids:
        # 开启多列布局
        cols = st.columns(len(active_ids))
        
        # 依次请求每个选中的模型
        for idx, mid in enumerate(active_ids):
            # 将当前完整的 session_state 传入以提取该模型的私有记忆
            res = ask_ai(mid, cols[idx], st.session_state.messages)
            
            if res:
                # 关键：将每个模型的回答带上 ID 存入总历史，供下一轮隔离过滤
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"[来自 {MODEL_CONFIG[mid]['name']}]: {res}",
                    "model_id": mid
                })
        
        # 所有模型运行结束后，为了保证 session_state 同步，进行一次静默刷新
        st.rerun()
