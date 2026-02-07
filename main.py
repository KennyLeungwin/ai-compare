import streamlit as st
import os
from openai import OpenAI
from typing import Optional
from datetime import datetime

# 1. 页面配置
st.set_page_config(
    page_title="五模型 AI 实验室 (Gemini 2.5 增强版)",
    layout="wide",
    page_icon="🧠"
)
st.title("⚡ 五模型聊天对比 (Gemini 2.5 Flash & GPT-3.5)")

# 2. 模型配置信息 (2026 最新版本)
MODEL_CONFIG = {
    # === ✨ Gemini 核心优化：选用 2.5 Flash (免费层性能之王) ===
    "gemini": {
        "name": "Gemini 2.5 Flash",
        "model": "gemini-2.5-flash", 
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "web_url": "https://aistudio.google.com/",
        "emoji": "✨",
        "env_key": "GEMINI_API_KEY",
        "params": {
            "temperature": 0.7,
            "max_tokens": 8192,
            "stream": True 
        }
    },
    # === 💬 GPT 还原：使用 OpenRouter 免费/低价 API ===
    "gpt": {
        "name": "GPT-3.5 Turbo",
        "model": "openai/gpt-3.5-turbo",
        "base_url": "https://openrouter.ai/api/v1",
        "web_url": "https://chat.openai.com",
        "emoji": "💬",
        "env_key": "OPENROUTER_API_KEY",
        "params": {
            "temperature": 0.7,
            "max_tokens": 2048,
            "stream": True
        }
    },
    "deepseek": {
        "name": "DeepSeek R1",
        "model": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com/v1",
        "emoji": "🚀",
        "env_key": "DEEPSEEK_API_KEY",
        "params": {"temperature": 0.6, "max_tokens": 4096, "stream": True}
    },
    "kimi": {
        "name": "Kimi Moonshot",
        "model": "moonshot-v1-8k",
        "base_url": "https://api.moonshot.cn/v1",
        "emoji": "🌙",
        "env_key": "KIMI_API_KEY",
        "params": {"temperature": 0.7, "max_tokens": 4096, "stream": True}
    },
    "qwen": {
        "name": "通义千问 Max",
        "model": "qwen-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "emoji": "🌸",
        "env_key": "QWEN_API_KEY",
        "params": {"temperature": 0.7, "max_tokens": 4096, "stream": True}
    }
}

# 3. 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. 侧边栏配置
with st.sidebar:
    st.header("⚙️ 模型调度中心")
    
    # API 状态展示 (自动检测环境变量)
    st.subheader("🔑 接口状态")
    enabled_models = {}
    for mid, config in MODEL_CONFIG.items():
        has_key = os.getenv(config['env_key'])
        status_color = "green" if has_key else "red"
        st.markdown(f":{status_color}[{config['emoji']} {config['name']} ({'已就绪' if has_key else '缺少 Key'})]")
        enabled_models[mid] = st.checkbox(f"启用 {config['name']}", value=has_key, key=f"check_{mid}")
    
    st.write("---")
    if st.button("🗑️ 清空上下文", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 5. 渲染历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. 核心对话执行引擎
def execute_chat(model_id: str, col, messages: list):
    config = MODEL_CONFIG[model_id]
    with col:
        st.subheader(f"{config['emoji']} {config['name']}")
        try:
            client = OpenAI(api_key=os.getenv(config['env_key']), base_url=config['base_url'])
            
            # 清理非标准 OpenAI 参数
            api_params = config['params'].copy()
            api_params["model"] = config['model']
            api_params["messages"] = messages
            
            # 流式渲染处理
            response_container = st.empty()
            full_content = ""
            
            with st.spinner(f"{config['name']} 正在思考..."):
                stream = client.chat.completions.create(**api_params)
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_content += chunk.choices[0].delta.content
                        response_container.markdown(full_content + "▌")
            
            response_container.markdown(full_content)
            return full_content
            
        except Exception as e:
            st.error(f"调用失败: {str(e)}")
            return None

# 7. 用户交互入口
if prompt := st.chat_input("输入你的问题..."):
    # 记录用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 筛选已启用的模型
    active_mids = [m for m, v in enabled_models.items() if v]
    if not active_mids:
        st.warning("⚠️ 请至少配置并启用一个模型。")
    else:
        # 分栏显示回复
        cols = st.columns(len(active_mids))
        all_responses = {}
        
        for idx, mid in enumerate(active_mids):
            res = execute_chat(mid, cols[idx], st.session_state.messages)
            if res:
                all_responses[mid] = res

        # 将 Gemini 2.5 的回答作为主回复存入历史（因为它最强）
        if "gemini" in all_responses:
            st.session_state.messages.append({"role": "assistant", "content": f"**[Gemini 2.5]**: {all_responses['gemini']}"})
