import streamlit as st
import os
from openai import OpenAI
from typing import Optional, Dict, Any
from datetime import datetime

# 1. 页面配置
st.set_page_config(
    page_title="六模型 AI 对话助手 (DeepSeek + Qwen + Gemini + Kimi + GPT + Mistral)",
    layout="wide",
    page_icon="🤖"
)
st.title("🧠 六模型聊天对比 (V8.1 统一UI版)")

# 2. 模型配置信息（新增 Mistral）
MODEL_CONFIG = {
    "deepseek": {
        "name": "DeepSeek Reasoning",
        "model": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com/v1",
        "web_url": "https://chat.deepseek.com",
        "emoji": "🚀",
        "env_key": "DEEPSEEK_API_KEY",
        "params": {
            "temperature": 0.7,
            "max_tokens": 8192,
            "stream": False,
            "reasoning_effort": "medium"
        }
    },
    "gemini": {
        "name": "Gemini 2.5 Flash",
        "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "web_url": "https://aistudio.google.com",
        "emoji": "✨",
        "env_key": "GEMINI_API_KEY",
        "params": {
            "temperature": 0.7,
            "max_tokens": 8192,
            "stream": False
        }
    },
    "kimi": {
        "name": "Kimi Moonshot",
        "model": "moonshot-v1-8k",
        "base_url": "https://api.moonshot.cn/v1",
        "web_url": "https://kimi.moonshot.cn",
        "emoji": "🌙",
        "env_key": "KIMI_API_KEY",
        "params": {
            "temperature": 0.7,
            "max_tokens": 4096,
            "stream": False
        }
    },
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
            "stream": False
        }
    },
    "qwen": {
        "name": "通义千问 Max",
        "model": "qwen-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "web_url": "https://tongyi.aliyun.com",
        "emoji": "🌸",
        "env_key": "QWEN_API_KEY",
        "params": {
            "temperature": 0.7,
            "max_tokens": 4096,
            "stream": False
        }
    },
    "mistral": {
        "name": "Mistral AI (Mixtral-8x22B)",
        "model": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "base_url": "https://api.mistral.ai/v1",
        "web_url": "https://mistral.ai",
        "emoji": "🦉",
        "env_key": "MISTRAL_API_KEY",
        "params": {
            "temperature": 0.7,
            "max_tokens": 8192,
            "stream": False,
            "safe_prompt": False
        }
    }
}

# 初始化增强设置
if "kimi_k25_enabled" not in st.session_state:
    st.session_state.kimi_k25_enabled = True
if "kimi_streaming_enabled" not in st.session_state:
    st.session_state.kimi_streaming_enabled = True
if "kimi_thinking_mode" not in st.session_state:
    st.session_state.kimi_thinking_mode = False

# 3. 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_responses" not in st.session_state:
    st.session_state.model_responses = {}
if "enable_qwen_search" not in st.session_state:
    st.session_state.enable_qwen_search = False

# 4. 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置面板")

    enabled_models = {}
    for model_id, config in MODEL_CONFIG.items():
        enabled_models[model_id] = st.checkbox(
            f"{config['emoji']} {config['name']}",
            value=True,
            key=f"enable_{model_id}"
        )

    st.write("---")

    st.subheader("🔑 API Key 状态")
    for model_id, config in MODEL_CONFIG.items():
        key = os.getenv(config['env_key'])
        status = "✅ 已配置" if key else "❌ 未配置"
        st.write(f"{config['emoji']} {config['name']}: {status}")

    st.write("---")

    with st.expander("高级设置"):
        reasoning_level = st.select_slider(
            "DeepSeek 推理强度",
            options=["low", "medium", "high"],
            value="medium",
            key="reasoning_level"
        )
        MODEL_CONFIG["deepseek"]["params"]["reasoning_effort"] = reasoning_level

        temperature = st.slider(
            "全局温度",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="global_temperature"
        )

        # 全局温度设置（排除 Mistral）
        for model_id in MODEL_CONFIG:
            if model_id != "mistral":
                MODEL_CONFIG[model_id]["params"]["temperature"] = temperature

        st.divider()
        st.caption("🌙 **Kimi K2.5 说明**")
        st.info("""
        Kimi K2.5 模型固定使用温度 = 1（不可调整）
        该模型通过其他方式控制输出风格，不受全局温度影响

        已启用优化：
        • 32k 输出长度
        • 流式传输
        • 256k 上下文
        """)

        # Mistral 特有设置
        st.divider()
        st.caption("🦉 **Mistral 特有设置**")
        mistral_safe_mode = st.toggle(
            "启用内容安全过滤 (safe_prompt)",
            value=False,
            key="mistral_safe_mode"
        )
        MODEL_CONFIG["mistral"]["params"]["safe_prompt"] = mistral_safe_mode

    if enabled_models.get("qwen", False):
        with st.expander("🌸 Qwen 专属增强"):
            st.session_state.enable_qwen_search = st.checkbox(
                "🔍 启用联网搜索",
                value=st.session_state.enable_qwen_search
            )

    if enabled_models.get("kimi", False):
        with st.expander("🌙 Kimi 专属增强", expanded=True):
            st.session_state.kimi_k25_enabled = st.toggle(
                "使用 Kimi K2.5（推荐）",
                value=st.session_state.kimi_k25_enabled,
                help="升级到最新模型，支持更长上下文和流式输出"
            )

            if st.session_state.kimi_k25_enabled:
                st.success("✅ Kimi K2.5 已启用")
                st.caption("温度固定为 1 | 输出长度: 32k | 上下文: 256k")

                st.session_state.kimi_streaming_enabled = st.toggle(
                    "启用流式传输",
                    value=st.session_state.kimi_streaming_enabled,
                    help="实时显示生成内容，避免超时"
                )

                st.session_state.kimi_thinking_mode = st.toggle(
                    "启用 Thinking 模式",
                    value=st.session_state.kimi_thinking_mode,
                    help="展示模型推理过程（如支持）"
                )
            else:
                st.info("使用原版 Kimi，跟随全局温度设置")

    st.write("---")

    col1, col2 = st.columns(2)
