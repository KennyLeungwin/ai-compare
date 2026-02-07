import streamlit as st
import os
from openai import OpenAI
from typing import Optional
from datetime import datetime

# 1. 页面配置
st.set_page_config(
    page_title="六模型 AI 对话助手 (DeepSeek + Qwen + Gemini + Kimi + GPT + Mistral)",
    layout="wide",
    page_icon="🤖"
)
st.title("🧠 六模型聊天对比 (V8.1 - 回答直接显示)")

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
        "model": "mistral-large-latest",  # 或 "mistralai/Mixtral-8x22B-Instruct-v0.1"
        "base_url": "https://api.mistral.ai/v1",
        "web_url": "https://mistral.ai",
        "emoji": "🦉",
        "env_key": "MISTRAL_API_KEY",
        "params": {
            "temperature": 0.7,
            "max_tokens": 8192,
            "stream": False,
        }
    }
}

# 3. 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_responses" not in st.session_state:
    st.session_state.model_responses = {}
if "enable_qwen_search" not in st.session_state:
    st.session_state.enable_qwen_search = False
if "kimi_k25_enabled" not in st.session_state:
    st.session_state.kimi_k25_enabled = True

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

        for model_id in MODEL_CONFIG:
            if model_id != "mistral" and not (model_id == "kimi" and st.session_state.get("kimi_k25_enabled", False)):
                MODEL_CONFIG[model_id]["params"]["temperature"] = temperature

        st.divider()
        st.caption("🌙 **Kimi K2.5 说明**")
        st.info("""
        Kimi K2.5 模型固定使用温度 = 1（不可调整）
        已启用优化：
        • 32k 输出长度
        • 256k 上下文
        """)

    if enabled_models.get("qwen", False):
        with st.expander("🌸 Qwen 专属增强"):
            st.session_state.enable_qwen_search = st.checkbox("🔍 启用联网搜索", value=st.session_state.enable_qwen_search)

    if enabled_models.get("kimi", False):
        with st.expander("🌙 Kimi 专属增强", expanded=True):
            st.session_state.kimi_k25_enabled = st.toggle("使用 Kimi K2.5（推荐）", value=st.session_state.kimi_k25_enabled)
            if st.session_state.kimi_k25_enabled:
                st.success("✅ Kimi K2.5 已启用")
                st.caption("温度固定为 1 | 输出长度: 32k")

    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.model_responses = {}
            st.rerun()
    with col2:
        if st.button("🔄 刷新页面", use_container_width=True):
            st.rerun()

# 5. 渲染历史消息
for message in st.session_state.messages:
    display_content = message["content"]
    if "[参考回答]" in display_content:
        display_content = display_content.split("[参考回答]: ")[-1]
    with st.chat_message(message["role"]):
        st.markdown(display_content)

# 6. 核心函数：调用 AI 模型
def ask_ai(model_id: str, col_obj, messages: list) -> Optional[str]:
    config = MODEL_CONFIG[model_id]
    
    with col_obj:
        st.markdown(f"### {config['emoji']} [{config['name']}]({config['web_url']})")
        
        api_key = os.getenv(config['env_key'])
        if not api_key:
            st.warning(f"请设置 {config['env_key']}")
            return None
        
        if not enabled_models.get(model_id, True):
            st.info("⏸️ 模型已禁用")
            return None
        
        try:
            client = OpenAI(api_key=api_key, base_url=config['base_url'])
            
            # ========== Kimi K2.5 ==========
            if model_id == "kimi" and st.session_state.get("kimi_k25_enabled", True):
                params = {
                    "model": "kimi-k2.5",
                    "messages": messages,
                    "max_tokens": 32768,
                    "stream": False,
                    "temperature": 1.0
                }
                st.caption("🌙 Kimi K2.5 | 温度: 1 (固定) | 32k 输出")
                
                with st.status("🌙 Kimi 思考中...", expanded=False) as status:
                    st.write("使用模型: kimi-k2.5")
                    response = client.chat.completions.create(**params)
                    answer = response.choices[0].message.content
                    
                    if hasattr(response, 'usage') and response.usage is not None:
                        st.write(f"Tokens: {response.usage.total_tokens}")
                    status.update(label="✅ 完成！", state="complete")
                
                # ✅ 关键：直接显示回答
                st.markdown(answer)
                st.session_state.model_responses[model_id] = {
                    "answer": answer, 
                    "model": "kimi-k2.5"
                }
                return answer
            
            # ========== Mistral AI ==========
            elif model_id == "mistral":
                params = {
                    "model": config["model"],
                    "messages": messages,
                    "temperature": config["params"]["temperature"],
                    "max_tokens": config["params"]["max_tokens"],
                    "stream": False
                }

                # 动态调整温度
                last_user_msg = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
                if "代码" in last_user_msg or "program" in last_user_msg.lower():
                    params["temperature"] = 0.3
                elif "创意" in last_user_msg or "story" in last_user_msg.lower():
                    params["temperature"] = 0.9

                with st.status("🦉 Mistral 正在思考中...", expanded=False) as status:
                    st.write(f"使用模型: {config['model']}")
                    response = client.chat.completions.create(**params)
                    answer = response.choices[0].message.content

                    if hasattr(response, 'usage') and response.usage is not None:
                        st.write(f"Tokens: {response.usage.total_tokens}")
                    status.update(label="✅ 完成！", state="complete")

                # ✅ 关键：直接显示回答（不在 try 内部嵌套太深）
                st.markdown(answer)
                st.session_state.model_responses[model_id] = {"answer": answer, "model": config["model"]}
                return answer

            # ========== 其他模型 ==========
            else:
                params = {
                    "model": config['model'],
                    "messages": messages,
                    **config['params']
                }

                if model_id == "qwen":
                    if st.session_state.get("enable_qwen_search", False):
                        params["enable_search"] = True
                    last_user_msg = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
                    if any(kw in last_user_msg.lower() for kw in ["json", "结构化", "表格"]):
                        params["response_format"] = {"type": "json_object"}

                elif model_id == "gemini":
                    last_user_msg = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
                    if "代码" in last_user_msg or "写个" in last_user_msg:
                        params["temperature"] = max(0.2, params["temperature"] - 0.3)

                elif model_id == "deepseek" and "reasoning_effort" in params:
                    if not params.get("stream"):
                        params.pop("stream", None)

                with st.status(f"{config['emoji']} 正在思考中...", expanded=False) as status:
                    st.write(f"使用模型: {config['model']}")
                    response = client.chat.completions.create(**params)
                    answer = response.choices[0].message.content

                    if hasattr(response, 'usage') and response.usage is not None:
                        st.write(f"Tokens: {response.usage.total_tokens}")
                    status.update(label=f"{config['emoji']} 完成！", state="complete")

                # ✅ 统一在此处显示
                st.markdown(answer)
                st.session_state.model_responses[model_id] = {"answer": answer, "model": config['model']}
                return answer

        except Exception as e:
            st.error(f"❌ {config['name']} 调用失败: {str(e)[:200]}")
            return None

# 7. 用户输入处理
if prompt := st.chat_input("向AI模型提问..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    enabled_model_ids = [mid for mid in MODEL_CONFIG.keys() if enabled_models.get(mid, True)]
    if not enabled_model_ids:
        st.warning("⚠️ 请至少启用一个模型")
        st.stop()
    
    cols = st.columns(len(enabled_model_ids))
    with st.spinner(f"正在同步调用 {len(enabled_model_ids)} 个模型..."):
        responses = {}
        for idx, model_id in enumerate(enabled_model_ids):
            answer = ask_ai(model_id, cols[idx], st.session_state.messages)
            if answer:
                responses[model_id] = answer
        
        # 添加参考回答到聊天记录（用于后续上下文）
        if responses:
            ref_order = ["deepseek", "qwen", "gemini", "kimi", "mistral", "gpt"]
            ref_model_id = next((mid for mid in ref_order if mid in responses), list(responses.keys())[0])
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"[参考回答 - {MODEL_CONFIG[ref_model_id]['name']}]: {responses[ref_model_id]}"
            })

# 8. 底部信息
st.sidebar.write("---")
st.sidebar.caption("🔄 版本 V8.1 | 所有模型回答直接显示 | Mistral & Kimi 已修复")
