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
st.title("🧠 六模型聊天对比 (V8.2 - 思考模式增强)")

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
        "web_url": "https://gemini.google.com/",
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
        "web_url": "https://www.qianwen.com/",
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
        "model": "mistral-large-latest",
        "base_url": "https://api.mistral.ai/v1",
        "web_url": "https://chat.mistral.ai/",
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
if "thinking_mode" not in st.session_state:
    st.session_state.thinking_mode = True  # 默认启用思考模式

# 4. 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置面板")

    enabled_models = {}
    for model_id, config in MODEL_CONFIG.items():
        default_enabled = False if model_id == "gpt" else True
        enabled_models[model_id] = st.checkbox(
            f"{config['emoji']} {config['name']}",
            value=default_enabled,
            key=f"enable_{model_id}"
        )

    st.write("---")
    st.subheader("🔑 API Key 状态")
    for model_id, config in MODEL_CONFIG.items():
        key = os.getenv(config['env_key'])
        status = "✅ 已配置" if key else "❌ 未配置"
        st.write(f"{config['emoji']} [{config['name']}]({config['web_url']}): {status}")

    st.write("---")

    with st.expander("高级设置"):
        # 思考模式开关 - 在高级设置里面
        st.subheader("🧠 思考模式")
        thinking_mode = st.toggle(
            "启用思考模式",
            value=st.session_state.thinking_mode,
            help="思考模式会显示推理过程，非思考模式直接给出答案",
            key="thinking_mode_toggle"
        )
        st.session_state.thinking_mode = thinking_mode
        
        if thinking_mode:
            st.success("✅ 思考模式已启用 - 显示详细推理过程")
            st.caption("适合复杂问题、数学计算、逻辑分析等需要推理的场景")
        else:
            st.info("⚡ 非思考模式 - 直接输出答案")
            st.caption("适合简单问答、快速查询等场景，响应更快")
        
        st.divider()
        
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
            
            # 根据思考模式调整消息
            current_messages = messages.copy()
            if st.session_state.thinking_mode:
                # 在思考模式下，添加提示词要求展示推理过程
                thinking_prompts = {
                    "deepseek": "\n\n请一步步展示你的思考过程，最后给出最终答案。",
                    "gemini": "\n\n请展示你的推理步骤，然后给出最终结论。",
                    "kimi": "\n\n请详细展示你的思考过程，然后给出答案。",
                    "gpt": "\n\n请逐步展示你的推理，最后给出结论。",
                    "qwen": "\n\n请展示你的思考步骤，然后给出最终回答。",
                    "mistral": "\n\n请一步步推理，展示思考过程，最后给出答案。"
                }
                
                last_user_msg_index = next(
                    (i for i in range(len(current_messages)-1, -1, -1) 
                     if current_messages[i]["role"] == "user"),
                    -1
                )
                
                if last_user_msg_index >= 0:
                    prompt = current_messages[last_user_msg_index]["content"]
                    thinking_prompt = thinking_prompts.get(model_id, "\n\n请展示你的思考过程。")
                    
                    # 检查是否已经是思考类问题
                    is_thinking_request = any(keyword in prompt.lower() for keyword in [
                        "思考", "推理", "分析", "解释", "为什么", "如何", "步骤",
                        "think", "reason", "analyze", "explain", "why", "how", "step"
                    ])
                    
                    if not is_thinking_request:
                        current_messages[last_user_msg_index]["content"] = prompt + thinking_prompt
            
            # ========== Kimi K2.5 ==========
            if model_id == "kimi" and st.session_state.get("kimi_k25_enabled", True):
                params = {
                    "model": "kimi-k2.5",
                    "messages": current_messages if st.session_state.thinking_mode else messages,
                    "max_tokens": 32768,
                    "stream": False,
                    "temperature": 1.0
                }
                
                # 思考模式的状态显示
                status_msg = "🌙 Kimi 思考中..." if st.session_state.thinking_mode else "🌙 Kimi 回答中..."
                with st.status(status_msg, expanded=st.session_state.thinking_mode) as status:
                    st.write(f"使用模型: kimi-k2.5")
                    st.write(f"模式: {'🧠 思考模式' if st.session_state.thinking_mode else '⚡ 非思考模式'}")
                    
                    response = client.chat.completions.create(**params)
                    answer = response.choices[0].message.content
                    
                    if hasattr(response, 'usage') and response.usage is not None:
                        st.write(f"Tokens: {response.usage.total_tokens}")
                    
                    status_label = "✅ 思考完成！" if st.session_state.thinking_mode else "✅ 回答完成！"
                    status.update(label=status_label, state="complete")
                
                # 显示回答
                st.markdown(answer)
                st.session_state.model_responses[model_id] = {
                    "answer": answer, 
                    "model": "kimi-k2.5",
                    "mode": "thinking" if st.session_state.thinking_mode else "direct"
                }
                return answer
            
            # ========== Mistral AI ==========
            elif model_id == "mistral":
            
                # ===== 精准身份锚定 =====
                mistral_system = {
                    "role": "system",
                    "content": """
            You are Mistral AI's assistant.
            You must NOT claim to be DeepSeek, OpenAI, Qwen, Gemini, or Kimi.
            You MAY say you are from Mistral AI.
            Do not copy other models' responses.
            """
                }
            
                # Mistral 不读取参考回答
                mistral_clean_messages = [m for m in messages if "[参考回答" not in m["content"]]
                mistral_messages = [mistral_system] + (current_messages if st.session_state.thinking_mode else mistral_clean_messages)
            
                params = {
                    "model": config["model"],
                    "messages": mistral_messages,
                    "temperature": config["params"]["temperature"],
                    "max_tokens": config["params"]["max_tokens"],
                    "stream": False
                }
            
                response = client.chat.completions.create(**params)
                answer = response.choices[0].message.content
            
                st.markdown(answer)
                return answer
            


            # ========== 其他模型 ==========
            else:
                params = {
                    "model": config['model'],
                    "messages": current_messages if st.session_state.thinking_mode else messages,
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

                status_msg = f"{config['emoji']} 思考中..." if st.session_state.thinking_mode else f"{config['emoji']} 回答中..."
                with st.status(status_msg, expanded=st.session_state.thinking_mode) as status:
                    st.write(f"使用模型: {config['model']}")
                    st.write(f"模式: {'🧠 思考模式' if st.session_state.thinking_mode else '⚡ 非思考模式'}")
                    
                    response = client.chat.completions.create(**params)
                    answer = response.choices[0].message.content

                    if hasattr(response, 'usage') and response.usage is not None:
                        st.write(f"Tokens: {response.usage.total_tokens}")
                    
                    status_label = f"{config['emoji']} 思考完成！" if st.session_state.thinking_mode else f"{config['emoji']} 回答完成！"
                    status.update(label=status_label, state="complete")

                st.markdown(answer)
                st.session_state.model_responses[model_id] = {
                    "answer": answer, 
                    "model": config['model'],
                    "mode": "thinking" if st.session_state.thinking_mode else "direct"
                }
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
    
    # 显示当前模式状态
    mode_status = "🧠 思考模式" if st.session_state.thinking_mode else "⚡ 非思考模式"
    st.info(f"当前模式: {mode_status} | 正在同步调用 {len(enabled_model_ids)} 个模型...")
    
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
st.sidebar.caption("🔄 版本 V8.2 | 支持思考/非思考模式切换 | 默认思考模式")
st.sidebar.write("### 模式说明")
st.sidebar.info("""
**🧠 思考模式**
- 展示详细推理过程
- 适合复杂问题、数学计算、逻辑分析
- 响应时间稍长

**⚡ 非思考模式**
- 直接给出答案
- 响应速度更快
- 适合简单问答、快速查询
""")
