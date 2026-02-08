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

# 2. 模型配置信息
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
    st.session_state.thinking_mode = True  

# 4. 侧边栏配置 (保留原 UI)
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
        st.subheader("🧠 思考模式")
        thinking_mode = st.toggle("启用思考模式", value=st.session_state.thinking_mode, key="thinking_mode_toggle")
        st.session_state.thinking_mode = thinking_mode
        st.divider()
        reasoning_level = st.select_slider("DeepSeek 推理强度", options=["low", "medium", "high"], value="medium", key="reasoning_level")
        MODEL_CONFIG["deepseek"]["params"]["reasoning_effort"] = reasoning_level
        temperature = st.slider("全局温度", 0.0, 1.0, 0.7, 0.1, key="global_temperature")
        for model_id in MODEL_CONFIG:
            if model_id != "mistral" and not (model_id == "kimi" and st.session_state.get("kimi_k25_enabled", False)):
                MODEL_CONFIG[model_id]["params"]["temperature"] = temperature
    
    if enabled_models.get("qwen", False):
        with st.expander("🌸 Qwen 专属增强"):
            st.session_state.enable_qwen_search = st.checkbox("🔍 启用联网搜索", value=st.session_state.enable_qwen_search)
    if enabled_models.get("kimi", False):
        with st.expander("🌙 Kimi 专属增强", expanded=True):
            st.session_state.kimi_k25_enabled = st.toggle("使用 Kimi K2.5（推荐）", value=st.session_state.kimi_k25_enabled)

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

# 5. 渲染历史消息 (保留原 UI)
for message in st.session_state.messages:
    display_content = message["content"]
    # UI 层面：如果是参考回答，剥离标签显示给用户看
    if "[参考回答" in display_content:
        display_content = display_content.split("]: ", 1)[-1]
    with st.chat_message(message["role"]):
        st.markdown(display_content)

# 6. 核心函数：调用 AI 模型 (逻辑增强版)
def ask_ai(model_id: str, col_obj, messages: list) -> Optional[str]:
    config = MODEL_CONFIG[model_id]
    
    with col_obj:
        st.markdown(f"### {config['emoji']} [{config['name']}]({config['web_url']})")
        api_key = os.getenv(config['env_key'])
        if not api_key or not enabled_models.get(model_id, True):
            return None
        
        try:
            client = OpenAI(api_key=api_key, base_url=config['base_url'])
            
            # 【核心逻辑：记忆隔离】
            # 创建一个只包含“用户提问”和“该模型自己回答过的话”的消息列表
            # 彻底屏蔽掉包含 [参考回答 - DeepSeek] 这种会误导身份的内容
            filtered_messages = []
            for m in messages:
                if m["role"] == "user":
                    filtered_messages.append(m)
                elif m["role"] == "assistant":
                    # 只有这条消息标注了属于当前 model_id，或者它是该模型上一轮的私有回复，才加入历史
                    if m.get("model_id") == model_id:
                        filtered_messages.append({"role": "assistant", "content": m["content"]})
            
            # 针对身份加固的 System Prompt
            system_prompts = {
                "mistral": "You are Mistral AI. Strictly maintain your identity.",
                "deepseek": "You are DeepSeek.",
                "gemini": "You are Gemini by Google.",
                "qwen": "你是通义千问。",
                "kimi": "你是 Kimi。"
            }
            if model_id in system_prompts:
                filtered_messages.insert(0, {"role": "system", "content": system_prompts[model_id]})

            # 思考模式处理
            if st.session_state.thinking_mode:
                thinking_prompts = {
                    "deepseek": "\n\n请一步步展示思考过程，最后给出答案。",
                    "mistral": "\n\n请一步步推理，展示思考过程，最后给出答案。",
                    "gemini": "\n\n请展示你的推理步骤，然后给出最终结论。"
                }
                last_msg = filtered_messages[-1]["content"]
                filtered_messages[-1]["content"] = last_msg + thinking_prompts.get(model_id, "\n\n请展示你的思考过程。")

            # 构建请求参数
            params = {
                "model": "kimi-k2.5" if (model_id == "kimi" and st.session_state.kimi_k25_enabled) else config['model'],
                "messages": filtered_messages,
                **config['params']
            }
            
            # 移除不兼容 stream 字段的特殊处理 (DeepSeek)
            if model_id == "deepseek": params.pop("stream", None)

            # 渲染状态
            status_msg = f"{config['emoji']} 思考中..." if st.session_state.thinking_mode else f"{config['emoji']} 回答中..."
            with st.status(status_msg, expanded=st.session_state.thinking_mode) as status:
                response = client.chat.completions.create(**params)
                answer = response.choices[0].message.content
                status.update(label=f"{config['emoji']} 完成！", state="complete")

            st.markdown(answer)
            return answer

        except Exception as e:
            st.error(f"❌ 调用失败: {str(e)[:100]}")
            return None

# 7. 用户输入处理
if prompt := st.chat_input("向AI模型提问..."):
    # 用户消息正常展示
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# 触发 AI 回复逻辑
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    enabled_model_ids = [mid for mid in MODEL_CONFIG.keys() if enabled_models.get(mid, True)]
    if enabled_model_ids:
        cols = st.columns(len(enabled_model_ids))
        current_responses = {}
        
        for idx, m_id in enumerate(enabled_model_ids):
            # 将完整的 messages 传进去，由 ask_ai 内部进行过滤
            res = ask_ai(m_id, cols[idx], st.session_state.messages)
            if res:
                current_responses[m_id] = res
        
        # 存入 session_state 时，保持你原有的“参考回答”UI 逻辑，但增加 model_id 标签
        if current_responses:
            ref_order = ["deepseek", "qwen", "gemini", "kimi", "mistral", "gpt"]
            ref_id = next((mid for mid in ref_order if mid in current_responses), list(current_responses.keys())[0])
            
            # 这里是关键：存入一条带 [参考回答] 的消息，但同时标注它是属于哪个 model_id 的
            # 这样下一次循环时，其他模型由于 model_id 不符，会自动无视这条消息
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"[参考回答 - {MODEL_CONFIG[ref_id]['name']}]: {current_responses[ref_id]}",
                "model_id": ref_id # 只有 ref_id 自己能看到这条历史
            })
            st.rerun()

# 8. 底部信息
st.sidebar.write("---")
st.sidebar.caption("🔄 版本 V8.2 | 记忆隔离版")
