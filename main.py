import streamlit as st
import os
from openai import OpenAI
from typing import Optional, Dict, Any
from datetime import datetime

# 1. 页面配置
st.set_page_config(
    page_title="五模型 AI 对话助手 (DeepSeek + Qwen 最新版)",
    layout="wide",
    page_icon="🤖"
)
st.title("🧠 五模型聊天对比 (V7.1 DeepSeek + Qwen 增强版)")

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
        "name": "Gemini 2.0 Flash",
        "model": "gemini-2.0-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "web_url": "https://gemini.google.com",
        "emoji": "✨",
        "env_key": "GEMINI_API_KEY",
        "params": {
            "temperature": 0.7,
            "max_tokens": 2048,
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
    }
}

# 3. 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_responses" not in st.session_state:
    st.session_state.model_responses = {}

# 默认关闭 Qwen 搜索（可在侧边栏开启）
if "enable_qwen_search" not in st.session_state:
    st.session_state.enable_qwen_search = False

# 4. 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置面板")
    
    # 模型选择
    st.subheader("选择要使用的模型")
    enabled_models = {}
    for model_id, config in MODEL_CONFIG.items():
        enabled_models[model_id] = st.checkbox(
            f"{config['emoji']} {config['name']}",
            value=True,
            key=f"enable_{model_id}"
        )
    
    st.write("---")
    
    # API Key 状态检查
    st.subheader("🔑 API Key 状态")
    for model_id, config in MODEL_CONFIG.items():
        key = os.getenv(config['env_key'])
        status = "✅ 已配置" if key else "❌ 未配置"
        st.write(f"{config['emoji']} {config['name']}: {status}")
    
    st.write("---")
    
    # 高级设置
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
            key="temperature"
        )
        for model_id in MODEL_CONFIG:
            MODEL_CONFIG[model_id]["params"]["temperature"] = temperature

    # 🌸 Qwen 专属增强（仅当启用时显示）
    if enabled_models.get("qwen", False):
        with st.expander("🌸 Qwen 专属增强"):
            st.session_state.enable_qwen_search = st.checkbox(
                "🔍 启用联网搜索（实时获取最新信息）",
                value=st.session_state.enable_qwen_search,
                help="适用于新闻、股价、赛事、政策等时效性问题。开启后 Qwen 会自动检索网络。"
            )
            st.info("💡 提示：此功能由 DashScope 提供，需确保 Qwen API Key 有搜索权限")

    st.write("---")
    
    # 控制按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.model_responses = {}
            st.rerun()
    
    with col2:
        if st.button("🔄 刷新页面", use_container_width=True):
            st.rerun()
    
    st.info("💡 提示：DeepSeek Reasoning 与 Qwen-Max 均为当前最新推理模型")

# 5. 渲染聊天记录
for message in st.session_state.messages:
    display_content = message["content"]
    if "[参考回答]" in display_content:
        display_content = display_content.split("[参考回答]: ")[-1]
    with st.chat_message(message["role"]):
        st.markdown(display_content)

# 6. 优化的AI调用函数（含 Qwen 增强）
def ask_ai(model_id: str, col_obj, messages: list) -> Optional[str]:
    config = MODEL_CONFIG[model_id]
    
    with col_obj:
        st.markdown(f"### {config['emoji']} [{config['name']}]({config['web_url']})")
        
        api_key = os.getenv(config['env_key'])
        if not api_key:
            st.warning(f"请设置 {config['env_key']} 环境变量")
            return None
        
        if not enabled_models.get(model_id, True):
            st.info("⏸️ 模型已禁用")
            return None
        
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=config['base_url']
            )
            
            # 构建基础参数
            params = {
                "model": config['model'],
                "messages": messages,
                **config['params']
            }

            # === 🌸 Qwen 特性增强 ===
            if model_id == "qwen":
                # 启用联网搜索
                if st.session_state.get("enable_qwen_search", False):
                    params["enable_search"] = True
                
                # 智能结构化输出（检测用户是否要求 JSON/表格）
                last_user_msg = ""
                for msg in reversed(messages):
                    if msg["role"] == "user":
                        last_user_msg = msg["content"]
                        break
                if any(kw in last_user_msg.lower() for kw in ["json", "结构化", "表格", "格式化输出"]):
                    params["response_format"] = {"type": "json_object"}

            # === 🚀 DeepSeek 特殊处理 ===
            elif model_id == "deepseek" and "reasoning_effort" in params:
                if not params["stream"]:
                    params.pop("stream", None)

            # 调用 API
            with st.status(f"{config['emoji']} 正在思考中...", expanded=False) as status:
                st.write(f"使用模型: {config['model']}")
                st.write(f"温度: {params.get('temperature', 0.7)}")
                if model_id == "qwen" and params.get("enable_search"):
                    st.write("🌐 联网搜索: 已启用")
                if model_id == "qwen" and params.get("response_format"):
                    st.write("📄 输出格式: JSON")

                response = client.chat.completions.create(**params)
                answer = response.choices[0].message.content

                if hasattr(response, 'usage'):
                    usage = response.usage
                    st.write(f"Token使用: {usage.total_tokens} (输入: {usage.prompt_tokens}, 输出: {usage.completion_tokens})")
                
                status.update(label=f"{config['emoji']} 完成！", state="complete")

            st.markdown(answer)
            
            st.session_state.model_responses[model_id] = {
                "answer": answer,
                "model": config['model'],
                "timestamp": st.session_state.get("query_time", "")
            }
            return answer

        except Exception as e:
            error_msg = str(e)
            st.error(f"调用失败 ({config['name']}): {error_msg[:150]}...")
            return None

# 7. 聊天输入
if prompt := st.chat_input("向AI模型提问..."):
    st.session_state.query_time = datetime.now().strftime("%H:%M:%S")
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    enabled_model_ids = [mid for mid in MODEL_CONFIG.keys() if enabled_models.get(mid, True)]
    
    if not enabled_model_ids:
        st.warning("⚠️ 请至少启用一个模型")
        st.stop()
    
    cols = st.columns(len(enabled_model_ids))
    
    with st.spinner(f"正在调用 {len(enabled_model_ids)} 个AI模型..."):
        responses = {}
        for idx, model_id in enumerate(enabled_model_ids):
            if idx < len(cols):
                answer = ask_ai(model_id, cols[idx], st.session_state.messages)
                if answer:
                    responses[model_id] = answer
        
        if responses:
            # 优先级：DeepSeek > Qwen > Gemini > 其他
            ref_order = ["deepseek", "qwen", "gemini", "kimi", "gpt"]
            ref_model_id = next((mid for mid in ref_order if mid in responses), list(responses.keys())[0])
            ref_ans = responses[ref_model_id]
            ref_name = MODEL_CONFIG[ref_model_id]["name"]
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"[参考回答 - {ref_name}]: {ref_ans}"
            })
            
            st.success(f"✅ 收到 {len(responses)}/{len(enabled_model_ids)} 个模型的响应")
            
            with st.expander("📊 响应统计"):
                for model_id, config in MODEL_CONFIG.items():
                    if model_id in responses:
                        st.write(f"{config['emoji']} **{config['name']}**: ✅ 已响应")
                    elif enabled_models.get(model_id, False):
                        st.write(f"{config['emoji']} **{config['name']}**: ❌ 无响应")

# 8. 底部信息
st.sidebar.write("---")
st.sidebar.caption(f"🌐 当前支持 {len(MODEL_CONFIG)} 个AI模型")
st.sidebar.caption("🔄 版本 7.1 | 支持 DeepSeek Reasoning + Qwen-Max 增强版")
