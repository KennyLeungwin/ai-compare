import streamlit as st
import os
from openai import OpenAI
from typing import Optional

# 1. 页面配置
st.set_page_config(
    page_title="六模型 AI 对话助手 (V8.3 - 稳定性增强版)",
    layout="wide",
    page_icon="🤖"
)
st.title("🧠 六模型聊天对比 (V8.3 - 修复 Kimi & 渲染报错)")

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
            "temperature": 1.0,  # Kimi 默认建议 1.0
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
if "thinking_mode" not in st.session_state:
    st.session_state.thinking_mode = True
if "kimi_k25_enabled" not in st.session_state:
    st.session_state.kimi_k25_enabled = True

# 4. 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置面板")
    enabled_models = {}
    for model_id, config in MODEL_CONFIG.items():
        default_val = False if model_id == "gpt" else True
        enabled_models[model_id] = st.checkbox(f"{config['emoji']} {config['name']}", value=default_val, key=f"check_{model_id}")
    
    st.write("---")
    st.subheader("🔑 API 状态")
    for model_id, config in MODEL_CONFIG.items():
        key = os.getenv(config['env_key'])
        st.caption(f"{config['emoji']} {config['name']}: {'✅' if key else '❌'}")

    with st.expander("高级设置"):
        st.session_state.thinking_mode = st.toggle("启用思考模式引导", value=st.session_state.thinking_mode)
        global_temp = st.slider("全局温度 (Kimi除外)", 0.0, 1.0, 0.7, 0.1)
        # 更新非 Kimi 模型的温度
        for mid in MODEL_CONFIG:
            if mid != "kimi":
                MODEL_CONFIG[mid]["params"]["temperature"] = global_temp
        
        st.session_state.kimi_k25_enabled = st.toggle("使用 Kimi K2.5 (强制 Temp=1.0)", value=st.session_state.kimi_k25_enabled)

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 5. 渲染历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        if "[参考回答" in content:
            content = content.split("]: ", 1)[-1]
        st.markdown(content)

# 6. 核心函数 (修复 Key 错误与温度限制)
def ask_ai(model_id: str, col_obj, messages: list) -> Optional[str]:
    if model_id not in MODEL_CONFIG: return None
    config = MODEL_CONFIG[model_id]
    
    with col_obj:
        st.markdown(f"### {config['emoji']} {config['name']}")
        api_key = os.getenv(config['env_key'])
        if not api_key:
            st.error("密钥缺失")
            return None
        
        try:
            client = OpenAI(api_key=api_key, base_url=config['base_url'])
            
            # 隔离上下文
            filtered = []
            for m in messages:
                if m["role"] == "user":
                    filtered.append(m.copy())
                elif m["role"] == "assistant" and m.get("model_id") == model_id:
                    filtered.append({"role": "assistant", "content": m["content"]})
            
            # 提示词增强
            if st.session_state.thinking_mode:
                filtered[-1]["content"] += "\n\n请展示思考过程并给出最终答案。"

            # 参数准备：针对 Kimi 特殊处理
            target_model = config['model']
            run_params = config['params'].copy()
            
            if model_id == "kimi":
                if st.session_state.kimi_k25_enabled:
                    target_model = "kimi-k2.5"
                run_params["temperature"] = 1.0  # 强制为 1.0 解决 Error 400
            
            if model_id == "deepseek":
                run_params.pop("stream", None)

            with st.status(f"{config['emoji']} 思考中...", expanded=False) as status:
                resp = client.chat.completions.create(
                    model=target_model,
                    messages=filtered,
                    **run_params
                )
                answer = resp.choices[0].message.content
                status.update(label="完成！", state="complete")
            
            st.markdown(answer)
            return answer

        except Exception as e:
            st.error(f"调用失败: {str(e)}")
            return None

# 7. 主逻辑
if prompt := st.chat_input("输入问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    active_ids = [mid for mid, ok in enabled_models.items() if ok]
    if active_ids:
        cols = st.columns(len(active_ids))
        current_res = {}
        
        for i, mid in enumerate(active_ids):
            res = ask_ai(mid, cols[i], st.session_state.messages)
            if res: current_res[mid] = res
        
        if current_res:
            # 选择一个作为参考记忆存入（防止对话上下文无限膨胀）
            ref_id = active_ids[0] 
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"[参考回答 - {MODEL_CONFIG[ref_id]['name']}]: {current_res[ref_id]}",
                "model_id": ref_id
            })
            st.rerun()
