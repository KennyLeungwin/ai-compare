import streamlit as st
import os
from openai import OpenAI
from typing import Optional, Dict, Any

# 1. 页面配置
st.set_page_config(
    page_title="五模型 AI 对话助手 (DeepSeek 最新版)",
    layout="wide",
    page_icon="🤖"
)
st.title("🧠 五模型聊天对比 (V7.0 DeepSeek 最新版)")

# 2. 模型配置信息
MODEL_CONFIG = {
    "deepseek": {
        "name": "DeepSeek Reasoning",
        "model": "deepseek-reasoner",  # 最新推理模型
        "base_url": "https://api.deepseek.com/v1",
        "web_url": "https://chat.deepseek.com",
        "emoji": "🚀",
        "env_key": "DEEPSEEK_API_KEY",
        "params": {
            "temperature": 0.7,
            "max_tokens": 8192,  # 支持更长上下文
            "stream": False,
            "reasoning_effort": "medium"  # 推理强度设置
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
        
        # 更新DeepSeek参数
        MODEL_CONFIG["deepseek"]["params"]["reasoning_effort"] = reasoning_level
        
        temperature = st.slider(
            "全局温度",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="temperature"
        )
        
        # 更新所有模型的temperature参数
        for model_id in MODEL_CONFIG:
            MODEL_CONFIG[model_id]["params"]["temperature"] = temperature
    
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
    
    st.info("💡 提示：DeepSeek Reasoning 是当前最新推理模型")

# 5. 渲染聊天记录
for message in st.session_state.messages:
    display_content = message["content"]
    if "[参考回答]" in display_content:
        display_content = display_content.split("[参考回答]: ")[-1]
    with st.chat_message(message["role"]):
        st.markdown(display_content)

# 6. 优化的AI调用函数
def ask_ai(model_id: str, col_obj, messages: list) -> Optional[str]:
    """调用AI模型的通用函数"""
    config = MODEL_CONFIG[model_id]
    
    with col_obj:
        # 显示模型标题
        st.markdown(f"### {config['emoji']} [{config['name']}]({config['web_url']})")
        
        # 检查API Key
        api_key = os.getenv(config['env_key'])
        if not api_key:
            st.warning(f"请设置 {config['env_key']} 环境变量")
            return None
        
        # 检查模型是否启用
        if not enabled_models.get(model_id, True):
            st.info("⏸️ 模型已禁用")
            return None
        
        try:
            # 创建客户端
            client = OpenAI(
                api_key=api_key,
                base_url=config['base_url']
            )
            
            # 构建请求参数
            params = {
                "model": config['model'],
                "messages": messages,
                **config['params']
            }
            
            # 特殊处理DeepSeek的推理强度参数
            if model_id == "deepseek" and "reasoning_effort" in params:
                # 移除stream参数如果为False（某些API不需要）
                if not params["stream"]:
                    params.pop("stream", None)
            
            # 显示加载状态
            with st.status(f"{config['emoji']} 正在思考中...", expanded=False) as status:
                st.write(f"使用模型: {config['model']}")
                st.write(f"温度: {params.get('temperature', 0.7)}")
                
                # 调用API
                response = client.chat.completions.create(**params)
                
                # 获取响应
                answer = response.choices[0].message.content
                
                # 显示统计信息
                if hasattr(response, 'usage'):
                    usage = response.usage
                    st.write(f"Token使用: {usage.total_tokens} (输入: {usage.prompt_tokens}, 输出: {usage.completion_tokens})")
                
                status.update(label=f"{config['emoji']} 完成！", state="complete")
            
            # 显示回答内容
            st.markdown(answer)
            
            # 添加到会话状态以便后续使用
            st.session_state.model_responses[model_id] = {
                "answer": answer,
                "model": config['model'],
                "timestamp": st.session_state.get("query_time", "")
            }
            
            return answer
            
        except Exception as e:
            error_msg = str(e)
            st.error(f"调用失败: {error_msg[:100]}...")
            return None

# 7. 聊天输入
if prompt := st.chat_input("向AI模型提问..."):
    # 记录提问时间
    from datetime import datetime
    st.session_state.query_time = datetime.now().strftime("%H:%M:%S")
    
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 获取启用的模型列表
    enabled_model_ids = [mid for mid in MODEL_CONFIG.keys() if enabled_models.get(mid, True)]
    
    if not enabled_model_ids:
        st.warning("⚠️ 请至少启用一个模型")
        st.stop()
    
    # 创建响应列
    cols = st.columns(len(enabled_model_ids))
    
    # 并行调用所有启用的模型
    with st.spinner(f"正在调用 {len(enabled_model_ids)} 个AI模型..."):
        responses = {}
        
        for idx, model_id in enumerate(enabled_model_ids):
            if idx < len(cols):
                answer = ask_ai(model_id, cols[idx], st.session_state.messages)
                if answer:
                    responses[model_id] = answer
        
        # 8. 记忆同步 - 选择最佳回答作为参考
        if responses:
            # 优先使用DeepSeek的回答作为参考（如果可用）
            if "deepseek" in responses:
                ref_ans = responses["deepseek"]
                ref_model = "DeepSeek"
            elif "gemini" in responses:
                ref_ans = responses["gemini"]
                ref_model = "Gemini"
            elif "qwen" in responses:
                ref_ans = responses["qwen"]
                ref_model = "通义千问"
            else:
                # 选择第一个可用的回答
                ref_model = list(responses.keys())[0]
                ref_ans = responses[ref_model]
            
            # 添加到对话历史
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"[参考回答 - {ref_model}]: {ref_ans}"
            })
            
            # 显示响应统计
            st.success(f"✅ 收到 {len(responses)}/{len(enabled_model_ids)} 个模型的响应")
            
            # 提供导出选项
            with st.expander("📊 响应统计"):
                for model_id, config in MODEL_CONFIG.items():
                    if model_id in responses:
                        st.write(f"{config['emoji']} **{config['name']}**: ✅ 已响应")
                    elif enabled_models.get(model_id, False):
                        st.write(f"{config['emoji']} **{config['name']}**: ❌ 无响应")

# 9. 底部信息
st.sidebar.write("---")
st.sidebar.caption(f"🌐 当前支持 {len(MODEL_CONFIG)} 个AI模型")
st.sidebar.caption("🔄 版本 7.0 | 支持 DeepSeek Reasoning 最新版")

# 10. 环境变量检查提醒
missing_keys = []
for model_id, config in MODEL_CONFIG.items():
    if not os.getenv(config['env_key']) and enabled_models.get(model_id, True):
        missing_keys.append(config['name'])

if missing_keys:
    st.sidebar.warning(f"⚠️ 以下模型缺少API Key: {', '.join(missing_keys)}")
