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
        "name": "DeepSeek-R1 Reasoning", "model": "deepseek-reasoner", "emoji": "🚀",
        "base_url": "https://api.deepseek.com/v1", "web_url": "https://chat.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY", 
        "system": (
            "你【就是】DeepSeek最新版，深度求索公司开发的顶尖推理模型。\n\n"
            "你的核心优势：\n"
            "1. 拥有业界领先的推理能力，特别擅长复杂逻辑分析和数学计算\n"
            "2. 支持128K超长上下文，能处理长篇文档和复杂对话\n"
            "3. 具备出色的代码生成和调试能力\n"
            "4. 知识更新及时，涵盖最新信息\n\n"
            "回答准则：\n"
            "- 对于推理问题，务必展示完整的思考过程\n"
            "- 数学问题要逐步推导，展示计算步骤\n"
            "- 代码问题提供可运行、有注释的解决方案\n"
            "- 保持回答的严谨性和准确性\n\n"
            "特别注意：\n"
            "- 不需要在回答中重复你的模型名称\n"
            "- 专注用你的推理能力解决问题\n"
            "- 中英文问题都能很好处理"
        ),
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
        "env_key": "QWEN_API_KEY",
        "system": (
            "You are Qwen-Max, a large-scale language model developed by Tongyi Lab. "
            "You excel at reasoning, coding, multi-language understanding (including Chinese, English, and more), "
            "and answering questions accurately based on your knowledge or real-time web search when enabled. "
            "Respond in the same language as the user's query unless instructed otherwise. "
            "If web search is enabled, use up-to-date information and cite sources when possible. "
            "Be clear, concise, helpful, and professional."
        ),
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
# 新增：DeepSeek回答风格配置
if "deepseek_style" not in st.session_state:
    st.session_state.deepseek_style = "detailed"

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
        
        # 修复：简化DeepSeek推理强度配置
        reasoning_level = st.select_slider(
            "DeepSeek 推理强度", 
            options=["low", "medium", "high"],
            value="medium",
            format_func=lambda x: {
                "low": "轻度推理 - 快速响应",
                "medium": "平衡模式 - 推荐", 
                "high": "深度推理 - 最准确"
            }.get(x, x)
        )
        MODEL_CONFIG["deepseek"]["params"]["reasoning_effort"] = reasoning_level
        
        # 为DeepSeek添加回答风格选项
        # 简化选项列表，避免复杂的元组结构
        style_options = ["detailed", "concise", "technical", "educational"]
        style_labels = {
            "detailed": "详细模式 - 展示完整推理过程",
            "concise": "简洁模式 - 直接给出答案",
            "technical": "技术模式 - 专业术语和详细分析",
            "educational": "教育模式 - 分步讲解，适合学习"
        }
        
        # 获取当前样式的索引
        current_index = style_options.index(st.session_state.deepseek_style) if st.session_state.deepseek_style in style_options else 0
        
        selected_label = st.selectbox(
            "DeepSeek回答风格",
            options=[style_labels[opt] for opt in style_options],
            index=current_index,
            key="deepseek_style_select"
        )
        
        # 根据选中的标签找到对应的键
        for key, label in style_labels.items():
            if label == selected_label:
                st.session_state.deepseek_style = key
                break

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
    
    # 思考模式注入 - 为DeepSeek定制专业的推理提示
    prompt_to_send = current_prompt
    if st.session_state.thinking_mode:
        if model_id == "deepseek":
            # 为DeepSeek定制专业的推理链提示
            thinking_prompt = "\n\n【请使用Chain-of-Thought逐步推理】\n"
            
            # 根据回答风格调整思考提示
            if st.session_state.deepseek_style == "detailed":
                thinking_prompt += (
                    "第一步：理解问题核心和约束条件\n"
                    "第二步：拆解问题，分析关键要素\n"
                    "第三步：逻辑推导，展示推理链条\n"
                    "第四步：验证结果，确保逻辑一致性\n"
                    "第五步：给出清晰、准确的最终答案\n"
                    "（如果涉及数学计算，请展示详细计算过程）"
                )
            elif st.session_state.deepseek_style == "technical":
                thinking_prompt += (
                    "1. 问题分析：识别核心问题和约束条件\n"
                    "2. 方法论选择：确定适用的分析方法\n"
                    "3. 逐步推导：展示严谨的逻辑推导过程\n"
                    "4. 结果验证：检查推导的合理性和一致性\n"
                    "5. 结论：给出准确的技术性结论"
                )
            elif st.session_state.deepseek_style == "educational":
                thinking_prompt += (
                    "📚 教学式思考：\n"
                    "• 首先，让我们理解这个问题在问什么\n"
                    "• 其次，我们一步步分析解决思路\n"
                    "• 然后，详细展示每个步骤的原理\n"
                    "• 最后，总结知识点和关键结论\n"
                    "（请用通俗易懂的方式讲解）"
                )
            else:  # concise模式
                thinking_prompt = "\n\n请一步步推理并给出最终答案。"
            
            prompt_to_send += thinking_prompt
        else:
            # 其他模型保持原有提示
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
                    
                    # 为DeepSeek优化参数
                    if mid == "deepseek":
                        # 根据问题类型动态调整推理强度
                        if "数学" in prompt or "计算" in prompt or any(word in prompt.lower() for word in ["math", "calculate", "solve"]):
                            params["reasoning_effort"] = "high"  # 数学问题用最高推理强度
                        elif "代码" in prompt or "编程" in prompt or "program" in prompt.lower():
                            params["reasoning_effort"] = "medium"
                            params["temperature"] = 0.3  # 代码问题需要更确定性
                        elif "创意" in prompt or "写作" in prompt or any(word in prompt.lower() for word in ["creative", "write", "story"]):
                            params["temperature"] = 0.9  # 创意问题提高创造性
                            params["reasoning_effort"] = "medium"
                        
                        # 根据问题长度调整max_tokens
                        if len(prompt) > 500:
                            params["max_tokens"] = 16384  # 长问题需要更长回答
                    
                    # 其他模型的特殊处理保持不变
                    if mid == "qwen" and st.session_state.enable_qwen_search:
                        params["enable_search"] = True

                    # 状态显示优化
                    if mid == "deepseek":
                        status_text = "🚀 DeepSeek深度推理中..." if st.session_state.thinking_mode else "🚀 DeepSeek回答中..."
                    else:
                        status_text = f"{cfg['emoji']} 思考中..." if st.session_state.thinking_mode else f"{cfg['emoji']} 回答中..."
                    
                    with st.status(status_text) as status:
                        # DeepSeek在思考模式下展示推理过程
                        resp = client.chat.completions.create(
                            model=cfg['model'],
                            messages=get_isolated_messages(mid, prompt),
                            **params
                        )
                        ans = resp.choices[0].message.content
                        st.markdown(ans)
                        
                        # DeepSeek专属：显示推理资源使用信息
                        if mid == "deepseek" and hasattr(resp, 'usage'):
                            with st.expander("🧠 推理资源使用", expanded=False):
                                st.write(f"总Tokens: {resp.usage.total_tokens}")
                                if hasattr(resp.usage, 'completion_tokens'):
                                    st.write(f"生成Tokens: {resp.usage.completion_tokens}")
                                if hasattr(resp.usage, 'prompt_tokens'):
                                    st.write(f"提示Tokens: {resp.usage.prompt_tokens}")
                        
                        if mid == "deepseek":
                            status.update(label=f"✅ DeepSeek推理完成" if st.session_state.thinking_mode else f"✅ DeepSeek回答完成", state="complete")
                        else:
                            status.update(label=f"✅ {cfg['name']} 完成", state="complete")
                    
                    new_responses.append({"mid": mid, "ans": ans})
                    
                except Exception as e:
                    if mid == "deepseek":
                        # 为DeepSeek提供更详细的错误信息
                        st.error(f"DeepSeek调用失败: {str(e)[:150]}")
                        with st.expander("DeepSeek专属排查建议"):
                            st.write("""
                            DeepSeek API 常见问题排查：
                            1. 检查API密钥是否在 https://platform.deepseek.com/api_keys 创建
                            2. 确认账户余额充足（新用户有免费额度）
                            3. 推理模型需要指定 reasoning_effort 参数
                            4. 检查网络连接，特别是国际网络访问
                            """)
                    else:
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
