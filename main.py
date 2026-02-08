import streamlit as st
import os
from openai import OpenAI
from typing import Optional
import streamlit.components.v1 as components
import uuid
import json

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
            "name": "Gemini 2.0 Flash", 
            "model": "gemini-2.0-flash", 
            "emoji": "✨",
            # 优化：使用标准 v1beta OpenAI 适配地址
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", 
            "web_url": "https://aistudio.google.com/",
            "env_key": "GEMINI_API_KEY", 
            "system": (
                "你是 Gemini 2.0 Flash，由 Google 开发的最新一代多模态大模型。\n"
                "你擅长超长上下文处理、极速响应和复杂的跨领域任务。\n"
                "请以专业、清晰且富有逻辑的方式回答用户。"
            ),
            "params": {
                "temperature": 1.0, 
                "max_tokens": 65536, # Gemini 2.0 支持极大的输出长度
                "top_p": 0.95
            }
        },
    # 【Kimi优化】修复base_url空格问题，优化配置
    "kimi": {
        "name": "Kimi K2.5", 
        "model": "kimi-k2.5",  # 确认正确的模型ID
        "emoji": "🌙",
        "base_url": "https://api.moonshot.cn/v1",  # 修复：去除末尾空格！
        "web_url": "https://kimi.moonshot.cn",
        "env_key": "KIMI_API_KEY",
        # 优化系统提示词
        "system": (
            "你是Kimi K2.5，由月之暗面（Moonshot AI）开发的先进多模态大语言模型。\n\n"
            "核心能力：\n"
            "1. 256K超长上下文窗口，可处理长篇文档和复杂多轮对话\n"
            "2. 1万亿参数MoE架构，32B激活参数，擅长推理、编程、分析任务\n"
            "3. 原生Agentic能力，支持工具调用和多步骤任务分解\n"
            "4. 深度思考模式，可展示完整推理链条\n\n"
            "回答准则：\n"
            "- 复杂问题先拆解分析，再逐步解决\n"
            "- 编程任务提供完整可运行代码\n"
            "- 长文档保持全局一致性\n"
            "- 多轮对话记住早期决策，保持连贯性"
        ),
        # 官方推荐参数
        "params": {
            "temperature": 1.0,      # 必须固定1.0
            "max_tokens": 32768,     # 默认32K
            "top_p": 0.95,          # 官方推荐
        }
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
        "name": "Mistral AI (Mixtral-8x22B)",
        "model": "mistral-large-latest",
        "base_url": "https://api.mistral.ai/v1",
        "web_url": "https://chat.mistral.ai/",
        "emoji": "🦉",
        "env_key": "MISTRAL_API_KEY",
        "system": """你是 Le Chat，由 Mistral AI 创建的 AI 助手。
1. 以简洁、专业的方式回答问题。
2. 如果用户问及你的身份，回答："我是 Le Chat，由 Mistral AI 创建的 AI 助手。"
3. 避免提及模型版本或技术细节，除非用户明确要求。
4. 优先解决用户的问题，保持回答的实用性和准确性。""",
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
if "enable_qwen_search" not in st.session_state:
    st.session_state.enable_qwen_search = False
if "kimi_k25_enabled" not in st.session_state:
    st.session_state.kimi_k25_enabled = True
# 【Kimi优化】新增Kimi专属状态
if "kimi_thinking_enabled" not in st.session_state:
    st.session_state.kimi_thinking_enabled = True
if "thinking_mode" not in st.session_state:
    st.session_state.thinking_mode = True
# 新增：DeepSeek回答风格配置
if "deepseek_style" not in st.session_state:
    st.session_state.deepseek_style = "detailed"
# 【DeepSeek优化】新增专属状态
if "deepseek_reasoning_level" not in st.session_state:
    st.session_state.deepseek_reasoning_level = "medium"
if "deepsearch_enabled" not in st.session_state:
    st.session_state.deepsearch_enabled = False
if "math_mode" not in st.session_state:
    st.session_state.math_mode = False
if "show_token_usage" not in st.session_state:
    st.session_state.show_token_usage = True

# 4. 通用复制按钮函数（安全、带反馈）
def copy_button(text: str, key: str, label: str = "📋 复制"):
    """
    真·复制到系统剪贴板（浏览器 navigator.clipboard）。
    key 必须唯一，否则组件会互相干扰。
    """
    btn_id = f"copy_btn_{key}"
    msg_id = f"copy_msg_{key}"

    # 用 json.dumps 生成安全的 JS 字符串字面量（比 html.escape 稳得多）
    js_text = json.dumps(text or "")

    components.html(
        f"""
        <div style="display:flex; align-items:center; gap:10px; margin:6px 0 14px 0;">
          <button id="{btn_id}"
            style="
              padding:6px 10px;
              border-radius:8px;
              border:1px solid rgba(49,51,63,0.2);
              background: white;
              cursor: pointer;
              font-size: 14px;
            "
          >{label}</button>
          <span id="{msg_id}" style="font-size:12px; opacity:0.7;"></span>
        </div>

        <script>
          (function() {{
            const btn = document.getElementById("{btn_id}");
            const msg = document.getElementById("{msg_id}");
            const text = {js_text};

            if (!btn) return;

            btn.addEventListener("click", async () => {{
              try {{
                await navigator.clipboard.writeText(text);
                msg.textContent = "✅ 已复制";
                setTimeout(() => msg.textContent = "", 1200);
              }} catch (e) {{
                msg.textContent = "❌ 复制失败（权限/非HTTPS/iframe限制）";
              }}
            }});
          }})();
        </script>
        """,
        height=45,
    )

# 5. 辅助函数：隔离记忆
def get_isolated_messages(model_id, current_prompt):
    cfg = MODEL_CONFIG[model_id]
    msgs = [{"role": "system", "content": cfg['system']}]
    
    prompt_to_send = current_prompt
    if st.session_state.thinking_mode:
        if model_id == "deepseek":
            thinking_prompt = "\n\n【请使用Chain-of-Thought逐步推理】\n"
            
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
            elif st.session_state.deepseek_style == "concise":
                thinking_prompt = "\n\n请直接给出最准确的答案。"
            elif st.session_state.deepseek_style == "creative":
                thinking_prompt += (
                    "✨ 请展现创意和想象力：\n"
                    "• 不拘泥于常规思维路径\n"
                    "• 展现独特的视角和见解\n"
                    "• 语言生动有趣，富有感染力\n"
                    "• 在合理范围内大胆创新"
                )
            else:
                thinking_prompt = "\n\n请一步步推理并给出最终答案。"
            
            # 【DeepSeek优化】数学模式增强
            if st.session_state.math_mode and any(kw in current_prompt for kw in ["数学", "计算", "方程", "公式", "算", "+", "-", "*", "/", "="]):
                thinking_prompt += "\n\n【数学模式】请特别注意：\n1. 每个计算步骤都要清晰展示\n2. 使用LaTeX格式表示数学公式\n3. 验证计算结果的合理性\n4. 提供多种解法（如适用）"
            
            prompt_to_send += thinking_prompt
        
        # 【Kimi优化】新增Kimi专属思考提示
        elif model_id == "kimi":
            thinking_prompt = "\n\n【深度思考模式】\n"
            
            if any(kw in current_prompt for kw in ["代码", "编程", "debug", "code", "函数", "算法"]):
                thinking_prompt += (
                    "请按以下步骤处理编程任务：\n"
                    "1. 需求解析：明确功能需求、输入输出、边界条件\n"
                    "2. 方案设计：选择算法和数据结构，说明复杂度\n"
                    "3. 代码实现：编写完整可运行代码，包含注释\n"
                    "4. 测试验证：提供测试用例，包括边界情况\n"
                    "5. 优化建议：指出性能瓶颈和改进方向"
                )
            elif any(kw in current_prompt for kw in ["分析", "比较", "为什么", "评估"]):
                thinking_prompt += (
                    "请使用结构化分析框架：\n"
                    "1. 问题拆解：识别核心要素和相互关系\n"
                    "2. 多角度分析：从技术、业务、用户等维度展开\n"
                    "3. 证据支撑：引用相关原理、数据或最佳实践\n"
                    "4. 权衡评估：分析各方案的优缺点\n"
                    "5. 结论建议：给出明确、可落地的建议"
                )
            elif len(current_prompt) > 2000:
                thinking_prompt += (
                    "这是一篇长文档，请利用长上下文优势：\n"
                    "1. 整体把握：先总结核心主题和整体结构\n"
                    "2. 关键点提取：识别重要论点、数据、结论\n"
                    "3. 深度解读：对关键部分进行详细分析\n"
                    "4. 关联整合：将不同部分的信息关联起来"
                )
            else:
                thinking_prompt += (
                    "请展示思考过程：\n"
                    "• 理解问题核心诉求\n"
                    "• 分析关键信息和约束条件\n"
                    "• 逻辑推导，逐步构建答案\n"
                    "• 验证结论的准确性和完整性\n"
                    "• 给出清晰、准确的最终回答"
                )
            
            prompt_to_send += thinking_prompt
        
        else:
            prompt_to_send += "\n\n请详细展示你的思考步骤，然后再给出最终回答。"

    for m in st.session_state.messages:
        if m["role"] == "user":
            msgs.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant" and m.get("model_id") == model_id:
            content = m["content"].split("]: ", 1)[-1] if "]: " in m["content"] else m["content"]
            msgs.append({"role": "assistant", "content": content})
    
    msgs.append({"role": "user", "content": prompt_to_send})
    return msgs

# 6. 渲染历史消息（支持复制每条助手回答）
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        display_content = msg["content"].split("]: ", 1)[-1] if "]: " in msg["content"] else msg["content"]
        st.markdown(display_content)

        # 仅对助手消息显示复制按钮
        if msg["role"] == "assistant":
            mid = msg.get("model_id", "unknown")
            model_name = MODEL_CONFIG.get(mid, {}).get("name", "Assistant")
            copy_button(
                f"[{model_name}]\n\n{display_content}",
                key=f"hist_{idx}_{mid}",
                label="📋 复制这条回答"
            )

# 7. 侧边栏 UI (完整回归)
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
    
    # 【DeepSeek优化】添加独立配置面板
    if enabled_models.get("deepseek"):
        with st.expander("🚀 DeepSeek 专属设置", expanded=True):
            st.caption("🚀 业界领先推理能力 | 128K上下文 | 代码生成专家")
            
            # 推理强度配置
            reasoning_level = st.select_slider(
                "推理强度配置", 
                options=["low", "medium", "high"],
                value=st.session_state.deepseek_reasoning_level,
                format_func=lambda x: {
                    "low": "轻度推理 - 快速响应",
                    "medium": "平衡模式 - 推荐", 
                    "high": "深度推理 - 最准确"
                }.get(x, x)
            )
            st.session_state.deepseek_reasoning_level = reasoning_level
            
            # 回答风格配置（扩展选项）
            style_options = ["detailed", "concise", "technical", "educational", "creative"]
            style_labels = {
                "detailed": "详细模式 - 展示完整推理过程",
                "concise": "简洁模式 - 直接给出答案",
                "technical": "技术模式 - 专业术语和详细分析",
                "educational": "教育模式 - 分步讲解，适合学习",
                "creative": "创意模式 - 灵活发挥，适合写作和创意"
            }
            
            selected_label = st.selectbox(
                "回答风格配置",
                options=[style_labels[opt] for opt in style_options],
                index=style_options.index(st.session_state.deepseek_style) if st.session_state.deepseek_style in style_options else 0,
                key="deepseek_style_select"
            )
            
            for key, label in style_labels.items():
                if label == selected_label:
                    st.session_state.deepseek_style = key
                    break
            
            # 实时网络搜索
            st.session_state.deepsearch_enabled = st.toggle(
                "启用实时网络搜索", 
                value=st.session_state.deepsearch_enabled,
                help="需要API Key支持联网搜索功能"
            )
            
            # 数学专用模式
            st.session_state.math_mode = st.toggle(
                "数学专用模式",
                value=st.session_state.math_mode,
                help="针对数学问题优化，增强计算精度和步骤展示"
            )
            
            # Token使用监控
            st.session_state.show_token_usage = st.toggle(
                "显示Token使用详情",
                value=st.session_state.show_token_usage,
                help="显示详细的Token使用统计"
            )
    
    with st.expander("高级设置"):
        st.subheader("🧠 思考模式")
        st.session_state.thinking_mode = st.toggle("启用思考模式", value=st.session_state.thinking_mode, help="思考模式会显式引导模型进行逻辑推理")
        
        # 【DeepSeek优化】保持原有配置兼容性
        reasoning_level = st.select_slider(
            "DeepSeek 推理强度", 
            options=["low", "medium", "high"],
            value=st.session_state.deepseek_reasoning_level,
            format_func=lambda x: {
                "low": "轻度推理 - 快速响应",
                "medium": "平衡模式 - 推荐", 
                "high": "深度推理 - 最准确"
            }.get(x, x)
        )
        st.session_state.deepseek_reasoning_level = reasoning_level
        
        # 为DeepSeek添加回答风格选项
        style_options = ["detailed", "concise", "technical", "educational"]
        style_labels = {
            "detailed": "详细模式 - 展示完整推理过程",
            "concise": "简洁模式 - 直接给出答案",
            "technical": "技术模式 - 专业术语和详细分析",
            "educational": "教育模式 - 分步讲解，适合学习"
        }
        
        current_index = style_options.index(st.session_state.deepseek_style) if st.session_state.deepseek_style in style_options else 0
        
        selected_label = st.selectbox(
            "DeepSeek回答风格",
            options=[style_labels[opt] for opt in style_options],
            index=current_index,
            key="deepseek_style_select_legacy"
        )
        
        for key, label in style_labels.items():
            if label == selected_label:
                st.session_state.deepseek_style = key
                break

        global_temp = st.slider("全局温度", 0.0, 1.0, 0.7)
        for mid in MODEL_CONFIG:
            if mid != "kimi": # Kimi K2.5 固定为 1
                MODEL_CONFIG[mid]["params"]["temperature"] = global_temp

    if enabled_models.get("qwen"):
        with st.expander("🌸 Qwen 专属增强"):
            st.session_state.enable_qwen_search = st.checkbox("🔍 启用联网搜索", value=st.session_state.enable_qwen_search)

    # 【Kimi优化】升级侧边栏配置面板
    if enabled_models.get("kimi"):
        with st.expander("🌙 Kimi K2.5 专属增强", expanded=True):
            st.caption("🚀 256K上下文 | 1T参数MoE | 原生Agentic能力")
            
            # 深度思考模式控制
            st.session_state.kimi_thinking_enabled = st.toggle(
                "启用深度思考模式", 
                value=st.session_state.kimi_thinking_enabled,
                help="展示Kimi的完整推理链条，适合复杂分析任务"
            )
            
            # 模型选择提示
            st.info("💡 如遇404错误，请确认：\n1. API Key已开通K2.5访问权限\n2. 账户余额充足\n3. 模型ID正确：kimi-k2.5")
            
            # 温度提示（固定1.0）
            st.caption("🌡️ 温度固定为1.0（K2.5官方推荐值）")

    st.write("---")
    if st.button("🗑️ 清空所有对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 8. 核心对话逻辑
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
                    
                    # 【DeepSeek优化】智能参数调整
                    if mid == "deepseek":
                        # 设置推理强度
                        params["reasoning_effort"] = st.session_state.deepseek_reasoning_level
                        
                        # 根据问题类型智能调整
                        if any(kw in prompt for kw in ["数学", "计算", "方程", "算数"]):
                            params["temperature"] = 0.1
                            params["reasoning_effort"] = "high"
                        elif any(kw in prompt for kw in ["代码", "编程", "算法", "数据结构"]):
                            params["temperature"] = 0.3
                            params["reasoning_effort"] = "medium"
                        elif any(kw in prompt for kw in ["创意", "写作", "故事", "诗歌"]):
                            params["temperature"] = 0.9
                            params["reasoning_effort"] = "low"
                        
                        # 根据输入长度调整输出长度
                        if len(prompt) > 1000:
                            params["max_tokens"] = 16384
                        elif len(prompt) > 3000:
                            params["max_tokens"] = 32768
                        
                        # 启用实时搜索
                        if st.session_state.deepsearch_enabled:
                            params["stream"] = True
                    
                    # 【Kimi优化】Kimi专属参数处理
                    if mid == "kimi":
                        # 强制锁定官方推荐参数
                        params["temperature"] = 1.0
                        params["top_p"] = 0.95
                        
                        # 动态调整max_tokens
                        if len(prompt) > 5000:
                            params["max_tokens"] = 65536
                        elif any(kw in prompt for kw in ["代码", "编程", "写作", "长文", "详细"]):
                            params["max_tokens"] = 32768
                        else:
                            params["max_tokens"] = 16384
                        
                        # 启用原生Thinking模式（通过extra_body）
                        if st.session_state.kimi_thinking_enabled:
                            params["extra_body"] = {"thinking": {"type": "enabled"}}
                    
                    # 其他模型的特殊处理保持不变
                    if mid == "qwen" and st.session_state.enable_qwen_search:
                        params["enable_search"] = True

                    # 状态显示优化
                    if mid == "deepseek":
                        status_text = "🚀 DeepSeek深度推理中..." if st.session_state.thinking_mode else "🚀 DeepSeek回答中..."
                        if st.session_state.deepsearch_enabled:
                            status_text = "🌐 DeepSeek联网搜索中..."
                    elif mid == "kimi":
                        status_text = "🌙 Kimi深度思考中..." if st.session_state.kimi_thinking_enabled else "🌙 Kimi回答中..."
                    else:
                        status_text = f"{cfg['emoji']} 思考中..." if st.session_state.thinking_mode else f"{cfg['emoji']} 回答中..."
                    
                    usage_info = None
                    reasoning_content = None
                    
                    with st.status(status_text) as status:
                        resp = client.chat.completions.create(
                            model=cfg['model'],
                            messages=get_isolated_messages(mid, prompt),
                            **params
                        )
                        ans = resp.choices[0].message.content
                        
                        # 保存使用信息
                        if hasattr(resp, 'usage'):
                            usage_info = resp.usage
                        
                        # 捕获Kimi的reasoning_content
                        if mid == "kimi":
                            reasoning_content = getattr(resp.choices[0].message, 'reasoning_content', None)
                        
                        # 更新状态
                        if mid == "deepseek":
                            label = "✅ DeepSeek推理完成" if st.session_state.thinking_mode else "✅ DeepSeek回答完成"
                            if st.session_state.deepsearch_enabled:
                                label = "✅ DeepSeek联网搜索完成"
                            status.update(label=label, state="complete")
                        elif mid == "kimi":
                            label = "✅ Kimi深度思考完成" if st.session_state.kimi_thinking_enabled else "✅ Kimi回答完成"
                            status.update(label=label, state="complete")
                        else:
                            status.update(label=f"✅ {cfg['name']} 完成", state="complete")
                    
                    # 显示答案 + 复制按钮
                    st.markdown(ans)
                    copy_button(
                        f"[{cfg['name']}]\n\n{ans}",
                        key=f"single_{mid}_{uuid.uuid4().hex[:8]}",
                        label="📋 复制本模型答案"
                    )
                    
                    # 【DeepSeek优化】增强的资源展示
                    if mid == "deepseek" and usage_info:
                        if st.session_state.show_token_usage:
                            with st.expander("📊 DeepSeek推理资源详情", expanded=True):
                                cols_usage = st.columns(4)
                                with cols_usage[0]:
                                    st.metric("总Tokens", usage_info.total_tokens)
                                if hasattr(usage_info, 'completion_tokens'):
                                    with cols_usage[1]:
                                        st.metric("生成Tokens", usage_info.completion_tokens)
                                if hasattr(usage_info, 'prompt_tokens'):
                                    with cols_usage[2]:
                                        st.metric("提示Tokens", usage_info.prompt_tokens)
                                if hasattr(usage_info, 'reasoning_tokens'):
                                    with cols_usage[3]:
                                        st.metric("推理Tokens", usage_info.reasoning_tokens)
                            
                            # 推理效率分析
                            if hasattr(usage_info, 'prompt_tokens') and hasattr(usage_info, 'completion_tokens'):
                                efficiency = (usage_info.completion_tokens / usage_info.prompt_tokens) if usage_info.prompt_tokens > 0 else 0
                                st.caption(f"🧮 推理效率: {efficiency:.2f} (生成/提示比率)")
                    
                    # 【Kimi优化】Kimi专属展示
                    if mid == "kimi":
                        # 显示思考过程
                        if reasoning_content and st.session_state.kimi_thinking_enabled:
                            with st.expander("🧠 Kimi的思考过程", expanded=False):
                                st.markdown(reasoning_content)
                        
                        # 显示Token使用情况
                        if usage_info:
                            with st.expander("📊 Token使用情况", expanded=False):
                                cols_usage = st.columns(3)
                                with cols_usage[0]:
                                    st.metric("输入Tokens", usage_info.prompt_tokens)
                                with cols_usage[1]:
                                    st.metric("输出Tokens", usage_info.completion_tokens)
                                with cols_usage[2]:
                                    st.metric("总Tokens", usage_info.total_tokens)
                                
                                ctx_percent = (usage_info.prompt_tokens / 256000) * 100
                                st.progress(min(ctx_percent/100, 1.0), 
                                           text=f"上下文利用率: {ctx_percent:.1f}% (256K窗口)")
                    
                    new_responses.append({"mid": mid, "ans": ans})
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # 【DeepSeek优化】增强的错误处理
                    if mid == "deepseek":
                        st.error(f"DeepSeek调用失败: {error_msg[:200]}")
                        
                        # 根据错误类型提供针对性建议
                        if "401" in error_msg or "unauthorized" in error_msg.lower():
                            st.error("🚨 401错误：API Key无效")
                            st.info("""
                            **DeepSeek专属排查建议:**
                            1. 检查API密钥是否在 https://platform.deepseek.com/api_keys 创建
                            2. 确认账户余额充足（新用户有免费额度）
                            3. 检查网络连接，特别是国际网络访问
                            4. 如果使用代理，请确保代理设置正确
                            """)
                        elif "429" in error_msg:
                            st.error("🚨 429错误：请求过于频繁")
                            st.info("DeepSeek有请求频率限制，请稍后再试")
                        elif "model_not_found" in error_msg.lower():
                            st.error("🚨 模型未找到")
                            st.info("请确认模型名称是否正确：`deepseek-reasoner`")
                        else:
                            st.info(f"""
                            **DeepSeek专属排查建议:**
                            - 错误详情: {error_msg[:300]}
                            - 请查看 https://platform.deepseek.com/api-docs 获取详细API文档
                            - 或访问 https://platform.deepseek.com/ 检查账户状态
                            """)
                    
                    # 【Kimi优化】增强错误处理
                    elif mid == "kimi":
                        st.error(f"Kimi调用失败: {error_msg[:200]}")
                        
                        # 根据错误类型提供针对性建议
                        if "404" in error_msg or "not_found" in error_msg.lower():
                            st.error("🚨 404错误：模型未找到")
                            st.info("""
                            **可能原因及解决方案:**
                            
                            1. **模型ID错误** 
                               - 当前使用的模型ID: `kimi-k2.5`
                               - 请确认这是您账户中可用的模型ID
                               - 可尝试在 https://platform.moonshot.cn/docs/guide/kimi-k2-5-quickstart 查看最新模型名称
                            
                            2. **API Key无K2.5权限**
                               - Kimi K2.5需要单独申请访问权限
                               - 请在 https://platform.moonshot.cn 查看您的可用模型列表
                               - 如未开通，可申请开通或联系客服
                            
                            3. **账户余额不足**
                               - 检查账户是否有可用余额
                               - 新用户通常有免费额度，但可能不包含K2.5
                            
                            4. **base_url错误**
                               - 当前使用的base_url: `https://api.moonshot.cn/v1`
                               - 请确认与您注册的区域一致（国内版/国际版）
                            """)
                        elif "401" in error_msg or "unauthorized" in error_msg.lower():
                            st.error("🚨 401错误：API Key无效")
                            st.info("请检查 KIMI_API_KEY 环境变量是否正确设置")
                        elif "429" in error_msg:
                            st.error("🚨 429错误：请求过于频繁")
                            st.info("请稍后再试，或检查账户的RPM/TPM限制")
                        else:
                            st.info(f"""
                            **一般排查建议:**
                            - 错误详情: {error_msg[:300]}
                            - 请查看 https://platform.moonshot.cn/docs/guide/faq 获取帮助
                            """)
                    
                    else:
                        st.error(f"调用失败: {error_msg[:100]}")

        # ✅ 一键复制全部模型回答（本轮汇总）
        if new_responses:
            all_text = "\n\n" + ("-" * 30) + "\n\n".join(
                [f"[{MODEL_CONFIG[item['mid']]['name']}]\n\n{item['ans']}" for item in new_responses]
            )

            st.write("---")
            st.subheader("🧾 本轮模型回答汇总")
            copy_button(
                all_text,
                key=f"all_{uuid.uuid4().hex[:8]}",
                label="📋 一键复制全部模型回答"
            )

        # 持久化
        if new_responses:
            st.session_state.messages.append({"role": "user", "content": prompt})
            for item in new_responses:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"[{MODEL_CONFIG[item['mid']]['name']}]: {item['ans']}",
                    "model_id": item['mid']
                })
