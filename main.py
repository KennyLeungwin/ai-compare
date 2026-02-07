import streamlit as st
import os
from openai import OpenAI
from typing import Optional, Dict, Any
from datetime import datetime

# 1. 页面配置
st.set_page_config(
    page_title="五模型 AI 对话助手 (DeepSeek + Qwen + Gemini 2.5)",
    layout="wide",
    page_icon="🤖"
)
st.title("🧠 五模型聊天对比 (V7.2 多模型增强版)")

# 2. 模型配置信息（修复：去除所有 URL 末尾空格）
MODEL_CONFIG = {
    "deepseek": {
        "name": "DeepSeek Reasoning",
        "model": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com/v1",  # 去除空格
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
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",  # 去除空格
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
        "base_url": "https://api.moonshot.cn/v1",  # 去除空格
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
        "base_url": "https://openrouter.ai/api/v1",  # 去除空格
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
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 去除空格
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

# ========== 新增：Kimi 智能温度优化配置 ==========
KIMI_TEMPERATURE_OPTIMIZER = {
    "enabled": True,
    "model": "kimi-k2.5",
    "base_url": "https://api.moonshot.cn/v1",  # 确保无空格
    "max_tokens": 32768,
    "stream": True,
    "top_p": 0.95,
    
    "temperature_rules": {
        "base_offset": 0.2,
        "task_adjustments": {
            "coding": {
                "keywords": ["代码", "编程", "python", "javascript", "写个函数", "写个类", "程序", "script", "function", "class"],
                "offset": -0.5,
                "description": "🖥️ 代码生成"
            },
            "analysis": {
                "keywords": ["分析", "推理", "为什么", "如何", "步骤", "证明", "比较", "评估", "解释", "原因"],
                "offset": 0.1,
                "description": "🧠 深度分析"
            },
            "creative": {
                "keywords": ["写", "创作", "故事", "诗歌", "文章", "创意", "想象", "假如", "如果"],
                "offset": 0.3,
                "description": "✨ 创意写作"
            },
            "translation": {
                "keywords": ["翻译", "translate", "英文", "中文", "日文"],
                "offset": -0.2,
                "description": "🌐 翻译任务"
            },
            "math": {
                "keywords": ["计算", "数学", "公式", "solve", "equation", "math", "证明"],
                "offset": -0.4,
                "description": "🔢 数学计算"
            }
        },
        "context_adjustments": {
            "long_threshold": 8000,
            "offset": -0.1,
            "description": "📄 长文本优化"
        },
        "min_temp": 0.1,
        "max_temp": 1.5
    }
}

# 新增：Kimi 增强功能配置
if "kimi_smart_temp_enabled" not in st.session_state:
    st.session_state.kimi_smart_temp_enabled = True
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

# ========== 新增：智能温度计算函数 ==========
def calculate_kimi_temperature(global_temp: float, messages: list) -> tuple[float, str, dict]:
    """基于全局温度和对话内容，计算最适合 Kimi 的温度"""
    optimizer = KIMI_TEMPERATURE_OPTIMIZER["temperature_rules"]
    
    base_temp = global_temp + optimizer["base_offset"]
    adjustments = [f"基础: {global_temp} + {optimizer['base_offset']} = {base_temp}"]
    
    last_message = next(
        (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
        ""
    ).lower()
    
    task_type = "general"
    task_offset = 0
    task_desc = "📝 通用对话"
    
    for task_key, task_config in optimizer["task_adjustments"].items():
        if any(keyword in last_message for keyword in task_config["keywords"]):
            task_offset = task_config["offset"]
            task_type = task_key
            task_desc = task_config["description"]
            adjustments.append(f"{task_desc}: {task_offset}")
            break
    
    context_offset = 0
    context_desc = ""
    total_length = sum(len(msg.get("content", "")) for msg in messages)
    
    if total_length > optimizer["context_adjustments"]["long_threshold"]:
        context_offset = optimizer["context_adjustments"]["offset"]
        context_desc = optimizer["context_adjustments"]["description"]
        adjustments.append(f"{context_desc}: {context_offset}")
    
    final_temp = base_temp + task_offset + context_offset
    final_temp = max(optimizer["min_temp"], min(optimizer["max_temp"], final_temp))
    
    description = f"{task_desc}"
    if context_desc:
        description += f" + {context_desc}"
    
    debug_info = {
        "global_temp": global_temp,
        "base_temp": base_temp,
        "task_type": task_type,
        "task_offset": task_offset,
        "context_offset": context_offset,
        "final_temp": final_temp,
        "adjustments": adjustments
    }
    
    return final_temp, description, debug_info

# 4. 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置面板")
    
    st.subheader("选择要使用的模型")
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
            MODEL_CONFIG[model_id]["params"]["temperature"] = temperature
        
        st.divider()
        st.caption("🌙 **Kimi 智能温度优化**")
        
        base_with_offset = temperature + KIMI_TEMPERATURE_OPTIMIZER["temperature_rules"]["base_offset"]
        base_with_offset = max(0.1, min(1.5, base_with_offset))
        st.info(f"""
        当前全局温度: {temperature}
        Kimi 基础温度: {base_with_offset:.1f} (全局 + 0.2 偏移)
        
        实际调用时会根据任务类型自动调整：
        • 代码任务: -0.5 (更确定)
        • 创意写作: +0.3 (更发散)
        • 分析推理: +0.1 (平衡)
        • 长文本: -0.1 (更稳定)
        """)
        
        st.session_state.kimi_smart_temp_enabled = st.toggle(
            "启用 Kimi 智能温度优化",
            value=st.session_state.kimi_smart_temp_enabled,
            help="根据任务类型自动调整温度，获得最佳效果"
        )

    if enabled_models.get("qwen", False):
        with st.expander("🌸 Qwen 专属增强"):
            st.session_state.enable_qwen_search = st.checkbox(
                "🔍 启用联网搜索",
                value=st.session_state.enable_qwen_search
            )

    if enabled_models.get("kimi", False):
        with st.expander("🌙 Kimi 专属增强", expanded=True):
            st.session_state.kimi_streaming_enabled = st.toggle(
                "启用流式传输 (推荐)",
                value=st.session_state.kimi_streaming_enabled,
                help="避免长文本连接中断，必须启用"
            )
            
            st.session_state.kimi_thinking_mode = st.toggle(
                "启用 Thinking 模式",
                value=st.session_state.kimi_thinking_mode,
                help="展示推理过程，适合复杂问题"
            )
            
            if st.session_state.kimi_smart_temp_enabled:
                st.success("✅ 智能温度优化运行中")
                with st.popover("查看温度调整规则"):
                    rules = KIMI_TEMPERATURE_OPTIMIZER["temperature_rules"]["task_adjustments"]
                    for task, config in rules.items():
                        st.write(f"**{config['description']}**: {config['offset']:+.1f}")
                        st.caption(f"关键词: {', '.join(config['keywords'][:3])}...")
            else:
                st.info("使用全局温度设置")

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
    
    st.info("💡 提示：Gemini 2.5 Flash 现已支持超长上下文处理")

# 5. 渲染聊天记录
for message in st.session_state.messages:
    display_content = message["content"]
    if "[参考回答]" in display_content:
        display_content = display_content.split("[参考回答]: ")[-1]
    with st.chat_message(message["role"]):
        st.markdown(display_content)

# 6. 优化的AI调用函数
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
            client = OpenAI(
                api_key=api_key,
                base_url=config['base_url']
            )
            
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

            # ========== 新增：Kimi 智能优化处理 ==========
            elif model_id == "kimi":
                if st.session_state.get("kimi_smart_temp_enabled", False):
                    global_temp = st.session_state.get("global_temperature", 0.7)
                    smart_temp, task_desc, debug_info = calculate_kimi_temperature(global_temp, messages)
                    params["temperature"] = smart_temp
                    
                    # 升级模型和参数
                    params["model"] = KIMI_TEMPERATURE_OPTIMIZER["model"]
                    params["max_tokens"] = KIMI_TEMPERATURE_OPTIMIZER["max_tokens"]
                    params["stream"] = st.session_state.get("kimi_streaming_enabled", True)
                    params["top_p"] = KIMI_TEMPERATURE_OPTIMIZER["top_p"]
                    
                    st.caption(f"🌙 {task_desc} | 温度: {smart_temp:.1f} (全局 {global_temp})")
                    
                    if params["stream"]:
                        return _stream_kimi_response(client, params, messages, task_desc, debug_info)
                else:
                    if st.session_state.get("kimi_streaming_enabled", True):
                        params["model"] = KIMI_TEMPERATURE_OPTIMIZER["model"]
                        params["max_tokens"] = KIMI_TEMPERATURE_OPTIMIZER["max_tokens"]
                        params["stream"] = True
                        return _stream_kimi_response(client, params, messages, "标准模式", {})

            # 调用 API（非流式）
            with st.status(f"{config['emoji']} 正在思考中...", expanded=False) as status:
                st.write(f"使用模型: {config['model']}")
                response = client.chat.completions.create(**params)
                answer = response.choices[0].message.content
                
                if hasattr(response, 'usage'):
                    st.write(f"Tokens: {response.usage.total_tokens}")
                status.update(label=f"{config['emoji']} 完成！", state="complete")

            st.markdown(answer)
            st.session_state.model_responses[model_id] = {"answer": answer, "model": config['model']}
            return answer

        except Exception as e:
            st.error(f"调用失败: {str(e)[:100]}")
            return None

# ========== 新增：Kimi 流式响应处理函数 ==========
def _stream_kimi_response(client, params, messages, task_desc, debug_info) -> Optional[str]:
    """处理 Kimi 的流式输出"""
    full_response = ""
    reasoning_content = ""
    
    try:
        with st.status(f"🌙 Kimi 思考中... ({task_desc})", expanded=st.session_state.get("kimi_thinking_mode", False)) as status:
            response_placeholder = st.empty()
            
            thinking_placeholder = None
            if st.session_state.get("kimi_thinking_mode", False):
                thinking_expander = st.expander("🤔 推理过程", expanded=False)
                thinking_placeholder = thinking_expander.empty()
            
            stream = client.chat.completions.create(**params)
            
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        reasoning_content += delta.reasoning_content
                        if thinking_placeholder:
                            thinking_placeholder.markdown(reasoning_content)
                    
                    if delta.content:
                        full_response += delta.content
                        response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            status.update(label="✅ 完成！", state="complete")
        
        st.session_state.model_responses["kimi"] = {
            "answer": full_response,
            "model": params["model"],
            "temperature": params["temperature"],
            "task_type": task_desc,
            "reasoning": reasoning_content if reasoning_content else None
        }
        
        return full_response
        
    except Exception as e:
        st.error(f"Kimi 流式输出失败: {str(e)[:150]}")
        return None

# 7. 聊天输入逻辑
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
    with st.spinner(f"正在同步调用 {len(enabled_model_ids)} 个模型..."):
        responses = {}
        for idx, model_id in enumerate(enabled_model_ids):
            answer = ask_ai(model_id, cols[idx], st.session_state.messages)
            if answer:
                responses[model_id] = answer
        
        if responses:
            ref_order = ["deepseek", "qwen", "gemini", "kimi", "gpt"]
            ref_model_id = next((mid for mid in ref_order if mid in responses), list(responses.keys())[0])
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"[参考回答 - {MODEL_CONFIG[ref_model_id]['name']}]: {responses[ref_model_id]}"
            })

# 8. 底部信息
st.sidebar.write("---")
st.sidebar.caption("🔄 版本 7.2 | 融入 Gemini 2.5 Flash 性能优化")

if enabled_models.get("kimi", False):
    st.sidebar.divider()
    if st.session_state.get("kimi_smart_temp_enabled", False):
        st.sidebar.success("🌙 Kimi 智能温度优化运行中")
        current_global = st.session_state.get("global_temperature", 0.7)
        base_temp = current_global + 0.2
        st.sidebar.caption(f"全局: {current_global} | Kimi基础: {base_temp:.1f}")
    else:
        st.sidebar.info("🌙 Kimi 使用全局温度")
