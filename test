import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化客户端
@st.cache_resource
def get_client():
    return OpenAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        base_url="https://api.mistral.ai/v1"
    )

client = get_client()

# 页面配置
st.set_page_config(
    page_title="Mistral API 测试",
    page_icon="🦉",
    layout="wide"
)

st.title("🦉 Mistral API 测试工具")

# 1. 测试模型列表
if st.button("🔍 查看可用模型列表"):
    with st.spinner("获取模型列表..."):
        try:
            models = client.models.list()
            st.success("成功获取模型列表！")
            st.subheader("可用模型:")
            for model in models.data:
                st.code(model.id)
        except Exception as e:
            st.error(f"获取模型列表失败: {e}")

# 2. 交互式测试
st.divider()
st.subheader("📝 交互式测试")

col1, col2 = st.columns(2)
with col1:
    model_name = st.selectbox(
        "选择模型:",
        [
            "mistral-tiny",
            "mistral-small",
            "mistral-medium",
            "mixtral-8x7b",
            "mixtral-8x22b"  # 可能需要申请
        ],
        index=3  # 默认选择 mixtral-8x7b
    )

    temperature = st.slider("温度:", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.number_input("最大 tokens:", 1, 32768, 1000)

with col2:
    prompt = st.text_area(
        "输入你的问题:",
        "介绍一下 Mixtral 模型的特点",
        height=150
    )

    stream_output = st.checkbox("启用流式输出", False)

if st.button("🚀 发送请求"):
    if not os.getenv("MISTRAL_API_KEY"):
        st.error("请先在 .env 文件中配置 MISTRAL_API_KEY")
    else:
        with st.spinner("正在请求 Mistral API..."):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream_output
                )

                if stream_output:
                    st.subheader("流式输出:")
                    response_box = st.empty()
                    full_response = ""
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            response_box.markdown(full_response)
                else:
                    st.subheader("响应结果:")
                    st.markdown(response.choices[0].message.content)

                    if hasattr(response, 'usage'):
                        st.caption(f"使用 Tokens: {response.usage.total_tokens}")

            except Exception as e:
                st.error(f"请求失败: {e}")

# 3. JSON 模式测试
st.divider()
st.subheader("📊 JSON 模式测试")

json_prompt = st.text_area(
    "输入 JSON 格式的请求:",
    "返回 JSON 格式的上海基本信息，包括名称、人口、面积（单位：平方公里）",
    height=100
)

if st.button("📊 获取 JSON 响应"):
    with st.spinner("正在生成 JSON 响应..."):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": json_prompt}],
                response_format={"type": "json_object"}
            )
            st.subheader("JSON 响应:")
            st.json(response.choices[0].message.content)
        except Exception as e:
            st.error(f"JSON 生成失败: {e}")

# 4. 显示 API 状态
st.divider()
st.subheader("🔧 API 状态")

if st.button("检查 API 连接"):
    try:
        # 简单请求测试连接
        client.models.list()
        st.success("✅ API 连接正常")
    except Exception as e:
        st.error(f"❌ API 连接失败: {e}")

# 侧边栏信息
st.sidebar.markdown("""
### Mistral API 测试工具

**功能:**
- 查看可用模型列表
- 交互式测试不同模型
- 测试流式输出
- 生成 JSON 格式响应
- 检查 API 连接状态

**注意:**
1. 需要在 `.env` 文件中配置 `MISTRAL_API_KEY`
2. `mixtral-8x22b` 需要申请权限
3. 流式输出可能需要更长时间
""")
