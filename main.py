import streamlit as st
import os
from openai import OpenAI

st.set_page_config(page_title="🧠 中文 AI 对比助手", layout="wide")
st.title("🧠 中文 AI 对比助手")
st.caption("DeepSeek / GPT / 通义千问 三模型对比")

prompt = st.text_area("请输入你的问题：", height=120)

if st.button("🚀 开始对比", type="primary"):
    if not prompt.strip():
        st.warning("请输入问题")
    else:
        cols = st.columns(3)

        # DeepSeek
        with cols[0]:
            st.subheader("🤖 DeepSeek")
            try:
                client = OpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com/v1"
                )
                r = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(r.choices[0].message.content)
            except Exception as e:
                st.error("调用失败")

        # GPT（OpenRouter）
        with cols[1]:
            st.subheader("💬 GPT-3.5")
            try:
                client = OpenAI(
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1"
                )
                r = client.chat.completions.create(
                    model="openai/gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(r.choices[0].message.content)
            except Exception as e:
                st.error("调用失败")

        # Qwen
        with cols[2]:
            st.subheader("🌸 通义千问")
            try:
                client = OpenAI(
                    api_key=os.getenv("QWEN_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                r = client.chat.completions.create(
                    model="qwen-max-latest",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(r.choices[0].message.content)
            except Exception as e:
                st.error("调用失败")
