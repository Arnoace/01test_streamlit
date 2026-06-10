import streamlit as st

st.title("我的第一个 Streamlit 网页")

st.write("你好，世界！")

name = st.text_input("请输入你的名字")

if name:
    st.success(f"欢迎你，{name}！")