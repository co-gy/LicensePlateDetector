import streamlit as st
from PIL import Image
import requests
import numpy as np
import cv2
import pandas as pd
from det_rec import det_rec


def login(user, password):
    table = st.session_state.user_table
    if len(table.loc[table["user"] == user]["password"].values) == 0:
        return False
    else:
        value = f'{table.loc[table["user"] == user]["password"].values[0]}'
        return password == value
    
# 设置页面配置
st.set_page_config(page_title="车牌检测", initial_sidebar_state="collapsed", layout="wide")

# 初始化会话状态
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'history' not in st.session_state:
    st.session_state.history = []
if 'user_table' not in st.session_state:
    st.session_state.user_table = pd.read_csv("./user.csv")

# 登录界面
if not st.session_state.logged_in:
    st.title("登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    st.session_state.logged_in = login(username, password)
    if st.button("进入页面"):
        if st.session_state.logged_in:
            st.success("点击按钮进入应用")
        else:
            st.error("用户名或密码错误")
else:
    # 应用界面
    col1, col2, col3 = st.columns(3)

    with col1:
        uploaded_file = st.file_uploader("点击上传图片: ", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="上传图片", use_container_width=True)

    with col2:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            print("shape:", image_np.shape, type(image_np))
            pred, crop_image = det_rec(image_np)
            crop_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR)

            if pred is not None:
                st.markdown(f"<h2>车牌识别：</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3>{pred}</h3>", unsafe_allow_html=True)

            if crop_image is not None:
                st.image(Image.fromarray(crop_image), caption="车牌检测结果", use_container_width=True)

            st.session_state.history.append((image, pred))

    with col3:
        for image, text in st.session_state.history:
            with st.expander(f"{text}"):
                st.image(image, caption=text, use_container_width=True)
