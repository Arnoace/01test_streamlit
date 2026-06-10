import streamlit as st
import cv2
import numpy as np
import requests
import tempfile

# 后端 FastAPI 的接口地址
API_URL = "http://127.0.0.1:8000/predict_license_plate"

# ==================== 1. 网页页面基本配置 ====================
st.set_page_config(page_title="AI 车牌检测与识别系统", layout="wide")
st.title("🚗 智能车牌检测与文本识别系统 (前后端分离版)")
st.markdown("欢迎使用车牌系统。请在左侧选择模式并上传文件，系统将请求后台 FastAPI 服务进行实时处理。")

# ==================== 2. 侧边栏配置 ====================
st.sidebar.header("⚙️ 配置面板")
app_mode = st.sidebar.selectbox("选择功能模式", ["照片车牌识别（文字识别）", "视频车牌检测（仅定位）"])
conf_threshold = st.sidebar.slider("置信度阈值 (Confidence)", 0.0, 1.0, 0.25, 0.05)

st.sidebar.success("🚀 前端控制台就绪（模型已托管至 FastAPI 后端）")

# ==================== 3. 核心业务逻辑 ====================

# ----------------- 模式一：照片车牌识别 -----------------
if app_mode == "照片车牌识别（文字识别）":
    uploaded_image = st.sidebar.file_uploader("上传照片 (JPG/PNG格式)", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📸 原始与检测画面")
        image_placeholder = st.empty()
    with col2:
        st.subheader("🔤 车牌文字识别结果")
        ocr_placeholder = st.empty()

    if uploaded_image is not None:
        # 将上传的文件转为 OpenCV 格式供前端画框渲染
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # 将文件指针重置，准备通过 requests 发送给后端
        uploaded_image.seek(0)
        
        # 1. 组装请求数据，向 FastAPI 发送网络请求
        files = {"file": (uploaded_image.name, uploaded_image.read(), uploaded_image.type)}
        data = {"conf_threshold": conf_threshold}
        
        with st.spinner("正在请求后台大模型服务..."):
            try:
                response = requests.post(API_URL, files=files, data=data)
                res_json = response.json()
            except Exception as e:
                st.error(f"❌ 无法连接到后台 FastAPI 服务，请检查后端是否启动。错误信息: {e}")
                st.stop()

        if res_json.get("success"):
            plates = res_json.get("plates", [])
            ocr_result_text = f"🟢 **检测结果**: 🚀 后端返回：画面中共发现 {res_json['plate_count']} 个车牌。\n\n---\n"
            
            # 2. 前端根据后端返回的坐标，在图片上画框并抠图展示
            annotated_img = img.copy()
            
            for plate in plates:
                x1, y1, x2, y2 = plate["box"]
                conf = plate["confidence"]
                text = plate["text"]
                
                # 在前端画框框
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated_img, f"Plate:{conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # 前端利用 OpenCV 抠图展示
                plate_crop = img[y1:y2, x1:x2]
                
                # 拼接右侧 Markdown 结果
                ocr_result_text += f"### 🆔 车牌 #{plate['id']}\n"
                ocr_result_text += f"- **置信度**: `{conf:.2f}`\n"
                ocr_result_text += f"- **识别文本**: <span style='color:red; font-size:24px; font-weight:bold;'>{text}</span>\n\n"
                
                # 渲染右侧裁剪小图
                if plate_crop.size > 0:
                    plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                    st.sidebar.image(plate_crop_rgb, caption=f"车牌 #{plate['id']} 裁剪小图")

            # 显示最终绘制好框的图片
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            image_placeholder.image(annotated_img_rgb, channels="RGB", use_container_width=True)
            ocr_placeholder.markdown(ocr_result_text, unsafe_allow_html=True)
        else:
            st.error(f"后端处理失败: {res_json.get('error')}")
    else:
        image_placeholder.info("💡 请在左侧面板上传一张包含车辆车牌的照片。")

# ----------------- 模式二：视频车牌检测（仅定位） -----------------
else:
    st.info("💡 提示：视频流通常适合前后端一体化或使用 WebSocket 传输。由于你采用了前后端分离架构，照片模式已成功解耦！若需视频支持，可将视频拆帧后重复调用照片接口。")