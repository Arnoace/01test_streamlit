import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import numpy as np
import easyocr

# ==================== 1. 网页页面基本配置 ====================
st.set_page_config(page_title="AI 车牌检测与识别系统", layout="wide")
st.title("🚗 智能车牌检测与文本识别系统")
st.markdown("欢迎使用车牌系统。请在左侧选择模式并上传文件，系统将实时进行车牌定位与文字识别。")

# ==================== 2. 侧边栏配置与模型加载 ====================
st.sidebar.header("⚙️ 配置面板")

# 功能模式选择
app_mode = st.sidebar.selectbox("选择功能模式", ["视频车牌检测（仅定位）", "照片车牌识别（文字识别）"])

# 模型路径配置
model_path = r"E:\AI_Curriculum_Design\yolo\yolo\runs\detect\train-3\weights\best.pt"

conf_threshold = st.sidebar.slider("置信度阈值 (Confidence)", 0.0, 1.0, 0.25, 0.05)

# 检查目标检测模型是否存在
if not os.path.exists(model_path):
    st.sidebar.error(f"❌ 未找到 YOLO 模型文件，请检查路径：{model_path}")
    st.stop()

# 缓存 YOLO 模型加载
@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

# 缓存 EasyOCR 加载（指定中英文识别：ch_sim 代表简体中文，en 代表英文）
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['ch_sim', 'en'], gpu=False)  # 默认用CPU，有显卡可设为 gpu=True

model = load_yolo_model(model_path)
reader = load_ocr_reader()
st.sidebar.success("🚀 YOLO模型 & OCR引擎 加载成功！")

# ==================== 3. 核心业务逻辑 ====================

# ----------------- 模式一：视频车牌检测 -----------------
if app_mode == "视频车牌检测（仅定位）":
    uploaded_video = st.sidebar.file_uploader("上传视频 (MP4格式)", type=["mp4", "avi", "mov"])
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📺 实时检测画面")
        video_placeholder = st.empty()
    with col2:
        st.subheader("📊 实时数据提取")
        data_placeholder = st.empty()

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        st.sidebar.info("正在处理视频流...")
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            
            results = model.predict(source=frame, conf=conf_threshold, device='cpu', verbose=False)
            annotated_frame = results[0].plot()
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
            boxes = results[0].boxes
            info_text = f"**当前帧数**: 第 {frame_idx} 帧\n\n"
            if len(boxes) > 0:
                info_text += f"🟢 **检测状态**: 发现 {len(boxes)} 个车牌！\n\n---\n"
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    info_text += f"**【车牌 {i+1}】**\n- 置信度: `{conf:.2f}`\n- 位置: `({int(x1)}, {int(y1)})` 到 `({int(x2)}, {int(y2)})`\n\n"
            else:
                info_text += "⚪ **检测状态**: 未发现车牌\n"
            data_placeholder.markdown(info_text)
        cap.release()
        st.sidebar.success("🎉 整个视频检测完毕！")
    else:
        video_placeholder.info("💡 请在左侧面板上传视频文件以启动检测系统。")

# ----------------- 模式二：照片车牌识别（新功能） -----------------
else:
    uploaded_image = st.sidebar.file_uploader("上传照片 (JPG/PNG格式)", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📸 原始与检测画面")
        image_placeholder = st.empty()
    with col2:
        st.subheader("🔤 车牌文字识别结果")
        ocr_placeholder = st.empty()

    if uploaded_image is not None:
        # 将上传的文件转为 OpenCV 图像格式
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # 1. 使用 YOLO 定位车牌
        results = model.predict(source=img, conf=conf_threshold, device='cpu', verbose=False)
        boxes = results[0].boxes
        
        # 渲染画了框的图片
        annotated_img = results[0].plot()
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        image_placeholder.image(annotated_img_rgb, channels="RGB", use_container_width=True)
        
        # 2. 开始进行 OCR 文字抠图与识别
        ocr_result_text = f"🟢 **检测结果**: 画面中共发现 {len(boxes)} 个车牌。\n\n---\n"
        
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                # 拿到车牌的整数坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # 【关键核心】：利用 OpenCV 的切片功能，直接把车牌图片从原图中“抠”出来
                plate_crop = img[y1:y2, x1:x2]
                
                if plate_crop.size > 0:
                    # 使用 EasyOCR 对抠出来的车牌小图进行文字识别
                    ocr_out = reader.readtext(plate_crop)
                    
                    # 拼接识别到的所有文字（防止车牌字母和汉字被识别成多段）
                    plate_text = "".join([res[1] for res in ocr_out]).strip()
                    # 过滤掉一些奇怪的标点
                    plate_text = plate_text.replace(" ", "").replace(".", "").upper()
                    
                    if not plate_text:
                        plate_text = "未能识别出清晰文字"
                    
                    # 将车牌小图转为 RGB 格式用于在网页上单独展示
                    plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                    
                    # 动态拼装前端 markdown 文本
                    ocr_result_text += f"### 🆔 车牌 #{i+1}\n"
                    ocr_result_text += f"- **置信度**: `{box.conf[0].item():.2f}`\n"
                    ocr_result_text += f"- **识别文本**: <span style='color:red; font-size:24px; font-weight:bold;'>{plate_text}</span>\n\n"
                    
                    # 顺便在右侧把抠出来的车牌局部小图展现出来，显得很专业
                    st.image(plate_crop_rgb, caption=f"车牌 #{i+1} 裁剪区域")
        else:
            ocr_result_text += "⚪ 未能在照片中定位到车牌，请调整左侧置信度阈值或更换更清晰的照片。"
            
        ocr_placeholder.markdown(ocr_result_text, unsafe_allow_html=True)
    else:
        image_placeholder.info("💡 请在左侧面板上传一张包含车辆车牌的照片。")