from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os

app = FastAPI(title="车牌检测识别后台服务")

# ==================== 1. 模型加载 ====================
MODEL_PATH = r"E:\AI_Curriculum_Design\yolo\yolo\runs\detect\train-3\weights\best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 未找到 YOLO 模型文件，请检查路径：{MODEL_PATH}")

# 初始化模型（启动时加载一次，常驻内存）
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

# ==================== 2. 核心 API 接口 ====================
@app.post("/predict_license_plate")
async def predict_license_plate(
    file: UploadFile = File(...), 
    conf_threshold: float = Form(0.25)
):
    try:
        # 1. 读取前端上传的图片字节流，并转为 OpenCV 格式
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(status_code=400, content={"error": "图片解码失败"})

        # 2. YOLO 目标检测
        results = model.predict(source=img, conf=conf_threshold, device='cpu', verbose=False)
        boxes = results[0].boxes
        
        # 准备返回给前端的解析数据列表
        plates_data = []

        # 3. 如果检测到了车牌，进行抠图与 OCR 识别
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                
                # 利用 OpenCV 切片抠图
                plate_crop = img[y1:y2, x1:x2]
                
                plate_text = "未能识别出清晰文字"
                if plate_crop.size > 0:
                    # EasyOCR 识别
                    ocr_out = reader.readtext(plate_crop)
                    raw_text = "".join([res[1] for res in ocr_out]).strip()
                    if raw_text:
                        plate_text = raw_text.replace(" ", "").replace(".", "").upper()

                # 将每个车牌的信息（包括裁剪后图片的 base64 编码，方便传输）组织起来
                # 为了省去复杂的图片传输，我们这里直接把车牌的坐标和文字回传，前端自己来抠图展示
                plates_data.append({
                    "id": i + 1,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2],
                    "text": plate_text
                })

        return {
            "success": True,
            "plate_count": len(boxes),
            "plates": plates_data
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)