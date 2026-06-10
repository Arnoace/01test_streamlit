# 车牌识别系统 🚗

> 这是一个用于学习实践的计算机视觉小项目，通过 YOLO 算法实现车牌检测与识别。

## 📋 项目简介

本项目是一个完整的车牌识别解决方案，从图像输入到车牌号输出，涵盖了计算机视觉和现代 Web 开发的全流程。

### 核心功能
- 🎯 **车牌检测**：基于 YOLO 算法的高精度车牌定位
- 🔤 **字符识别**：自动识别车牌上的字符（支持中文、英文、数字）
- 🌐 **Web 界面**：Streamlit 构建的友好交互界面
- 🚀 **API 服务**：FastAPI 提供高性能后端服务

## 🛠️ 技术栈

| 技术 | 用途 |
|------|------|
| **YOLO** | 车牌目标检测 |
| **FastAPI** | 后端 API 框架 |
| **Streamlit** | 前端交互界面 |
| **OpenCV** | 图像预处理 |
| **PyTorch** | 深度学习框架 |

## 📸 效果展示


### 车牌检测效果

### Web 界面
<img width="1892" height="1772" alt="image" src="https://github.com/user-attachments/assets/b733b3c9-6457-4801-8803-11a25a16d173" />


## 🚀 快速开始
打开两个终端分别运行:
 uvicorn backend:app --reload
 streamlit run frontend.py 

### 环境要求
- Python 3.12
- pip 包管理器

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/Arnoace/01test_streamlit.git
cd 01test_streamlit
