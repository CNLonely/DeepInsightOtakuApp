# DeepInsight Otaku App - 动漫角色识别工具

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.1.1-green.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个基于深度学习的动漫角色识别Web应用，支持人脸检测、特征提取、角色识别等功能，并提供B站机器人自动回复服务。

## 🌟 功能特性

### 核心功能
- **🎯 动漫角色识别**: 基于ResNet50+ArcFace的高精度角色识别
- **👥 多人脸检测**: 支持单张图片中多个角色的同时识别
- **📊 置信度评估**: 智能识别结果置信度评估和修正
- **🖼️ 图片上传**: 支持多种格式的图片上传和批量处理
- **🎨 图库管理**: 完整的角色图库浏览和管理系统

### 技术特性
- **🔧 双后端支持**: 支持PyTorch和ONNX两种推理后端
- **⚡ 高性能**: 基于FAISS的快速特征检索
- **🎭 YOLO检测**: 集成YOLO模型进行人脸检测
- **📈 实时统计**: 识别结果统计和可视化
- **🔐 用户认证**: 完整的用户登录和权限管理

### B站机器人功能
- **🤖 自动回复**: 支持B站评论自动识别和回复
- **📝 模板系统**: 可自定义的回复模板
- **⏰ 定时轮询**: 自动检测新评论
- **🎨 表情装饰**: 智能添加表情符号
- **🔒 隐私保护**: 支持结果混淆处理

## 📋 系统要求

- **Python**: 3.11+
- **CUDA**: 11.8+ (可选，用于GPU加速)
- **内存**: 8GB+ RAM
- **存储**: 10GB+ 可用空间
- **操作系统**: Windows/Linux/macOS

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/DeepInsightOtakuApp.git
cd DeepInsightOtakuApp
```

### 2. 安装依赖

#### 使用pip安装
```bash
pip install -r requirements.txt
```

#### 使用conda安装（推荐）
```bash
conda create -n deepinsight python=3.11
conda activate deepinsight
pip install -r requirements.txt
```

### 3. 配置模型文件

将预训练模型文件放置在 `models/` 目录下：

```
models/
├── pytorch/
│   ├── recognition_model.pth    # 识别模型
│   └── yolo_model.pt           # YOLO检测模型
├── onnx/
│   ├── recognition_model.onnx  # ONNX识别模型
│   └── yolo_model.onnx        # ONNX YOLO模型
├── feature_db.npy              # 特征数据库
├── class_to_idx.json           # 类别映射
└── character_meta_restructured.json  # 角色元数据
```

### 4. 启动应用

```bash
python app.py
```

应用将在 `http://localhost:8000` 启动。

## 📁 项目结构

```
DeepInsightOtakuApp/
├── app/                    # 应用核心代码
│   ├── controllers/        # 控制器层
│   ├── services/          # 服务层
│   ├── core/              # 核心模块
│   ├── utils/             # 工具函数
│   └── config/            # 配置管理
├── config/                # 配置文件
├── data/                  # 训练和测试数据
├── models/                # 预训练模型
├── templates/             # HTML模板
├── static/                # 静态资源
├── logs/                  # 日志文件
├── app.py                 # 应用入口
├── requirements.txt       # Python依赖
└── Dockerfile            # Docker配置
```

## 🔧 配置说明

### 主配置文件 (`config/config.json`)

```json
{
  "project_name": "动漫角色识别工具",
  "recognition_threshold": 0.5,
  "max_faces": 20,
  "recognition_backend": "pytorch",
  "glass_opacity": 0.50,
  "background": {
    "type": "image",
    "image": "static/backgrounds/default.png"
  }
}
```

### B站机器人配置 (`config/bilibot_config.json`)

```json
{
  "enabled": false,
  "polling_interval_base": 15,
  "trigger_keyword": "识别动漫",
  "confidence_threshold": 0.55,
  "reply_templates": [
    "您要找的是不是：\n【《{video_title}》】\n{timeline}"
  ]
}
```

## 🐳 Docker部署

### 构建镜像

```bash
docker build -t deepinsight-otaku .
```

### 运行容器

```bash
docker run -d \
  --name deepinsight-app \
  -p 8000:8000 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  deepinsight-otaku
```

## 📖 使用指南

### Web界面使用

1. **访问应用**: 打开浏览器访问 `http://localhost:8000`
2. **用户登录**: 使用管理员账户登录系统
3. **上传图片**: 在识别页面上传动漫角色图片
4. **查看结果**: 系统将显示识别结果和置信度
5. **管理图库**: 在图库页面浏览和管理角色图片

### B站机器人使用

1. **启用机器人**: 在 B站机器人页 设置
2. **设置关键词**: 配置触发关键词
3. **自定义回复**: 编辑回复模板

### API接口

#### 识别接口
```http
POST /recognize
Content-Type: multipart/form-data

file: [图片文件]
```

#### 图库接口
```http
GET /gallery?page=1&per_page=20
```

#### 统计接口
```http
GET /statistics
```

## 🔍 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认模型文件完整性
   - 检查CUDA版本兼容性

2. **内存不足**
   - 减少 `max_faces` 配置值
   - 使用CPU模式运行
   - 增加系统内存

3. **识别精度低**
   - 调整 `recognition_threshold` 阈值
   - 检查训练数据质量
   - 更新特征数据库

4. **B站机器人无响应**
   - 检查Cookie是否过期
   - 确认网络连接正常
   - 查看日志文件错误信息

### 日志查看

```bash
# 查看应用日志
tail -f logs/app.log

# 查看历史日志
ls logs/history/
```

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [ResNet](https://arxiv.org/abs/1512.03385) - 残差网络架构
- [ArcFace](https://arxiv.org/abs/1801.07698) - 人脸识别损失函数
- [Flask](https://flask.palletsprojects.com/) - Web框架
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO实现
- [FAISS](https://github.com/facebookresearch/faiss) - 向量检索
- [Bilibili API Python](https://github.com/Nemo2011/bilibili-api) - B站API

---

⭐ 如果这个项目对你有帮助，请给它一个星标！ 
