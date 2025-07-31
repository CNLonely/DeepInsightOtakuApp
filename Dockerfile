FROM python:3.11.4

WORKDIR /app

# 安装 OpenCV 所需的系统依赖和中文字体
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx fonts-noto-cjk --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt 并安装 Python 依赖
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

COPY config/ ./config/

# 复制 meta 目录（元数据）
COPY meta/ ./meta/

# 复制 models 目录（所有模型）
COPY models/ ./models/

# 复制所有模板
COPY templates/ ./templates/

# 复制所有静态资源
COPY static/ ./static/

COPY app.py .

# 暴露端口
EXPOSE 8000

# 容器启动命令
CMD ["python", "app.py"] 