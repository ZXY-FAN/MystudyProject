# 使用Python官方镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app/ ./app/
COPY ml/ ./ml/

# 创建必要的目录
RUN mkdir -p ml/registry logs

# 暴露端口
EXPOSE 5000

# 设置环境变量（默认值）
ENV FLASK_APP=app/main.py
ENV FLASK_ENV=production
ENV FLASK_DEBUG=0
ENV MODEL_PATH=ml/registry/model.pkl

# 设置非root用户（安全最佳实践）
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# 启动命令
CMD ["python", "app/main.py"]