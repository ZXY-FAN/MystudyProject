# 使用Python官方镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app/ ./app/
COPY ml/ ./ml/

# 创建必要的目录
RUN mkdir -p ml/registry

# 暴露端口
EXPOSE 5000

# 设置环境变量
ENV FLASK_APP=app/main.py
ENV FLASK_ENV=production

# 启动命令
CMD ["python", "app/main.py"]