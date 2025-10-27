"""
主应用入口点
实现一个简单的机器学习API服务
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from app.models.model_handler import ModelHandler

# 初始化Flask应用
app = Flask(__name__)

# 全局模型处理器
model_handler = ModelHandler()


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查端点"""
    return jsonify({"status": "healthy", "message": "ML Service is running"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    预测端点
    接收JSON数据并返回模型预测结果
    """
    try:
        # 获取请求数据
        data = request.get_json()

        # 验证输入数据
        if not data or "features" not in data:
            return jsonify({"error": "Missing features in request"}), 400

        features = data["features"]

        # 进行预测
        prediction = model_handler.predict(features)

        return jsonify({"prediction": prediction.tolist(), "status": "success"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model_info", methods=["GET"])
def model_info():
    """返回当前模型信息"""
    return jsonify(model_handler.get_model_info())


if __name__ == "__main__":
    # 启动应用
    app.run(host="0.0.0.0", port=5000, debug=False)
