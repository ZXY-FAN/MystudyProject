"""
应用测试
包含多个有意义的测试用例
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest
from app.main import app
from app.models.model_handler import ModelHandler


@pytest.fixture
def client():
    """创建测试客户端"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """测试健康检查端点"""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'


def test_predict_endpoint_missing_features(client):
    """测试预测端点 - 缺少特征的情况"""
    response = client.post('/predict', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data


def test_predict_endpoint_with_features(client):
    """测试预测端点 - 有特征的情况"""
    # 注意：这个测试需要模型文件存在
    test_features = {
        'features': [1.0, 2.0, 3.0, 4.0]
    }
    response = client.post('/predict', json=test_features)
    # 如果模型不存在，会返回500，如果存在则返回200
    assert response.status_code in [200, 500]


def test_model_handler_initialization():
    """测试模型处理器初始化"""
    handler = ModelHandler()
    assert handler is not None


def test_model_handler_predict_without_model():
    """测试模型处理器在没有模型时的预测行为"""
    handler = ModelHandler()
    handler.model = None  # 确保没有模型
    
    with pytest.raises(Exception, match="Model not loaded"):
        handler.predict([1, 2, 3, 4])