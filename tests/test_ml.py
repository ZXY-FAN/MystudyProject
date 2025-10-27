"""
机器学习管道测试
"""

import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ml.data_pipeline import DataPipeline
import numpy as np


def test_data_pipeline_initialization():
    """测试数据管道初始化"""
    pipeline = DataPipeline()
    assert pipeline is not None
    assert pipeline.config is not None


def test_data_pipeline_config_loading():
    """测试配置加载"""
    pipeline = DataPipeline()
    config = pipeline.config

    assert "data" in config
    assert "model" in config
    assert "training" in config
    assert "test_size" in config["data"]


def test_data_loading():
    """测试数据加载"""
    pipeline = DataPipeline()
    X, y = pipeline.load_data("dummy_path")

    assert X is not None
    assert y is not None
    assert len(X) == len(y)


def test_data_preprocessing():
    """测试数据预处理"""
    pipeline = DataPipeline()

    # 创建测试数据
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, 100)

    X_train, X_test, y_train, y_test = pipeline.preprocess_data(X, y)

    # 检查数据分割是否正确
    expected_test_size = int(len(X) * pipeline.config["data"]["test_size"])
    assert len(X_test) == expected_test_size
    assert len(X_train) == len(X) - expected_test_size
    assert len(y_train) == len(y) - expected_test_size


def test_pipeline_integration():
    """测试完整管道集成"""
    pipeline = DataPipeline()
    X_train, X_test, y_train, y_test = pipeline.run_pipeline("dummy_path")

    # 检查返回的数据形状
    assert X_train.shape[1] == X_test.shape[1]  # 特征数量相同
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
