"""
数据预处理管道
处理原始数据并准备训练数据
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import os


class DataPipeline:
    def __init__(self, config_path="ml/configs/training_config.yaml"):
        """初始化数据管道"""
        self.config = self.load_config(config_path)

    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def load_data(self, data_path):
        """
        加载数据

        Args:
            data_path: 数据文件路径

        Returns:
            加载的数据
        """
        # 这里使用示例数据，实际项目中从文件加载
        # 创建一个简单的分类数据集
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=1000,
            n_features=4,
            n_redundant=0,
            n_informative=4,
            n_clusters_per_class=1,
            random_state=42,
        )

        return X, y

    def preprocess_data(self, X, y):
        """
        预处理数据

        Args:
            X: 特征数据
            y: 标签数据

        Returns:
            预处理后的训练和测试数据
        """
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"],
        )

        return X_train, X_test, y_train, y_test

    def run_pipeline(self, data_path):
        """
        运行完整的数据管道

        Args:
            data_path: 数据文件路径

        Returns:
            处理后的数据
        """
        print("Loading data...")
        X, y = self.load_data(data_path)

        print("Preprocessing data...")
        X_train, X_test, y_train, y_test = self.preprocess_data(X, y)

        print(f"Data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")

        return X_train, X_test, y_train, y_test
