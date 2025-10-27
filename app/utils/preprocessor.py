"""
数据预处理工具
"""

from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self):
        """初始化预处理器"""
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, data):
        """拟合预处理器"""
        self.scaler.fit(data)
        self.is_fitted = True

    def transform(self, data):
        """转换数据"""
        if not self.is_fitted:
            raise Exception("Preprocessor not fitted yet")
        return self.scaler.transform(data)

    def fit_transform(self, data):
        """拟合并转换数据"""
        return self.scaler.fit_transform(data)
