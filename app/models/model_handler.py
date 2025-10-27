"""
模型处理器
负责加载模型并进行预测
"""
import joblib
import os
import numpy as np

class ModelHandler:
    def __init__(self):
        """初始化模型处理器"""
        self.model = None
        self.model_path = os.getenv('MODEL_PATH', 'ml/registry/model.pkl')
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
            else:
                print(f"Model file not found at {self.model_path}")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def predict(self, features):
        """
        使用模型进行预测
        
        Args:
            features: 输入特征数组
            
        Returns:
            预测结果
        """
        if self.model is None:
            raise Exception("Model not loaded")
        
        # 转换输入为numpy数组
        features_array = np.array(features).reshape(1, -1)
        
        # 进行预测
        prediction = self.model.predict(features_array)
        
        return prediction[0]
    
    def get_model_info(self):
        """返回模型信息"""
        if self.model is None:
            return {'status': 'no_model_loaded'}
        
        return {
            'status': 'loaded',
            'model_type': type(self.model).__name__,
            'features_expected': getattr(self.model, 'n_features_in_', 'unknown')
        }