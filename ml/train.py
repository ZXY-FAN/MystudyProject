"""
训练脚本
使用MLflow跟踪实验
"""
import sys
import os
import mlflow
import mlflow.sklearn
import joblib
import yaml
import git
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# 添加项目根目录到 Python 路径（在所有导入之后）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.data_pipeline import DataPipeline

def get_git_info():
    """获取Git提交信息"""
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha
    except Exception:  # 修复：使用具体的异常类型而不是裸 except
        return "unknown"

def train_model():
    """训练模型并记录实验"""
    
    # 加载配置 - 使用安全的文件读取方式
    config_path = 'ml/configs/training_config.yaml'
    config = {
        'data': {'test_size': 0.2, 'random_state': 42},
        'model': {'random_forest': {'n_estimators': 100, 'max_depth': 10}},
        'training': {'random_state': 42}
    }
    
    if os.path.exists(config_path):
        try:
            # 尝试多种编码方式
            encodings = ['utf-8', 'gbk', 'latin-1']
            for encoding in encodings:
                try:
                    with open(config_path, 'r', encoding=encoding) as file:
                        loaded_config = yaml.safe_load(file)
                        if loaded_config:
                            config = loaded_config
                            print(f"Successfully loaded config with {encoding} encoding")
                            break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using default configuration")
    else:
        print("Config file not found, using default configuration")
    
    # 设置MLflow实验
    mlflow.set_experiment("classification_experiment")
    
    # 开始MLflow运行
    with mlflow.start_run():
        print("Starting model training...")
        
        # 记录Git提交信息
        git_sha = get_git_info()
        mlflow.set_tag("git_commit", git_sha)
        
        # 运行数据管道（不再依赖外部数据文件）
        pipeline = DataPipeline()
        X_train, X_test, y_train, y_test = pipeline.run_pipeline()
        
        # 记录数据集信息
        mlflow.log_param("dataset_size", len(X_train))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("git_commit", git_sha)
        mlflow.log_param("data_source", "synthetic_generated")
        
        # 记录超参数
        params = config['model']['random_forest']
        for param, value in params.items():
            mlflow.log_param(param, value)
        
        # 创建并训练模型
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=config['training']['random_state']
        )
        
        print("Training model...")
        model.fit(X_train, y_train)
        
        # 进行预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 记录指标
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        print(f"Model trained - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        
        # 保存模型
        model_path = "ml/registry/model.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        
        # 记录模型
        mlflow.sklearn.log_model(model, "model")
        
        print("Training completed and logged to MLflow")

if __name__ == '__main__':
    train_model()