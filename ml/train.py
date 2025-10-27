"""
训练脚本
使用MLflow跟踪实验
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import mlflow.sklearn
import joblib
import yaml
import git
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from ml.data_pipeline import DataPipeline


def setup_mlflow_remote():
    """设置远程 MLflow 跟踪服务器"""
    # 从环境变量获取 MLflow 配置
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "classification_experiment")
    
    print(f"Setting up MLflow with tracking URI: {tracking_uri}")
    print(f"Experiment name: {experiment_name}")
    
    # 设置 MLflow 跟踪 URI
    mlflow.set_tracking_uri(tracking_uri)
    
    # 设置认证信息（如果提供）
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        print("MLflow authentication configured")
    
    # 检查 MLflow 服务器连接
    try:
        response = requests.get(f"{tracking_uri}/api/2.0/mlflow/experiments/list", timeout=10)
        if response.status_code == 200:
            print("✅ MLflow server connection successful")
        else:
            print(f"⚠️ MLflow server returned status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to connect to MLflow server: {e}")
        print("Continuing with training anyway...")
    
    return experiment_name


def get_git_info():
    """获取Git提交信息"""
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        branch = repo.active_branch.name
        return sha, branch
    except Exception as e:
        print(f"Could not get git info: {e}")
        return "unknown", "unknown"


def load_config():
    """加载训练配置"""
    config_path = "ml/configs/training_config.yaml"
    config = {
        "data": {"test_size": 0.2, "random_state": 42},
        "model": {"random_forest": {"n_estimators": 100, "max_depth": 10}},
        "training": {"random_state": 42},
    }

    if os.path.exists(config_path):
        try:
            # 尝试多种编码方式
            encodings = ["utf-8", "gbk", "latin-1"]
            for encoding in encodings:
                try:
                    with open(config_path, "r", encoding=encoding) as file:
                        loaded_config = yaml.safe_load(file)
                        if loaded_config:
                            config = loaded_config
                            msg = f"Successfully loaded config with {encoding} encoding"
                            print(msg)
                            break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using default configuration")
    else:
        print("Config file not found, using default configuration")
    
    return config


def train_model():
    """训练模型并记录实验到远程 MLflow 服务器"""
    
    # 设置远程 MLflow
    experiment_name = setup_mlflow_remote()
    
    # 加载配置
    config = load_config()

    # 设置 MLflow 实验
    mlflow.set_experiment(experiment_name)

    # 开始 MLflow 运行
    with mlflow.start_run():
        print("Starting model training...")

        # 记录 Git 提交信息
        git_sha, git_branch = get_git_info()
        mlflow.set_tag("git_commit", git_sha)
        mlflow.set_tag("git_branch", git_branch)
        mlflow.set_tag("code_version", git_sha)
        
        # 记录环境信息
        mlflow.set_tag("python_version", sys.version)
        mlflow.set_tag("platform", sys.platform)

        # 运行数据管道
        pipeline = DataPipeline()
        X_train, X_test, y_train, y_test = pipeline.run_pipeline()

        # 记录数据集信息
        mlflow.log_param("dataset_size", len(X_train))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("git_commit", git_sha)
        mlflow.log_param("data_source", "synthetic_generated")
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # 记录超参数
        params = config["model"]["random_forest"]
        for param, value in params.items():
            mlflow.log_param(param, value)
        
        # 记录训练配置
        mlflow.log_param("test_size", config["data"]["test_size"])
        mlflow.log_param("random_state", config["training"]["random_state"])

        # 创建并训练模型
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=config["training"]["random_state"],
        )

        print("Training model...")
        model.fit(X_train, y_train)

        # 进行预测
        y_pred = model.predict(X_test)

        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # 记录指标
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # 记录训练完成状态
        mlflow.set_tag("training_status", "completed")

        print(f"Model trained - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

        # 保存模型到本地（备份）
        model_path = "ml/registry/model.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        
        # 记录模型文件路径
        mlflow.log_artifact(model_path, "model")

        # 记录模型到 MLflow（远程）
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="random_forest_classifier"
        )

        print("Training completed and logged to MLflow")
        
        # 返回训练结果
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "model_path": model_path,
            "run_id": mlflow.active_run().info.run_id
        }


if __name__ == "__main__":
    try:
        result = train_model()
        print(f"Training successful! Run ID: {result['run_id']}")
        print(f"Accuracy: {result['accuracy']:.4f}, F1 Score: {result['f1_score']:.4f}")
    except Exception as e:
        print(f"Training failed: {e}")
        # 记录失败状态到 MLflow
        try:
            mlflow.set_tag("training_status", "failed")
            mlflow.log_param("error", str(e))
        except:
            pass
        sys.exit(1)