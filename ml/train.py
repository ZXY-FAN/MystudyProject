"""
训练脚本
使用MLflow跟踪实验
"""

import mlflow
import mlflow.sklearn
import joblib
import os
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from ml.data_pipeline import DataPipeline
import git


def get_git_info():
    """获取Git提交信息"""
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha
    except:
        return "unknown"


def train_model():
    """训练模型并记录实验"""

    # 加载配置
    with open("ml/configs/training_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # 设置MLflow实验
    mlflow.set_experiment("classification_experiment")

    # 开始MLflow运行
    with mlflow.start_run():
        print("Starting model training...")

        # 记录Git提交信息
        git_sha = get_git_info()
        mlflow.set_tag("git_commit", git_sha)

        # 运行数据管道
        pipeline = DataPipeline()
        X_train, X_test, y_train, y_test = pipeline.run_pipeline("data/raw_data.csv")

        # 记录数据集信息
        mlflow.log_param("dataset_size", len(X_train))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("git_commit", git_sha)

        # 记录超参数
        params = config["model"]["random_forest"]
        for param, value in params.items():
            mlflow.log_param(param, value)

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

        print(f"Model trained - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

        # 保存模型
        model_path = "ml/registry/model.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

        # 记录模型
        mlflow.sklearn.log_model(model, "model")

        # 记录混淆矩阵（示例）
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        # 保存并记录图表
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        print("Training completed and logged to MLflow")


if __name__ == "__main__":
    train_model()
