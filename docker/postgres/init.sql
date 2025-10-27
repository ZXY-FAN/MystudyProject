-- 为 MLflow 创建数据库（如果还不存在）
SELECT 'CREATE DATABASE mlflowdb'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflowdb')\gexec