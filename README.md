# ML DevOps 项目

一个集成了DevOps和MLOps实践的机器学习应用项目。

## 项目概述
- **应用**: 基于Flask的机器学习API服务
- **模型**: 随机森林分类器
- **功能**: 提供预测接口和模型管理

## 开发流程

### 代码提交 → 构建 → 测试 → 部署

1. **代码提交**: 开发者在feature分支开发
2. **Pull Request**: 创建PR到dev分支
3. **CI流水线**:
   - 代码检查 (ruff, black)
   - 运行测试 (pytest)
   - 构建Docker镜像
4. **代码审查**: 通过后合并到dev分支
5. **Staging部署**: 合并到staging分支触发staging部署
6. **生产部署**: 合并到main分支触发生产部署

## 分支策略

- `main`: 生产环境代码
- `staging`: 预生产环境代码
- `dev`: 开发集成分支
- `feature/*`: 功能开发分支

## 触发流水线

| 分支 | 触发动作 | 执行流水线 |
|------|----------|------------|
| feature/* | push/PR | 测试 + 构建 |
| dev | push/PR | 测试 + 构建 |
| staging | push/merge | 测试 + 构建 + Staging部署 |
| main | push/merge | 测试 + 构建 + 生产部署 |

## 快速开始

### 本地开发
```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/

# 启动应用
python app/main.py