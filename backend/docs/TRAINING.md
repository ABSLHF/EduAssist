# 训练模块使用说明

使用公开数据集做小模型微调，目标是可复现、可展示。

## 1. 依赖安装
在 `backend` 目录执行：

```powershell
pip install -r requirements.txt
```

新增关键依赖：
- `transformers`
- `torch`
- `datasets`
- `accelerate`

## 2. 训练任务类型
`POST /model/train` 支持以下 `task_type`：

1. `text_classification`  
   - 本地 CSV（`training/data/sample_labels.csv`）  
   - TF-IDF + LinearSVC（已有最小训练）

2. `text_classification_hf`  
   - HuggingFace 分类数据集微调  
   - 默认模型：`bert-base-chinese`

3. `qa_extractive_hf`  
   - HuggingFace 抽取式问答微调  
   - 默认数据集：`cmrc2018`  
   - 默认模型：`bert-base-chinese`

## 3. 训练请求示例

### 3.1 分类微调（HF）
```json
{
  "task_type": "text_classification_hf",
  "dataset_name": "clue",
  "dataset_config": "iflytek",
  "model_name": "bert-base-chinese",
  "epochs": 1,
  "batch_size": 8,
  "learning_rate": 2e-5,
  "max_samples": 800
}
```

### 3.2 抽取式问答微调（HF）
```json
{
  "task_type": "qa_extractive_hf",
  "dataset_name": "cmrc2018",
  "model_name": "bert-base-chinese",
  "epochs": 1,
  "batch_size": 4,
  "learning_rate": 3e-5,
  "max_samples": 400
}
```

## 4. 训练状态查询
- 接口：`GET /model/train/{job_id}`
- 成功时会看到：
  - `status = success`
  - `metrics`（JSON 字符串，包含 accuracy/f1 或 em/f1）
  - `model_path`（如 `models/cls_20260217_...`）

## 5. 预测接口

### 5.1 分类预测
- `POST /model/predict`
```json
{
  "text": "数组和链表的存储结构有什么区别",
  "model_path": null
}
```

### 5.2 抽取式问答预测
- `POST /model/qa_predict`
```json
{
  "question": "什么是数据结构",
  "context": "数据结构是计算机存储、组织数据的方式。",
  "model_path": null
}
```

返回：
```json
{
  "answer": "计算机存储、组织数据的方式",
  "confidence": 0.73
}
```

## 6. 与主问答接口联动（可选）
- 配置项：`.env` 中 `ENABLE_SMALL_QA_ASSIST=true`
- 含义：在 `POST /qa` 的 fallback 路径里，若检测到已训练 QA 小模型，会尝试用小模型从检索片段抽取一句答案。
- 若小模型不可用，会自动回退原 fallback，不影响主流程。

## 7. 常见失败与处理
1. 下载数据集失败（网络问题）  
   - 训练任务会 `failed`，`error_message` 有具体错误。
2. 数据集字段不匹配  
   - 会提示缺失 `label` 或问答字段。
3. 模型未训练就预测  
   - 接口返回 400，提示先训练。

## 8. 自动化全链路验收（在线服务）
- 脚本：`scripts/verify_e2e.py`
- 用途：自动校验认证、课程、资料、问答、作业、推荐、知识图谱、微调与预测链路。
- 运行：
  - `python scripts/verify_e2e.py --base-url http://127.0.0.1:8000`
- 报告输出：
  - `docs/verification_report.json`
  - `docs/verification_report.md`

## 9. 教学评测集（30题）
- 文件：`training/data/teaching_eval_30.csv`
- 字段：
  - `question`：问题
  - `gold_answer`：标准答案
  - `knowledge_point`：知识点
  - `difficulty`：难度（easy/medium/hard）
- 用途：
  - 固定题集做“微调前后对比”，避免只看单次演示结果。
  - 可手工记录命中情况，或后续扩展为自动评测脚本。

## 10. 当前建议参数（CPU可跑）
1. 分类微调（HF）
```json
{
  "task_type": "text_classification_hf",
  "dataset_name": "clue",
  "dataset_config": "iflytek",
  "model_name": "bert-base-chinese",
  "epochs": 1,
  "batch_size": 4,
  "learning_rate": 2e-5,
  "max_samples": 1000
}
```

2. 抽取式问答微调（HF）
```json
{
  "task_type": "qa_extractive_hf",
  "dataset_name": "cmrc2018",
  "model_name": "bert-base-chinese",
  "epochs": 1,
  "batch_size": 2,
  "learning_rate": 3e-5,
  "max_samples": 300
}
```

## 11. 标签可读性说明（分类）
- HF 分类模型训练后会在模型目录保存 `label_map.json`。
- `POST /model/predict` 会优先把 `LABEL_70` 或 `70` 映射为可读标签。
- 若映射缺失，返回兜底格式 `label_70`，避免前端出现纯数字标签。

## 12. 教学评测对比（30题自动化）
- 脚本：`scripts/eval_teaching_30.py`
- 输入：`training/data/teaching_eval_30.csv`
- 对比对象：
  - `POST /qa`（RAG问答）
  - `POST /model/qa_predict`（小模型抽取）
- 输出：
  - `docs/teaching_eval_report_*.json`
  - `docs/teaching_eval_report_*.md`

### 运行示例
```powershell
python scripts/eval_teaching_30.py --base-url http://127.0.0.1:8000 --course-id 10
```

### 常用参数
- `--teacher-username` / `--teacher-password`：教师账号（用于 `qa_predict`）。
- `--student-username` / `--student-password`：学生账号（用于 `/qa`）。
- `--qa-model-path`：指定 QA 模型目录（可选，不填默认取最新成功模型）。
- `--context-mode`：
  - `retrieve`（默认）：调用本地检索得到片段作为 `qa_predict` 上下文（最贴近真实问答流程）。
  - `global`：用整套题的问答对拼成上下文，适合快速联调，不建议用于最终指标。
  - `gold`：每题仅用该题标准答案做上下文，适合上限测试。

## 13. RAG 检索优化结果（2026-02-20）
- 目标：提升“教师上传资料后”的真实检索命中率（`context_mode=retrieve`）。
- 代码改动：
  - `app/rag/pipeline.py`：扩大召回候选、关键词与同义词重排、语义未命中时词法兜底。
  - `app/api/qa.py`：问题关键词抽取增强、标题噪声行过滤、fallback 仅返回最相关一句。
  - `app/api/materials.py`：上传成功后清理该课程 QA 缓存，避免旧答案污染评测。
- 结果对比（同一评测集 `training/data/teaching_eval_30.csv`）：
  - 基线：`10/30 (0.3333)`（2026-02-20 02:10:39）
  - 优化后：`17/30 (0.5667)`（2026-02-20 03:16:47）
  - 提升：`+0.2334`（+7 题）

### 复现实验命令
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
python scripts/eval_teaching_30.py --base-url http://127.0.0.1:8000 --course-id 17 --context-mode retrieve --hit-threshold 0.2
```
