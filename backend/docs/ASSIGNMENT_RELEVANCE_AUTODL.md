# 作业判题模型（AutoDL 单卡 5090）使用说明

## 1. 生成训练数据（多源组合）

在 `backend` 目录执行：

```bash
python training/build_assignment_relevance_mix.py \
  --out-dir training/data/assignment_relevance_mix \
  --include-ocnli \
  --include-lcqmc \
  --local-train training/data/assignment_local/train.jsonl \
  --local-validation training/data/assignment_local/validation.jsonl \
  --sas-train training/data/sas_bench/train.jsonl \
  --sas-validation training/data/sas_bench/validation.jsonl
```

产出：

- `training/data/assignment_relevance_mix/train.jsonl`
- `training/data/assignment_relevance_mix/validation.jsonl`
- `training/data/assignment_relevance_mix/manifest.json`

## 2. 启动作业判题模型训练（两阶段 A->B）

通过后端接口 `/model/train`，示例 payload：

```json
{
  "task_type": "assignment_relevance_hf",
  "dataset_name": "assignment_relevance_mix_local",
  "dataset_config": "training/data/assignment_relevance_mix",
  "model_name": "hfl/chinese-roberta-wwm-ext",
  "epochs": 1,
  "batch_size": 16,
  "learning_rate": 2e-5,
  "max_samples": 24000
}
```

说明：

- `epochs=1` 时，系统会按默认两阶段参数执行（A=3, B=2）。
- 训练产物写入 `models/assignment_rel_stage_*`。

## 3. 启用线上判题融合

在 `.env` 配置：

```dotenv
ENABLE_ASSIGNMENT_RELEVANCE_MODEL=true
ASSIGNMENT_RELEVANCE_MODEL_PATH=
ASSIGNMENT_RELEVANCE_THRESHOLD_HI=0.70
ASSIGNMENT_RELEVANCE_THRESHOLD_LO=0.25
ASSIGNMENT_RELEVANCE_USE_RERANKER=false
ASSIGNMENT_RELEVANCE_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
ASSIGNMENT_RELEVANCE_RERANKER_WEIGHT=0.25
```

说明：

- `ASSIGNMENT_RELEVANCE_MODEL_PATH` 为空时，优先读取最新 `ModelRun(task_type=assignment_relevance_hf)` 成功模型。
- 判题流程：模型判题 + 规则兜底 + ambiguous 走 LLM 二判。
