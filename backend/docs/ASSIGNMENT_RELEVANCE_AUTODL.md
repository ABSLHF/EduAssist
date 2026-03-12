# 作业判题模型（AutoDL 单卡 5090）使用说明

## 1. 目标
本文档用于快速跑通“作业相关性模型”的训练与上线流程：
1. 组合数据构建（公开数据 + 本地标注）。
2. 触发 `assignment_relevance_hf` 两阶段训练（A->B）。
3. 启用线上判题融合（规则 + 模型 + LLM 二判）。

## 2. 数据构建（手动方式）
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

输出文件：
- `training/data/assignment_relevance_mix/train.jsonl`
- `training/data/assignment_relevance_mix/validation.jsonl`
- `training/data/assignment_relevance_mix/manifest.json`

## 3. 一键脚本（推荐）
已提供脚本：`scripts/run_assignment_relevance_autodl.py`。

脚本会自动执行：
1. 构建混合数据集（可跳过）。
2. 教师登录获取 token。
3. 调用 `/model/train` 触发训练。
4. 轮询 `/model/train/{job_id}` 直到成功/失败。
5. 查询 `/model/active` 输出当前生效模型。

示例：

```bash
python scripts/run_assignment_relevance_autodl.py \
  --base-url http://127.0.0.1:8000 \
  --username teacher1 \
  --password 123456 \
  --include-ocnli \
  --include-lcqmc \
  --local-train training/data/assignment_local/train.jsonl \
  --local-validation training/data/assignment_local/validation.jsonl \
  --dataset-config training/data/assignment_relevance_mix \
  --dataset-name assignment_relevance_mix_local \
  --model-name hfl/chinese-roberta-wwm-ext \
  --epochs 1 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --max-samples 24000
```

如果你已经提前构建好数据，可加 `--skip-build`。

## 4. 训练接口（手动方式）
如果不用一键脚本，可直接调用接口：

`POST /model/train`

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
- 当 `epochs=1` 时，后端会按默认两阶段参数执行（A=3, B=2）。
- 训练产物输出到 `models/assignment_rel_stage_*`。

## 5. 启用线上判题融合
在 `.env` 设置：

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
- `ASSIGNMENT_RELEVANCE_MODEL_PATH` 为空时，会优先读取最新成功 `assignment_relevance_hf` 模型。
- 判题流程：模型判题 + 规则兜底 + ambiguous 走 LLM 二判。

## 6. 验收建议
至少验证以下样例：
1. 题目“什么是进程”，答案为进程定义 -> `relevant`。
2. 同题答案为线程定义 -> `off_topic`。
3. 标题“作业1”但描述明确，答案正确 -> 不应误判偏题。
