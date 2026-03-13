# 作业评语单链路（single_v2）说明

## 1. 运行链路

当前评语链路固定为：

1. 诊断层：规则定档（invalid / off_topic / partial / good）
2. 生成层：本地SFT模型生成评语草稿（可回退外部LLM）
3. 约束层：清洗评分/Markdown/越界建议，输出稳定格式

说明：
- 不再要求所有回答都必须三段式。
- 对 invalid/off_topic 默认只输出：`问题 + 改进建议`。
- 题干未要求时，禁止把“复杂度/应用场景/比较分析”当作必改项。

## 2. 配置项（.env）

```dotenv
ASSIGNMENT_FEEDBACK_PIPELINE_VERSION=single_v2
ENABLE_ASSIGNMENT_FEEDBACK_SFT_MODEL=false
ASSIGNMENT_FEEDBACK_SFT_MODEL_PATH=
ASSIGNMENT_FEEDBACK_EXTERNAL_FALLBACK=true
ASSIGNMENT_FEEDBACK_SFT_MAX_NEW_TOKENS=220
ASSIGNMENT_FEEDBACK_SFT_TEMPERATURE=0.2
ASSIGNMENT_FEEDBACK_SFT_TOP_P=0.9
```

建议首版：
- 本地模型未就绪时，保持 `ENABLE_ASSIGNMENT_FEEDBACK_SFT_MODEL=false`，并打开外部兜底。
- 本地模型训练完成后，设置 `ENABLE_ASSIGNMENT_FEEDBACK_SFT_MODEL=true` 并填入模型路径。

## 3. 训练数据构建（SFT）

```bash
python training/build_assignment_feedback_sft_mix.py \
  --out-dir training/data/assignment_feedback_sft_mix \
  --include-scientsbank \
  --include-beetle \
  --from-feedback-mix training/data/assignment_feedback_mix
```

如有本地教师样本，可叠加：

```bash
python training/build_assignment_feedback_sft_mix.py \
  --out-dir training/data/assignment_feedback_sft_mix \
  --include-scientsbank \
  --include-beetle \
  --from-feedback-mix training/data/assignment_feedback_mix \
  --local-train training/data/assignment_feedback_local/train.jsonl \
  --local-validation training/data/assignment_feedback_local/validation.jsonl
```

## 4. SFT训练（Qwen2.5-7B-Instruct + QLoRA）

训练任务类型：`assignment_feedback_sft_hf`

可通过训练API触发，或在脚本中直接调用 `training/train_assignment_feedback_sft_hf.py`。

依赖（AutoDL）：

```bash
pip install peft bitsandbytes
```

## 5. 离线对比评估

使用脚本：`backend/scripts/eval_assignment_feedback_shadow.py`

用途：
- 对比“直接LLM评语”与“single_v2链路评语”
- 输出误判率、越界建议率、时延和样例

