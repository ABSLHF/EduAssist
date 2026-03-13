# 作业批改与评语链路（阶段1 + 阶段2）

## 1. 阶段1（已接入）

当前新增了三种运行模式（环境变量）：

```dotenv
ASSIGNMENT_FEEDBACK_MODE=legacy   # legacy|shadow|v2
ASSIGNMENT_FEEDBACK_SHADOW_LOG_PATH=logs/assignment_feedback_shadow.jsonl
ENABLE_ASSIGNMENT_FEEDBACK_MODEL=false
ASSIGNMENT_FEEDBACK_MODEL_PATH=
```

- `legacy`：保持旧链路（先相关性判定，再生成评语）。
- `shadow`：线上返回旧评语，同时后台生成新版评语并写 JSONL 对比日志。
- `v2`：启用新版链路（证据检索 + 结构化批改 + 评语生成）。
- `ENABLE_ASSIGNMENT_FEEDBACK_MODEL` / `ASSIGNMENT_FEEDBACK_MODEL_PATH`：阶段2模型接入预留开关，当前默认关闭。

新版链路核心：
1. 证据检索：按 `course_id + 题干 + 学生答案` 检索课程资料片段（top_k=4）。
2. 结构化批改：输出 4 维中间结果（概念正确性、要点覆盖、术语准确性、表达清晰度）。
3. 评语生成：要求固定三段 `优点/问题/改进建议`，且禁止输出分数。
4. 失败兜底：LLM失败时返回规则模板评语，不中断提交。

## 2. 阶段2（数据与训练）

新增数据脚本：

```bash
python training/build_assignment_feedback_mix.py \
  --out-dir training/data/assignment_feedback_mix \
  --include-scientsbank \
  --include-beetle \
  --local-train training/data/assignment_feedback_local/train.jsonl \
  --local-validation training/data/assignment_feedback_local/validation.jsonl
```

输出统一字段：
- `question`
- `reference_answer`
- `student_answer`
- `rubric_labels`（含 `raw_label` 和 `relevance`）
- `teacher_feedback`
- `source`

后续可用于：
- 判定模型（多头分类/多标签）
- 评语生成模型（如 QLoRA）
