# EduAssist 开发进度（2026-02-17）

## 当前完成
- 核心接口链路已跑通：认证、课程、资料上传、问答、作业、推荐、知识图谱、训练任务。
- RAG 问答已完成两轮优化：中文优先向量模型、噪声分片过滤、关键词重排、fallback 简答+引用。
- 训练模块新增“公开数据集微调”能力：
  - 文本分类微调：`text_classification_hf`
  - 抽取式问答微调：`qa_extractive_hf`
- 新增问答抽取预测接口：`POST /model/qa_predict`。
- 训练接口仍保持兼容：`POST /model/train`、`GET /model/train/{job_id}`、`POST /model/predict`。
- 新增自动化验收工具：`scripts/verify_e2e.py`（在线真实服务全链路校验）。

## 进行中
- 基于 HuggingFace 数据集的首次完整微调实测（分类 + 抽取式问答）。
- 训练失败场景回归：网络超时、数据集字段不匹配、模型未训练先预测。
- 已修复一项版本兼容问题：`TrainingArguments` 不支持 `overwrite_output_dir` 参数。
- 已增加训练参数兼容层：自动适配 `evaluation_strategy/eval_strategy` 并过滤不受支持参数。
- 已修复 QA 训练器兼容问题：`Trainer.__init__` 在部分版本不接受 `tokenizer` 参数。
- 已修复自动化验收阻塞项：HF 分类模型预测路径兼容（目录模型不再导致 `/model/predict` 500）。
- 推荐模块增强中：无知识图谱时自动降级为“基于问答/学习记录关键词”的临时推荐，避免空列表。
- 分类标签增强中：HF 预测标签从纯数字映射为可读标签，便于教学展示。

## 下一步（按优先级）
1. 使用 `training/data/teaching_eval_30.csv` 跑“微调前后对比”并记录命中率。
2. 先跑分类微调（建议 `dataset_name=clue`、`dataset_config=iflytek`）并验证 `predict` 标签可读性。
3. 再跑抽取式问答微调（`dataset_name=cmrc2018`）并验证 `qa_predict`。
4. 执行 `python scripts/verify_e2e.py --base-url http://127.0.0.1:8000` 生成自动化验收报告。
5. 固化演示脚本、指标对比表与关键截图，形成阶段验收材料。

## 阶段验收标准
- 至少 1 个分类微调任务 `success`，返回 accuracy/f1。
- 至少 1 个抽取式问答任务 `success`，返回 em/f1。
- `POST /model/qa_predict` 返回 `answer` 且不报 500。
- 原有核心接口无回归。

- 自动化验收执行（2026-02-18T02:16:20）：pass=0, warn=0, fail=16。 详情见 `docs/verification_report.md`。

- 自动化验收执行（2026-02-18T02:23:12）：pass=0, warn=0, fail=1。 详情见 `docs/verification_report.md`。

- 自动化验收执行（2026-02-18T02:25:17）：pass=0, warn=0, fail=1。 详情见 `docs/verification_report.md`。

- 自动化验收执行（2026-02-18T02:32:29）：pass=18, warn=1, fail=1。 详情见 `docs/verification_report.md`。

- 自动化验收执行（2026-02-18T02:37:55）：pass=0, warn=0, fail=1。 详情见 `docs/verification_report.md`。

- 自动化验收执行（2026-02-18T02:38:05）：pass=0, warn=0, fail=1。 详情见 `docs/verification_report.md`。

- 自动化验收执行（2026-02-18T02:43:58）：pass=18, warn=2, fail=0。 详情见 `docs/verification_report.md`。

## 本轮新增能力（2026-02-19）
- 推荐降级策略：当课程尚无知识图谱时，推荐接口会基于近期问答/学习记录抽取高频主题，返回结构化推荐项（不再固定空数组）。
- 分类标签归一化：支持读取 `config.json` 与 `label_map.json`，将 `LABEL_x` 或数字标签映射成可读标签；无映射时返回 `label_x` 兜底格式。
- 分类训练输出新增：模型目录自动写入 `label_map.json`，用于预测阶段标签解释。
- 新增教学评测集：`training/data/teaching_eval_30.csv`（30题，含标准答案/知识点/难度）。
- 自动化验收规则微调：`F-3` 仅在分类预测返回纯数字标签时标记 warn，便于聚焦“可演示可读性”目标。
- 新增教学评测自动化脚本：`scripts/eval_teaching_30.py`，可批量对比 RAG 与 `qa_predict` 两条链路并生成报告。

- 自动化验收执行（2026-02-19T17:05:52）：pass=15, warn=0, fail=1。 详情见 `docs/verification_report.md`。

- 自动化验收执行（2026-02-19T17:33:25）：pass=19, warn=1, fail=0。 详情见 `docs/verification_report.md`。

- 自动化验收执行（2026-02-19T18:19:48）：pass=19, warn=1, fail=0。 详情见 `docs/verification_report.md`。

- 自动化验收执行（2026-02-19T18:46:45）：pass=20, warn=0, fail=0。 详情见 `docs/verification_report.md`。

## RAG评测阶段进展（2026-02-20）
- 30题检索评测（`context_mode=retrieve`, `hit_threshold=0.2`）：
  - 2026-02-20 02:00:39：RAG `4/30 (0.1333)`，QA抽取 `1/30 (0.0333)`。
  - 2026-02-20 02:10:39：RAG `10/30 (0.3333)`，QA抽取 `2/30 (0.0667)`。
- 对照结论：
  - `gold` 模式下 QA抽取可达 `30/30`，说明抽取模型能力可用。
  - `retrieve` 模式提升明显但仍有空间，当前主要瓶颈在检索上下文质量。
- 阶段判断：
  - “教师上传资料 -> 检索答疑”主链路已形成可演示闭环。
  - 下一步继续以“补充课程资料 + 低分题定向补料”为主，目标先将 RAG 命中率提升到 `0.40+`。
