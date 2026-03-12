# EduAssist API 文档（2026-02-17）

## 启动
- 复制 `.env.sample` 为 `.env`
- 安装依赖：`pip install -r requirements.txt`
- 初始化/升级表结构：`python -m app.migrate`
- 启动服务：`uvicorn app.main:app --reload`

## P0 核心接口
- `POST /auth/register` 注册
- `POST /auth/login` 登录
- `POST /auth/token` Swagger 授权令牌
- `POST /courses/` 教师创建课程
- `GET /courses/` 课程列表
- `POST /courses/{id}/join` 学生加入课程
- `POST /materials?course_id=&title=` 教师上传资料
  - 返回：`material_id`、`chunks`、`parse_status`、`parse_error`、`extracted_chars`
- `POST /qa` 课程问答
  - 请求可选：`history`（多轮对话历史，role/content）
  - 返回：`answer`、`source_type`、`mode`、`references`

## 作业接口
- `POST /assignments/` 发布作业
- `GET /assignments/{course_id}` 作业列表
- `POST /assignments/{assignment_id}/submit` 传统提交入口
- `POST /submissions?assignment_id=` 通用提交入口
- `GET /submissions/{submission_id}/feedback` 获取反馈详情

## 推荐与知识图谱
- `GET /recommendations/{course_id}` 返回推荐项列表
  - 每项含：`knowledge_point`、`reason`、`score`
- `GET /kg/{course_id}` 查询图谱
- `POST /kg/{course_id}/points` 新增知识点
- `POST /kg/{course_id}/edges` 新增关系
- `POST /kg/{course_id}/extract` 自动抽取并入库
- `POST /kg/{course_id}/candidates` 生成候选（可选自动入库）

## 训练接口
- `POST /model/predict` 分类预测
- `POST /model/qa_predict` 抽取式问答预测
- `POST /model/train` 触发训练任务
- `GET /model/train/{job_id}` 查询训练状态
  - `task_type` 支持：
    - `text_classification`（本地 CSV）
    - `text_classification_hf`（HF 分类微调）
    - `qa_extractive_hf`（HF 抽取式问答微调）

## 手动验收顺序
1. 教师注册登录并创建课程。
2. 教师上传资料，确认 `parse_status=success`。
3. 学生加入课程并提问，确认 `mode` 与 `references`。
4. 发布作业并提交，查看反馈接口返回。
5. 调用推荐和图谱候选接口。
6. 触发训练任务并查询状态。
7. 可选：训练 `qa_extractive_hf` 后调用 `/model/qa_predict` 验证抽取答案。

## 自动化验收
- 脚本：`scripts/verify_e2e.py`
- 运行：
  - `python scripts/verify_e2e.py --base-url http://127.0.0.1:8000`
- 输出：
  - `docs/verification_report.json`
  - `docs/verification_report.md`
- 退出码：
  - `0` 全部通过
  - `1` 存在阻塞失败项
