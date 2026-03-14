# 作业批改 SFT（AutoDL 同一实例重启）执行手册

本手册用于“同一 AutoDL 实例重启后”继续完成：

1. 资产校验  
2. SFT 训练提交与轮询  
3. 启用本地 SFT 模型  
4. 四类样例回归验收

## 0. 终端分工

- 终端A：运行后端服务（`uvicorn`）
- 终端B：执行训练、配置、验收命令

以下命令默认在 AutoDL 容器中执行。

## 1. 终端B：资产与环境校验

```bash
cd /root/autodl-tmp/EduAssist/backend
source .venv/bin/activate
python -V

python scripts/check_assignment_feedback_sft_assets.py \
  --backend-dir /root/autodl-tmp/EduAssist/backend \
  --model-dir /root/autodl-tmp/hf_models/Qwen2.5-7B-Instruct \
  --dataset-dir training/data/assignment_feedback_sft_mix
```

期望：输出 JSON 中 `"ok": true`。  
若为 `false`，先补齐缺失目录/文件，再继续。

## 2. 终端A：启动训练 API 服务

```bash
cd /root/autodl-tmp/EduAssist/backend
source .venv/bin/activate
python -m app.migrate
pkill -f "uvicorn.*app.main_train:app.*8001" || true
uvicorn --app-dir . app.main_train:app --host 0.0.0.0 --port 8001
```

保持终端A不关闭。

## 3. 终端B：连通性与登录校验

```bash
cd /root/autodl-tmp/EduAssist/backend
source .venv/bin/activate
curl -s http://127.0.0.1:8001/

TOKEN=$(curl -s -X POST "http://127.0.0.1:8001/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"teacher_train","password":"123456"}' \
  | python -c "import sys,json;print(json.load(sys.stdin).get('access_token',''))")
echo "$TOKEN"
```

## 4. 终端B：提交 SFT 训练任务

```bash
cd /root/autodl-tmp/EduAssist/backend
source .venv/bin/activate

python scripts/run_assignment_feedback_sft_autodl.py \
  --skip-build \
  --base-url http://127.0.0.1:8001 \
  --username teacher_train \
  --password 123456 \
  --dataset-name assignment_feedback_sft_mix \
  --dataset-config training/data/assignment_feedback_sft_mix \
  --model-name /root/autodl-tmp/hf_models/Qwen2.5-7B-Instruct \
  --epochs 2 \
  --batch-size 1 \
  --learning-rate 2e-4 \
  --max-samples 16000
```

期望：最终输出 `status=success`，并看到本次 `model_path`。

## 5. 终端B：启用本地 SFT 模型

将 `<NEW_MODEL_PATH>` 替换为上一步输出的 `model_path` 绝对路径。

```bash
cd /root/autodl-tmp/EduAssist/backend
source .venv/bin/activate

grep -q '^ENABLE_ASSIGNMENT_FEEDBACK_SFT_MODEL=' .env \
  && sed -i 's|^ENABLE_ASSIGNMENT_FEEDBACK_SFT_MODEL=.*|ENABLE_ASSIGNMENT_FEEDBACK_SFT_MODEL=true|' .env \
  || echo 'ENABLE_ASSIGNMENT_FEEDBACK_SFT_MODEL=true' >> .env

grep -q '^ASSIGNMENT_FEEDBACK_SFT_MODEL_PATH=' .env \
  && sed -i "s|^ASSIGNMENT_FEEDBACK_SFT_MODEL_PATH=.*|ASSIGNMENT_FEEDBACK_SFT_MODEL_PATH=<NEW_MODEL_PATH>|" .env \
  || echo "ASSIGNMENT_FEEDBACK_SFT_MODEL_PATH=<NEW_MODEL_PATH>" >> .env

grep -q '^ASSIGNMENT_FEEDBACK_EXTERNAL_FALLBACK=' .env \
  && sed -i 's|^ASSIGNMENT_FEEDBACK_EXTERNAL_FALLBACK=.*|ASSIGNMENT_FEEDBACK_EXTERNAL_FALLBACK=true|' .env \
  || echo 'ASSIGNMENT_FEEDBACK_EXTERNAL_FALLBACK=true' >> .env
```

## 6. 终端A：切业务服务并重启

先 `Ctrl + C` 停掉 `main_train`，再执行：

```bash
cd /root/autodl-tmp/EduAssist/backend
source .venv/bin/activate
python -m app.migrate
pkill -f "uvicorn.*app.main:app.*8000" || true
uvicorn --app-dir . app.main:app --host 0.0.0.0 --port 8000
```

## 7. 终端B：验证模型已启用

```bash
cd /root/autodl-tmp/EduAssist/backend
source .venv/bin/activate

TOKEN=$(curl -s -X POST "http://127.0.0.1:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"teacher_train","password":"123456"}' \
  | python -c "import sys,json;print(json.load(sys.stdin).get('access_token',''))")

curl -s "http://127.0.0.1:8000/model/active" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

期望字段：

- `enable_assignment_feedback_sft_model: true`
- `assignment_feedback_sft_model_path: <NEW_MODEL_PATH>`

## 8. 终端B：四类样例回归

先准备一个测试作业 `assignment_id`（文本作业）。

```bash
cd /root/autodl-tmp/EduAssist/backend
source .venv/bin/activate

python scripts/verify_assignment_feedback_four_cases.py \
  --base-url http://127.0.0.1:8000 \
  --username student_train \
  --password 123456 \
  --assignment-id <ASSIGNMENT_ID>
```

验收要点：

- `feedback` 非空
- `score` 为 `null`
- 无多余 Markdown 噪声
- invalid/off_topic 不出现“优点”

## 9. 失败回滚（仅开关回滚）

若线上效果不满足预期，仅回滚开关，不删除训练产物：

```bash
cd /root/autodl-tmp/EduAssist/backend
source .venv/bin/activate
grep -q '^ENABLE_ASSIGNMENT_FEEDBACK_SFT_MODEL=' .env \
  && sed -i 's|^ENABLE_ASSIGNMENT_FEEDBACK_SFT_MODEL=.*|ENABLE_ASSIGNMENT_FEEDBACK_SFT_MODEL=false|' .env \
  || echo 'ENABLE_ASSIGNMENT_FEEDBACK_SFT_MODEL=false' >> .env
```

然后重启 `app.main` 服务。
