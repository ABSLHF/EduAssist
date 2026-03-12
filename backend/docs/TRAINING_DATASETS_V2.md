# Training Data V2 (Multi-Subject)

This note describes how to use multi-subject datasets (C-Eval/CMMLU/AGIEval style MCQ) with the current extractive QA training pipeline.

## 1) Convert MCQ to extractive QA

The training pipeline expects:

- `question`
- `context`
- `answers.text[0]`
- `answers.answer_start[0]`

Use `training/convert_mcq_to_extractive.py`:

```bash
python training/convert_mcq_to_extractive.py \
  --input training/data/ceval_raw \
  --out-dir training/data/ceval_extractive \
  --source-name ceval_mcq
```

You can repeat `--input`:

```bash
python training/convert_mcq_to_extractive.py \
  --input training/data/cmmlu_raw \
  --input training/data/agieval_raw \
  --out-dir training/data/mcqa_extractive \
  --source-name mcq_multi_subject
```

## 2) Merge multiple datasets

Use `training/merge_qa_jsonl.py`:

```bash
python training/merge_qa_jsonl.py \
  --dataset base=training/data/edu_mix_qa_public \
  --dataset ceval=training/data/ceval_extractive \
  --dataset cmmlu=training/data/cmmlu_extractive \
  --out-dir training/data/edu_mix_qa_v2 \
  --resplit \
  --validation-ratio 0.1
```

Optional limits:

```bash
python training/merge_qa_jsonl.py \
  --dataset base=training/data/edu_mix_qa_public \
  --dataset ceval=training/data/ceval_extractive \
  --out-dir training/data/edu_mix_qa_v2_small \
  --max-train-per-dataset 6000 \
  --max-validation-per-dataset 800
```

## 3) Train with merged dataset

Call your existing training API with:

- `dataset_name=edu_mix_qa_local`
- `dataset_config=training/data/edu_mix_qa_v2`

## 4) Recommended ratio

- Extractive-native data (CMRC/DRCD/DuReader extractive): at least 70%
- MCQ-converted data (C-Eval/CMMLU/AGIEval): up to 20%
- Course local data: around 10%

This keeps extraction behavior stable while adding multi-subject coverage.

