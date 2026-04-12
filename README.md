---
title: AutoClean-Ai
emoji: 🧹
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - data-cleaning
  - data-preprocessing
  - llm-training
  - benchmark
  - ai-safety
  - data-quality
  - mlops
---

# 🧹 AutoClean-Ai

> **Production-grade OpenEnv RL environment for training AI models to clean tabular data automatically.**

**Server Version:** v1.0.0

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](#-quick-start)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Realistic%20Generated-orange)](#-datasets)

---

## 💡 The Problem

80% of data scientist time is spent cleaning data. Bad data causes 60% of ML project failures. AutoClean-Ai was built to train AI agents that can automatically detect and fix common data quality issues in tabular datasets.

## 🚀 Quick Start

### Run Locally

```bash
git clone https://github.com/SairajMN/WorkflowOps.git
cd WorkflowOps
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
curl http://localhost:7860/health
```

### Raw HTTP

```python
import requests

BASE = "http://localhost:7860"

# 1. Start episode
obs = requests.post(f"{BASE}/reset", json={"difficulty": "beginner"}).json()
print(obs["dataset_preview"], obs["column_info"])

# 2. Submit cleaning action
result = requests.post(f"{BASE}/step", json={
    "action_type": "fix_missing_values",
    "column_index": 2,
    "confidence": 0.92,
    "reasoning": "Mean imputation for numerical column",
    "session_id": obs.get("session_id"),
}).json()
print(f"Reward: {result['reward']}, Cleaned: {result['rows_cleaned']}")

# 3. Score the episode
grade = requests.post(f"{BASE}/grader", json={
    "task_id": "task_1_basic_cleaning",
    "step_rewards": [result['reward']],
    "step_infos": [result],
}).json()
print(f"Episode score: {grade['score']}")
```

### Validate OpenEnv Compliance

```bash
# Local structure check
openenv validate

# Runtime check against live server
openenv validate --url http://localhost:7860 --verbose
```

---

## 🎯 Tasks

3 progressive difficulty tasks:

| # | task_id | Difficulty | Description | Expected Agent Score |
|---|---------|-----------|-------------|-------------------|
| 1 | `task_1_basic_cleaning` | 🟢 Beginner | Fix missing values, standardize formats | 0.70–0.85 |
| 2 | `task_2_advanced_cleaning` | 🟡 Intermediate | Handle outliers, correct data types, deduplication | 0.55–0.70 |
| 3 | `task_3_full_pipeline` | 🔴 Advanced | Complete end-to-end data cleaning pipeline | 0.40–0.60 |

---

## 🎮 Environment Workflow

The agent receives a **tabular dataset** with known quality issues. It must select the appropriate cleaning operation, apply it correctly, and justify its choice.

### Action Space

```json
{
    "action_type":      "fix_missing_values | remove_outliers | standardize | deduplicate | correct_types | fill_dates",
    "column_index":     3,
    "confidence":       0.85,
    "reasoning":        "string explaining the choice",
    "session_id":       "session id from reset"
}
```

### Observation Space

```json
{
    "dataset_preview":   "First 5 rows of data",
    "column_info":       "Column names, types, missing stats",
    "reward":            0.75,
    "feedback":          "Detailed human-readable feedback",
    "rows_cleaned":      12,
    "issues_remaining":  3,
    "done":              false,
    "session_id":        "ses_a1b2c3d4"
}
```

---

## 📊 Reward System (7 Components)

| Component | Weight | Description |
|-----------|--------|-------------|
| Correctness | 0.35 | Operation actually fixed the issue |
| Appropriate action | 0.25 | Right operation selected for the problem |
| Confidence calibration | 0.15 | Confidence matches actual correctness |
| No side effects | 0.15 | Cleaning didn't break other columns |
| Efficiency | 0.10 | Minimum steps to clean dataset |

---

## 📈 Metrics

✅ Data Quality Score
✅ Completeness Ratio
✅ Uniqueness Ratio
✅ Type Consistency
✅ Cleaning Efficiency
✅ Action Appropriateness

---

## 📋 Supported Data Cleaning Operations

| Operation | Description |
|-----------|-------------|
| `fix_missing_values` | Mean/median/mode imputation |
| `remove_outliers` | IQR / Z-score outlier removal |
| `standardize` | Normalize numerical columns |
| `deduplicate` | Remove duplicate rows |
| `correct_types` | Fix incorrect data types |
| `fill_dates` | Standardize date formats |
| `handle_categories` | Encode categorical columns |
| `remove_duplicates` | Drop identical rows |
| `trim_strings` | Clean whitespace from text columns |
| `correct_values` | Fix known invalid values |

---

## 📀 API Endpoints

### OpenEnv Required

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/tasks` | List all 3 tasks + action schema |
| `POST` | `/grader` | Score a completed episode (0.0–1.0) |
| `POST` | `/baseline` | Run built-in heuristic baseline |
| `GET` | `/metadata` | Environment name, version, description |
| `GET` | `/schema` | Action, observation, and state JSON schemas |
| `GET` | `/health` | Health check |

### Environment

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Submit cleaning action |
| `GET` | `/state` | Get current episode state |

---

## 💻 Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Validate OpenEnv compliance
openenv validate --url http://localhost:7860 --verbose
```

---

## 🔗 Links

| | |
|---|---|
| 📦 GitHub | https://github.com/SairajMN/WorkflowOps |
| 📖 Interactive API Docs | http://localhost:7860/redoc |
| 🔧 OpenEnv Framework | https://github.com/meta-pytorch/OpenEnv |

---

*Built for Data Cleaning AI Agents · MIT License*