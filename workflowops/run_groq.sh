#!/bin/bash
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export OPENAI_API_KEY=your_groq_api_key_here
export ENV_URL=http://localhost:8000
export TASK_NAME=email_triage

echo "✅ Running WorkflowOps with Groq Llama 3.1 70B"
python3 inference.py
