#!/usr/bin/env python3
import os
import sys
import json
import time
from typing import Dict, Any
from openai import OpenAI
import requests


def main():
    print("[START]")
    
    api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    env_url = os.getenv("ENV_URL", "http://localhost:8000")
    task = os.getenv("TASK_NAME", "email_triage")
    
    client = OpenAI(base_url=api_base, api_key=os.getenv("OPENAI_API_KEY"))
    
    # Reset environment
    resp = requests.post(f"{env_url}/env/{task}/reset")
    resp.raise_for_status()
    data = resp.json()
    
    episode_id = data["episode_id"]
    observation = data["observation"]
    
    step = 0
    
    while not observation.get("done", False):
        step += 1
        print(f"[STEP] {step}")
        
        try:
            # Call LLM
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an operations assistant. Respond ONLY with valid JSON action."},
                    {"role": "user", "content": json.dumps(observation["observation"])}
                ],
                temperature=0.0,
                max_tokens=256
            )
            
            action_text = response.choices[0].message.content.strip()
            action = json.loads(action_text)
            
            # Execute step
            resp = requests.post(f"{env_url}/env/step", json={
                "episode_id": episode_id,
                "action": action
            })
            resp.raise_for_status()
            observation = resp.json()
            
        except Exception as e:
            print(f"[ERROR] {str(e)}", file=sys.stderr)
            # Fallback action
            resp = requests.post(f"{env_url}/env/step", json={
                "episode_id": episode_id,
                "action": {}
            })
            observation = resp.json()
        
        time.sleep(0.1)
    
    print(f"[FINAL SCORE] {observation.get('reward', 0.0)}")
    print("[END]")


if __name__ == "__main__":
    main()
