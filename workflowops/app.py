import os
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPExceptio
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any
import uvicor

from tasks.email_triage import EmailTriageTask
from tasks.support_ticket import SupportTicketTask
from tasks.schedule_conflict import ScheduleConflictTask

app = FastAPI(title="WorkflowOps OpenEnv", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/web")
async def web():
    return RedirectResponse(url="/static/index.html")

TASKS = {
    "email_triage": EmailTriageTask,
    "support_ticket": SupportTicketTask,
    "schedule_conflict": ScheduleConflictTask
}

active_envs: Dict[str, Any] = {}


class ActionRequest(BaseModel):
    episode_id: str
    action: Dict[str, Any]


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "workflowops"}


@app.get("/ready")
async def ready():
    return {"ready": True, "tasks": list(TASKS.keys())}


@app.post("/env/{task_name}/reset")
@app.post("/{task_name}/reset")
async def reset_environment(task_name: str):
    if task_name not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    
    env = TASKS[task_name]()
    observation = env.reset()
    
    active_envs[env.state.episode_id] = env
    
    return {
        "episode_id": env.state.episode_id,
        "observation": observation.model_dump()
    }


@app.post("/env/step")
@app.post("/step")
async def step_environment(request: ActionRequest):
    if request.episode_id not in active_envs:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    env = active_envs[request.episode_id]
    observation = env.step(request.action)
    
    if observation.done:
        del active_envs[request.episode_id]
    
    return observation.model_dump()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
