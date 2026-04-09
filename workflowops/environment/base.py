from typing import Dict, Any, Optional, List
from uuid import uuid4
from pydantic import BaseModel, Field
from dataclasses import dataclass


@dataclass
class State:
    episode_id: str
    step_count: int


class Environment:
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    
    @property
    def state(self) -> State:
        raise NotImplementedError()
    
    def reset(self):
        raise NotImplementedError()
    
    def step(self, action):
        raise NotImplementedError()


class Observation(BaseModel):
    done: bool = False
    reward: float = 0.0
    observation: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseWorkflowEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    
    def __init__(self, seed: Optional[int] = None):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.seed = seed
        self.history: List[Dict[str, Any]] = []
        self.max_steps: int = 20
        self.task_state: Dict[str, Any] = {}
        
    def reset(self) -> Observation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.history = []
        self.task_state = {}
        return Observation(
            done=False,
            reward=0.0,
            observation={"status": "ready", "episode_id": self._state.episode_id},
            metadata={"reset_count": len(self.history)}
        )
    
    def step(self, action: Dict[str, Any]) -> Observation:
        self._state.step_count += 1
        
        # Validate action
        if not isinstance(action, dict):
            return Observation(
                done=True,
                reward=-0.5,
                observation={"error": "Invalid action format"},
                metadata={"step": self._state.step_count}
            )
        
        # Record history
        self.history.append({
            "step": self._state.step_count,
            "action": action,
            "timestamp": self._state.episode_id
        })
        
        # Check max steps
        if self._state.step_count >= self.max_steps:
            return Observation(
                done=True,
                reward=0.0,
                observation={"status": "max_steps_reached"},
                metadata={"step": self._state.step_count}
            )
        
        return self._execute_action(action)
    
    def _execute_action(self, action: Dict[str, Any]) -> Observation:
        raise NotImplementedError("Subclasses must implement _execute_action")
    
    @property
    def state(self) -> State:
        return self._state