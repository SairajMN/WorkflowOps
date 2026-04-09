from typing import Dict, Any, Tuple
from graders.base import BaseGrader
from environment.base import BaseWorkflowEnvironment, Observation


class ScheduleConflictGrader(BaseGrader):
    def grade(self, state: Dict[str, Any], action: Dict[str, Any], step_count: int) -> Tuple[float, Dict[str, Any]]:
        events = state.get("events", [])
        solution = action.get("schedule", [])
        
        if not isinstance(solution, list):
            return self.apply_invalid_action_penalty(0.0), {"error": "Invalid schedule format"}
        
        # Check for overlaps
        overlaps = 0
        for i in range(len(solution)):
            for j in range(i+1, len(solution)):
                e1 = solution[i]
                e2 = solution[j]
                if not (e1["end"] <= e2["start"] or e2["end"] <= e1["start"]):
                    overlaps += 1
        
        overlap_penalty = overlaps * 0.25
        
        # Check all events are scheduled
        missing_events = len(events) - len(solution)
        missing_penalty = missing_events * 0.15
        
        base_score = 1.0 - overlap_penalty - missing_penalty
        final_score = self.apply_efficiency_penalty(max(0.0, base_score), step_count)
        
        return final_score, {
            "overlaps": overlaps,
            "missing_events": missing_events,
            "base_score": base_score,
            "final_score": final_score
        }


class ScheduleConflictTask(BaseWorkflowEnvironment):
    TEST_SCHEDULES = [
        {
            "events": [
                {"id": 1, "title": "Team Standup", "duration": 15, "priority": 2},
                {"id": 2, "title": "Client Call", "duration": 60, "priority": 1},
                {"id": 3, "title": "Code Review", "duration": 30, "priority": 3},
                {"id": 4, "title": "Lunch Break", "duration": 45, "priority": 2}
            ],
            "available_slots": [
                {"start": 540, "end": 720},  # 9:00 - 12:00
                {"start": 780, "end": 1020}  # 13:00 - 17:00
            ]
        }
    ]
    
    def reset(self) -> Observation:
        super().reset()
        self.task_state["schedule_index"] = 0
        self.task_state["events"] = self.TEST_SCHEDULES[0]["events"]
        self.task_state["available_slots"] = self.TEST_SCHEDULES[0]["available_slots"]
        
        return Observation(
            done=False,
            reward=0.0,
            observation={
                "task": "schedule_conflict",
                "events": self.task_state["events"],
                "available_slots": self.task_state["available_slots"]
            }
        )
    
    def _execute_action(self, action: Dict[str, Any]) -> Observation:
        grader = ScheduleConflictGrader()
        score, meta = grader.grade(self.task_state, action, self._state.step_count)
        
        return Observation(
            done=True,
            reward=score,
            observation={"final_score": score},
            metadata=meta
        )
