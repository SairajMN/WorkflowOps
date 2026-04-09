from typing import Dict, Any, Tuple
from graders.base import BaseGrader
from environment.base import BaseWorkflowEnvironment, Observation


class SupportTicketGrader(BaseGrader):
    PRIORITIES = ["critical", "high", "medium", "low"]
    
    def grade(self, state: Dict[str, Any], action: Dict[str, Any], step_count: int) -> Tuple[float, Dict[str, Any]]:
        ticket = state.get("current_ticket", {})
        correct_priority = ticket.get("correct_priority")
        
        selected_priority = action.get("priority")
        estimated_time = action.get("estimated_time", 0)
        assignee = action.get("assignee")
        
        score = 0.0
        
        # Priority score (60%)
        if selected_priority == correct_priority:
            score += 0.6
        else:
            priority_distance = abs(self.PRIORITIES.index(selected_priority) - self.PRIORITIES.index(correct_priority))
            score += max(0.0, 0.6 - (priority_distance * 0.2))
        
        # Time estimation score (30%)
        correct_time = ticket.get("correct_time", 0)
        time_error = abs(estimated_time - correct_time) / max(correct_time, 1)
        score += max(0.0, 0.3 - (time_error * 0.3))
        
        # Assignee score (10%)
        if assignee == ticket.get("correct_assignee"):
            score += 0.1
        
        final_score = self.apply_efficiency_penalty(score, step_count)
        
        return final_score, {
            "priority_score": score >= 0.6,
            "time_score": score >= 0.3,
            "assignee_score": score >= 0.1,
            "total": final_score
        }


class SupportTicketTask(BaseWorkflowEnvironment):
    TEST_TICKETS = [
        {
            "id": 101,
            "title": "Database connection failure",
            "description": "Cannot connect to primary database. All transactions failing.",
            "correct_priority": "critical",
            "correct_time": 30,
            "correct_assignee": "database-team"
        },
        {
            "id": 102,
            "title": "User password reset request",
            "description": "User cannot log in, needs password reset.",
            "correct_priority": "medium",
            "correct_time": 10,
            "correct_assignee": "support"
        },
        {
            "id": 103,
            "title": "Feature request: Dark mode",
            "description": "Would like dark mode option for dashboard.",
            "correct_priority": "low",
            "correct_time": 480,
            "correct_assignee": "frontend"
        }
    ]
    
    def reset(self) -> Observation:
        super().reset()
        self.task_state["ticket_index"] = 0
        self.task_state["current_ticket"] = self.TEST_TICKETS[0]
        self.task_state["score"] = 0.0
        
        return Observation(
            done=False,
            reward=0.0,
            observation={
                "task": "support_ticket",
                "ticket": self.task_state["current_ticket"],
                "priorities": SupportTicketGrader.PRIORITIES
            }
        )
    
    def _execute_action(self, action: Dict[str, Any]) -> Observation:
        grader = SupportTicketGrader()
        score, meta = grader.grade(self.task_state, action, self._state.step_count)
        
        self.task_state["score"] += score
        self.task_state["ticket_index"] += 1
        
        if self.task_state["ticket_index"] >= len(self.TEST_TICKETS):
            final_score = self.task_state["score"] / len(self.TEST_TICKETS)
            return Observation(
                done=True,
                reward=final_score,
                observation={"final_score": final_score},
                metadata=meta
            )
        
        self.task_state["current_ticket"] = self.TEST_TICKETS[self.task_state["ticket_index"]]
        
        return Observation(
            done=False,
            reward=score,
            observation={
                "ticket": self.task_state["current_ticket"],
                "current_score": self.task_state["score"]
            },
            metadata=meta
        )
