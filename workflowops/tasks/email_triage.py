from typing import Dict, Any, Tuple
from graders.base import BaseGrader
from environment.base import BaseWorkflowEnvironment, Observation


class EmailTriageGrader(BaseGrader):
    CATEGORIES = ["urgent", "important", "normal", "spam"]
    
    def grade(self, state: Dict[str, Any], action: Dict[str, Any], step_count: int) -> Tuple[float, Dict[str, Any]]:
        email = state.get("current_email", {})
        correct_category = email.get("correct_category")
        
        selected_category = action.get("category")
        
        if not selected_category or selected_category not in self.CATEGORIES:
            return self.apply_invalid_action_penalty(0.0), {"error": "Invalid category"}
        
        if selected_category == correct_category:
            base_score = 1.0
        else:
            # Partial credit for near correct
            if correct_category == "urgent" and selected_category == "important":
                base_score = 0.6
            elif correct_category == "important" and selected_category == "urgent":
                base_score = 0.6
            else:
                base_score = 0.2
        
        final_score = self.apply_efficiency_penalty(base_score, step_count)
        
        return final_score, {
            "base_score": base_score,
            "correct": selected_category == correct_category,
            "step_count": step_count
        }


class EmailTriageTask(BaseWorkflowEnvironment):
    TEST_EMAILS = [
        {
            "id": 1,
            "subject": "SERVER OUTAGE - All systems down",
            "body": "Production servers are not responding. Immediate action required.",
            "correct_category": "urgent",
            "sender": "sysadmin@company.com"
        },
        {
            "id": 2,
            "subject": "Quarterly Performance Review",
            "body": "Please schedule your performance review meeting.",
            "correct_category": "important",
            "sender": "hr@company.com"
        },
        {
            "id": 3,
            "subject": "Team lunch tomorrow",
            "body": "We are having team lunch at 1pm. Let us know if you are coming.",
            "correct_category": "normal",
            "sender": "team@company.com"
        },
        {
            "id": 4,
            "subject": "YOU WON $1,000,000!!!",
            "body": "Click here to claim your prize now!",
            "correct_category": "spam",
            "sender": "scam@fake.com"
        }
    ]
    
    def reset(self) -> Observation:
        super().reset()
        self.task_state["email_index"] = 0
        self.task_state["current_email"] = self.TEST_EMAILS[0]
        self.task_state["graded"] = 0
        self.task_state["score"] = 0.0
        
        return Observation(
            done=False,
            reward=0.0,
            observation={
                "task": "email_triage",
                "email": self.task_state["current_email"],
                "categories": EmailTriageGrader.CATEGORIES,
                "remaining": len(self.TEST_EMAILS)
            }
        )
    
    def _execute_action(self, action: Dict[str, Any]) -> Observation:
        grader = EmailTriageGrader()
        score, meta = grader.grade(self.task_state, action, self._state.step_count)
        
        self.task_state["score"] += score
        self.task_state["graded"] += 1
        
        # Move to next email
        self.task_state["email_index"] += 1
        
        if self.task_state["email_index"] >= len(self.TEST_EMAILS):
            final_score = self.task_state["score"] / len(self.TEST_EMAILS)
            return Observation(
                done=True,
                reward=final_score,
                observation={"final_score": final_score, "total_graded": self.task_state["graded"]},
                metadata=meta
            )
        
        self.task_state["current_email"] = self.TEST_EMAILS[self.task_state["email_index"]]
        
        return Observation(
            done=False,
            reward=score,
            observation={
                "email": self.task_state["current_email"],
                "current_score": self.task_state["score"],
                "remaining": len(self.TEST_EMAILS) - self.task_state["email_index"]
            },
            metadata=meta
        )
