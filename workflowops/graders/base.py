from typing import Dict, Any, Tuple


class BaseGrader:
    """Base grader class with standard scoring logic"""
    
    def __init__(self, max_score: float = 1.0):
        self.max_score = max_score
        self.efficiency_penalty_per_step = 0.02
        self.invalid_action_penalty = 0.15
    
    def grade(self, state: Dict[str, Any], action: Dict[str, Any], step_count: int) -> Tuple[float, Dict[str, Any]]:
        """
        Grade an action against current state
        Returns: (score [0,1], metadata)
        """
        raise NotImplementedError("Subclasses must implement grade method")
    
    def apply_efficiency_penalty(self, base_score: float, step_count: int) -> float:
        """Apply penalty for taking too many steps"""
        penalty = step_count * self.efficiency_penalty_per_step
        return max(0.0, base_score - penalty)
    
    def apply_invalid_action_penalty(self, score: float) -> float:
        """Apply penalty for invalid actions"""
        return max(0.0, score - self.invalid_action_penalty)
