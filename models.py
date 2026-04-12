"""Professional-grade data contracts for AutoClean-Ai.

This module defines the core data structures for a complex RL environment
that trains AI models to perform data cleaning operations on tabular datasets.
"""

from typing import Optional, Dict, Any, List, Literal
from enum import Enum
import uuid
from pydantic import BaseModel, Field

from openenv.core.env_server import Action, Observation, State


class DifficultyLevel(Enum):
    """Difficulty levels for cleaning tasks."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class CleaningActionType(str, Enum):
    """Available data cleaning actions."""
    DROP_NULLS = "drop_nulls"
    FILL_NULLS = "fill_nulls"
    REMOVE_DUPLICATES = "remove_duplicates"
    FILTER_ROWS = "filter_rows"
    DROP_COLUMNS = "drop_columns"
    CONVERT_TYPES = "convert_types"
    VALIDATE_EMAIL = "validate_email"
    OUTLIER_REMOVAL = "outlier_removal"
    NORMALIZE = "normalize"
    SUBMIT = "submit"
    REVERT = "revert"


class DatasetInfo(BaseModel):
    """Dataset metadata and quality metrics."""
    shape: List[int] = Field(default_factory=lambda: [0, 0])
    columns: List[str] = Field(default_factory=list)
    null_counts: Dict[str, int] = Field(default_factory=dict)
    null_percentages: Dict[str, float] = Field(default_factory=dict)
    duplicate_count: int = 0
    dtypes: Dict[str, str] = Field(default_factory=dict)
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    outlier_counts: Dict[str, int] = Field(default_factory=dict)
    quality_score: float = 0.0


class RewardBreakdown(BaseModel):
    """Detailed breakdown of reward components."""
    null_improvement: float = 0.0
    duplicate_improvement: float = 0.0
    outlier_improvement: float = 0.0
    valid_email_count: int = 0
    type_correctness: float = 0.0
    normalization_score: float = 0.0
    efficiency_bonus: float = 0.0
    action_validity: float = 0.0
    progress_bonus: float = 0.0
    penalty: float = 0.0
    total: float = 0.0


class DataCleaningAction(Action):
    """
    Action space for the AI agent.

    The AI must provide:
    - Action type from allowed operations
    - Parameters specific to the action
    """
    action_type: CleaningActionType
    params: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""


class DataCleaningObservation(Observation):
    """
    Observation space with rich feedback signals.

    Provides the AI with detailed information about:
    - Current dataset state and quality metrics
    - Previous action results
    - Detailed reward breakdown
    - Available valid actions
    - Task progress
    """
    # Core dataset info
    dataset_info: DatasetInfo = Field(default_factory=DatasetInfo)
    
    # Episode state
    done: bool = False
    reward: Optional[float] = None
    
    # Feedback
    message: str = ""
    available_actions: List[CleaningActionType] = Field(default_factory=list)
    step_count: int = 0
    task_id: str = ""
    
    # Performance metrics
    quality_score: float = 0.0
    previous_quality: float = 0.0
    quality_improvement: float = 0.0
    
    # Detailed reward breakdown
    reward_breakdown: Optional[RewardBreakdown] = None
    
    # History
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Difficulty and progress
    difficulty_level: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    task_progress: float = 0.0
    
    # Extended metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EpisodeStatistics(BaseModel):
    """Comprehensive statistics for an episode."""
    episode_id: str = ""
    total_steps: int = 0
    initial_quality: float = 0.0
    final_quality: float = 0.0
    quality_improvement: float = 0.0
    nulls_removed: int = 0
    duplicates_removed: int = 0
    outliers_removed: int = 0
    emails_validated: int = 0
    actions_taken: Dict[str, int] = Field(default_factory=dict)
    reward_history: List[float] = Field(default_factory=list)
    efficiency_score: float = 0.0
    total_reward: float = 0.0


class DataCleaningState(State):
    """
    Comprehensive state tracking for the RL environment.

    Tracks episode-level and agent-level state.
    """
    # Episode identification
    episode_id: Optional[str] = None
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Step tracking
    step_count: int = 0
    max_steps: int = 15
    
    # Dataset state
    dataset_info: DatasetInfo = Field(default_factory=DatasetInfo)
    initial_dataset_info: DatasetInfo = Field(default_factory=DatasetInfo)
    
    # Performance tracking
    total_reward: float = 0.0
    reward_history: List[float] = Field(default_factory=list)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Quality metrics
    current_quality_score: float = 0.0
    best_quality_score: float = 0.0
    
    # Task state
    current_task_id: str = ""
    difficulty_level: str = "intermediate"
    
    # Timestamps
    episode_start_time: Optional[float] = None
    last_step_time: Optional[float] = None
    
    # Metadata for extensibility
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "current_quality_score": self.current_quality_score,
            "best_quality_score": self.best_quality_score,
            "total_reward": self.total_reward,
            "current_task_id": self.current_task_id,
            "difficulty_level": self.difficulty_level,
            **self.metadata
        }


class EnvironmentConfig(BaseModel):
    """Configuration for the data cleaning environment."""
    # Episode configuration
    max_steps_per_episode: int = 15
    min_steps_for_completion: int = 3
    
    # Early stopping configuration
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_reward: float = 0.01
    
    # Reward configuration
    reward_weights: Dict[str, float] = Field(default_factory=lambda: {
        "null_improvement": 0.25,
        "duplicate_improvement": 0.20,
        "outlier_improvement": 0.20,
        "email_validation": 0.15,
        "type_correctness": 0.10,
        "efficiency": 0.10,
    })
    
    # Difficulty configuration
    initial_difficulty: str = "intermediate"
    adaptive_difficulty: bool = True
    
    # Task configuration
    tasks: List[Dict[str, Any]] = Field(default_factory=list)