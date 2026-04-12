"""Professional-grade AutoClean-Ai Data Cleaning Environment.

This module implements a sophisticated, production-ready RL environment with:
- 10 standard data cleaning operations
- 3 progressive difficulty tasks
- Shaped rewards with partial progress signals
- Comprehensive episode management
- Model-agnostic design (works with any LLM)
- Real-time metrics and logging
- Session management for concurrent users
"""

import uuid
import time
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

# Add directories to path for imports to work in both local and HF Spaces
import sys
import os
_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_dir)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from openenv.core.env_server import Environment

from models import (
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
    EpisodeStatistics,
    RewardBreakdown,
    DatasetInfo,
    DifficultyLevel,
    EnvironmentConfig,
    CleaningActionType,
)
# Import from same directory for HF Spaces deployment compatibility
from grader import (
    calculate_reward,
    calculate_dataset_quality_score,
    grade_task_result,
)
from dataset_loader import DatasetGenerator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EpisodePhase(Enum):
    """Phases of an episode."""
    INITIALIZATION = "initialization"
    ACTIVE = "active"
    CLEANING = "cleaning"
    GRADING = "grading"
    COMPLETION = "completion"


class DataCleaningEnvironment(Environment[DataCleaningAction, DataCleaningObservation, DataCleaningState]):
    """
    Professional-grade OpenEnv environment for training AI to perform data cleaning.

    Features:
    - 10 standard data cleaning operations
    - 3 progressive difficulty tasks
    - Shaped rewards with partial progress signals
    - Deterministic grading 0.0-1.0
    - Comprehensive metrics tracking
    - Session management
    """

    SUPPORTS_CONCURRENT_SESSIONS = True
    VERSION = "1.0.0"

    def __init__(
        self,
        transform=None,
        config: Optional[EnvironmentConfig] = None,
        session_id: Optional[str] = None,
        dataset_generator: Optional["DatasetGenerator"] = None,
        dataset_loader: Optional["DatasetGenerator"] = None
    ):
        super().__init__(transform=transform)

        # Configuration
        self.config = config or EnvironmentConfig()
        self.session_id = session_id or str(uuid.uuid4())[:8]

        # Dataset management (support both parameter names for backwards compatibility)
        self.dataset_generator = dataset_generator or dataset_loader or DatasetGenerator()
        self.dataset_loader = self.dataset_generator
        
        # Episode state
        self.episode_id: Optional[str] = None
        self.episode_phase: EpisodePhase = EpisodePhase.INITIALIZATION
        self.step_count: int = 0
        
        # Current dataset
        self.df: Optional[pd.DataFrame] = None
        self.initial_df: Optional[pd.DataFrame] = None
        self.dataset_history: List[pd.DataFrame] = []
        
        # Performance tracking
        self.reward_history: List[float] = []
        self.action_history: List[Dict[str, Any]] = []
        self.quality_history: List[float] = []
        
        # Early stopping tracking
        self.consecutive_noop_actions: int = 0
        self.consecutive_repeated_actions: int = 0
        self.early_stop_reason: Optional[str] = None
        
        # Performance metrics
        self.last_step_time: Optional[float] = None
        self.episode_start_time: Optional[float] = None
        
        # Task state
        self.current_task_id: str = ""
        self.current_difficulty: str = "intermediate"
        
        logger.info(f"Initialized DataCleaningEnvironment (session={self.session_id})")

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs
    ) -> DataCleaningObservation:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            episode_id: Custom episode ID
            difficulty: Starting difficulty level
            task_id: Specific task to run

        Returns:
            Initial observation
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate episode ID
        self.episode_id = episode_id or f"ep_{uuid.uuid4().hex[:8]}"
        self.episode_start_time = time.time()
        self.last_step_time = time.time()

        # Reset counters
        self.step_count = 0
        self.reward_history = []
        self.action_history = []
        self.quality_history = []
        self.dataset_history = []
        self.consecutive_noop_actions = 0
        self.consecutive_repeated_actions = 0
        self.early_stop_reason = None

        # Determine task and difficulty
        if task_id:
            self.current_task_id = task_id
        else:
            # Map difficulty to default task
            diff_task_map = {
                "beginner": "task_1_basic_cleaning",
                "intermediate": "task_2_intermediate_cleaning", 
                "advanced": "task_3_full_pipeline"
            }
            self.current_task_id = diff_task_map.get(difficulty or "intermediate", "task_1_basic_cleaning")

        # Generate dataset based on task
        self.df = self.dataset_generator.generate_dataset(self.current_task_id, seed=seed)
        self.initial_df = self.df.copy()
        self.dataset_history.append(self.df.copy())
        
        # Calculate initial quality score
        initial_quality = calculate_dataset_quality_score(self.df, self.current_task_id)
        self.quality_history.append(initial_quality)
        
        self.episode_phase = EpisodePhase.ACTIVE

        logger.info(f"Reset episode {self.episode_id} task={self.current_task_id} rows={len(self.df)}")

        return self._create_observation(
            message="Episode started. Perform data cleaning operations on the dataset.",
            metadata={"phase": self.episode_phase.value}
        )

    def step(
        self,
        action: DataCleaningAction,
        **kwargs
    ) -> DataCleaningObservation:
        """
        Process the AI's action and return the next observation.

        Executes the requested data cleaning operation, calculates reward,
        and returns updated state.
        """
        current_time = time.time()
        step_duration = current_time - (self.last_step_time or current_time)
        self.last_step_time = current_time

        if self.df is None:
            return self._create_error_observation("No active dataset. Call /reset first.")

        # Handle submit action
        if action.action_type == CleaningActionType.SUBMIT:
            return self._end_episode()

        # Handle revert action
        if action.action_type == CleaningActionType.REVERT:
            if len(self.dataset_history) > 1:
                self.dataset_history.pop()
                self.df = self.dataset_history[-1].copy()
                return self._create_observation(
                    message="Reverted to previous state",
                    reward=0.0
                )
            else:
                return self._create_observation(
                    message="Cannot revert: no previous state available",
                    reward=-0.05
                )

        # Validate and execute action
        try:
            self.df = self._execute_action(self.df, action)
            self.dataset_history.append(self.df.copy())
            
            # Calculate reward and quality
            previous_quality = self.quality_history[-1] if self.quality_history else 0.0
            current_quality = calculate_dataset_quality_score(self.df, self.current_task_id)
            
            reward, reward_info = calculate_reward(
                df=self.df,
                initial_df=self.initial_df,
                previous_quality=previous_quality,
                current_quality=current_quality,
                action=action,
                task_id=self.current_task_id,
                step_count=self.step_count
            )
            
            self.quality_history.append(current_quality)
            
            # Update tracking
            self.reward_history.append(reward)
            self.action_history.append({
                "action_type": action.action_type,
                "params": action.params,
                "reward": reward,
                "quality_improvement": current_quality - previous_quality
            })
            
            # Check for repeated actions
            if len(self.action_history) >= 2:
                last_action = self.action_history[-2]
                if last_action["action_type"] == action.action_type and last_action["params"] == action.params:
                    self.consecutive_repeated_actions += 1
                    reward -= 0.05 * self.consecutive_repeated_actions
                else:
                    self.consecutive_repeated_actions = 0

            self.step_count += 1

            # Check for early stopping
            early_stop = self._check_early_stopping()
            
            done = self.step_count >= self.config.max_steps_per_episode or early_stop

            if early_stop:
                done = True
                self.early_stop_reason = early_stop
                self.episode_phase = EpisodePhase.COMPLETION

            return self._create_observation(
                message=f"Executed {action.action_type} successfully",
                reward=reward,
                done=done,
                metadata={
                    "step": self.step_count,
                    "previous_quality": previous_quality,
                    "current_quality": current_quality,
                    "quality_improvement": current_quality - previous_quality,
                    "reward_breakdown": reward_info,
                }
            )

        except Exception as e:
            logger.warning(f"Action execution failed: {e}")
            return self._create_error_observation(f"Invalid action: {str(e)}")

    def state(self) -> DataCleaningState:
        """Return comprehensive state of the environment."""
        # Calculate derived metrics
        avg_reward = sum(self.reward_history) / max(1, len(self.reward_history))
        current_quality = self.quality_history[-1] if self.quality_history else 0.0
        best_quality = max(self.quality_history) if self.quality_history else 0.0

        # Build episode statistics
        episode_stats = EpisodeStatistics(
            episode_id=self.episode_id or "",
            total_steps=self.step_count,
            initial_quality=self.quality_history[0] if self.quality_history else 0.0,
            final_quality=current_quality,
            quality_improvement=current_quality - (self.quality_history[0] if self.quality_history else 0.0),
            actions_taken={action["action_type"]: sum(1 for a in self.action_history if a["action_type"] == action["action_type"]) for action in self.action_history},
            reward_history=self.reward_history.copy(),
            total_reward=sum(self.reward_history),
        )

        dataset_info = self._get_dataset_info(self.df) if self.df is not None else DatasetInfo()
        initial_dataset_info = self._get_dataset_info(self.initial_df) if self.initial_df is not None else DatasetInfo()

        return DataCleaningState(
            episode_id=self.episode_id,
            session_id=self.session_id,
            step_count=self.step_count,
            max_steps=self.config.max_steps_per_episode,
            dataset_info=dataset_info,
            initial_dataset_info=initial_dataset_info,
            total_reward=sum(self.reward_history),
            reward_history=self.reward_history.copy(),
            action_history=self.action_history.copy(),
            current_quality_score=current_quality,
            best_quality_score=best_quality,
            current_task_id=self.current_task_id,
            difficulty_level=self.current_difficulty,
            episode_start_time=self.episode_start_time if hasattr(self, 'episode_start_time') else None,
            last_step_time=self.last_step_time,
            metadata={
                "phase": self.episode_phase.value,
                "version": self.VERSION,
            }
        )

    def close(self) -> None:
        """Clean up resources."""
        logger.info(f"Closed environment (session={self.session_id})")

    def _execute_action(self, df: pd.DataFrame, action: DataCleaningAction) -> pd.DataFrame:
        """Execute the requested cleaning action on the dataframe."""
        params = action.params
        
        if action.action_type == CleaningActionType.DROP_NULLS:
            column = params.get("column")
            if column and column in df.columns:
                return df.dropna(subset=[column]).reset_index(drop=True)
            else:
                return df.dropna().reset_index(drop=True)

        elif action.action_type == CleaningActionType.FILL_NULLS:
            column = params.get("column")
            strategy = params.get("strategy", "mean")
            
            if column not in df.columns:
                raise ValueError(f"Column {column} not found")
                
            if strategy == "mean":
                df[column] = df[column].fillna(df[column].mean())
            elif strategy == "median":
                df[column] = df[column].fillna(df[column].median())
            elif strategy == "mode":
                df[column] = df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 0)
            elif strategy == "forward_fill":
                df[column] = df[column].ffill()
            elif strategy == "backward_fill":
                df[column] = df[column].bfill()
            return df

        elif action.action_type == CleaningActionType.REMOVE_DUPLICATES:
            columns = params.get("columns")
            if columns:
                return df.drop_duplicates(subset=columns).reset_index(drop=True)
            else:
                return df.drop_duplicates().reset_index(drop=True)

        elif action.action_type == CleaningActionType.VALIDATE_EMAIL:
            column = params.get("column")
            drop_invalid = params.get("drop_invalid", False)
            
            if column not in df.columns:
                raise ValueError(f"Column {column} not found")
                
            # Simple email validation regex
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            valid_mask = df[column].astype(str).str.match(email_pattern, na=False)
            
            if drop_invalid:
                return df[valid_mask].reset_index(drop=True)
            else:
                return df

        elif action.action_type == CleaningActionType.OUTLIER_REMOVAL:
            column = params.get("column")
            multiplier = params.get("multiplier", 1.5)
            
            if column not in df.columns:
                raise ValueError(f"Column {column} not found")
                
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].reset_index(drop=True)

        elif action.action_type == CleaningActionType.CONVERT_TYPES:
            column = params.get("column")
            dtype = params.get("dtype")
            
            if column not in df.columns:
                raise ValueError(f"Column {column} not found")
                
            if dtype == "int":
                df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
            elif dtype == "float":
                df[column] = pd.to_numeric(df[column], errors='coerce')
            elif dtype == "str":
                df[column] = df[column].astype(str)
            elif dtype == "datetime":
                df[column] = pd.to_datetime(df[column], errors='coerce')
            return df

        elif action.action_type == CleaningActionType.NORMALIZE:
            column = params.get("column")
            method = params.get("method", "minmax")
            
            if column not in df.columns:
                raise ValueError(f"Column {column} not found")
                
            if method == "minmax":
                min_val = df[column].min()
                max_val = df[column].max()
                if max_val != min_val:
                    df[column] = (df[column] - min_val) / (max_val - min_val)
            elif method == "zscore":
                mean_val = df[column].mean()
                std_val = df[column].std()
                if std_val != 0:
                    df[column] = (df[column] - mean_val) / std_val
            return df

        elif action.action_type == CleaningActionType.DROP_COLUMNS:
            columns = params.get("columns", [])
            existing_columns = [col for col in columns if col in df.columns]
            return df.drop(columns=existing_columns).reset_index(drop=True)

        elif action.action_type == CleaningActionType.FILTER_ROWS:
            column = params.get("column")
            operator = params.get("operator")
            value = params.get("value")
            
            if column not in df.columns:
                raise ValueError(f"Column {column} not found")
                
            if operator == ">":
                return df[df[column] > value].reset_index(drop=True)
            elif operator == "<":
                return df[df[column] < value].reset_index(drop=True)
            elif operator == "==":
                return df[df[column] == value].reset_index(drop=True)
            elif operator == ">=":
                return df[df[column] >= value].reset_index(drop=True)
            elif operator == "<=":
                return df[df[column] <= value].reset_index(drop=True)
            else:
                raise ValueError(f"Unknown operator: {operator}")

        else:
            raise ValueError(f"Unknown action type: {action.action_type}")

    def _get_dataset_info(self, df: pd.DataFrame) -> DatasetInfo:
        """Generate comprehensive dataset metadata and quality metrics."""
        return DatasetInfo(
            shape=[df.shape[0], df.shape[1]],
            columns=list(df.columns),
            null_counts=df.isnull().sum().to_dict(),
            null_percentages=(df.isnull().sum() / len(df) * 100).to_dict(),
            duplicate_count=df.duplicated().sum(),
            dtypes=df.dtypes.astype(str).to_dict(),
            numeric_columns=list(df.select_dtypes(include=[np.number]).columns),
            categorical_columns=list(df.select_dtypes(exclude=[np.number]).columns),
            quality_score=calculate_dataset_quality_score(df, self.current_task_id)
        )

    def _create_observation(
        self,
        message: str = "",
        reward: Optional[float] = None,
        done: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DataCleaningObservation:
        """Create a comprehensive observation."""
        dataset_info = self._get_dataset_info(self.df) if self.df is not None else DatasetInfo()
        
        all_actions = list(CleaningActionType)
        available_actions = [action for action in all_actions if action != CleaningActionType.SUBMIT]
        
        reward_breakdown = metadata.get("reward_breakdown") if metadata else None
        
        previous_quality = self.quality_history[-2] if len(self.quality_history) >= 2 else 0.0
        current_quality = self.quality_history[-1] if self.quality_history else 0.0
        
        return DataCleaningObservation(
            dataset_info=dataset_info,
            done=done,
            reward=reward,
            message=message,
            available_actions=available_actions,
            step_count=self.step_count,
            task_id=self.current_task_id,
            quality_score=current_quality,
            previous_quality=previous_quality,
            quality_improvement=current_quality - previous_quality,
            reward_breakdown=reward_breakdown,
            action_history=self.action_history.copy(),
            difficulty_level=DifficultyLevel(self.current_difficulty) if self.current_difficulty in DifficultyLevel.__members__ else DifficultyLevel.INTERMEDIATE,
            task_progress=self.step_count / self.config.max_steps_per_episode,
            metadata=metadata or {}
        )

    def _create_error_observation(self, error_message: str) -> DataCleaningObservation:
        """Create an error observation."""
        return DataCleaningObservation(
            done=False,
            reward=-0.1,
            message=f"Error: {error_message}",
            step_count=self.step_count,
            task_id=self.current_task_id,
            metadata={"error": error_message}
        )

    def _end_episode(self) -> DataCleaningObservation:
        """End the current episode and perform final grading."""
        self.episode_phase = EpisodePhase.GRADING
        
        # Calculate final grade
        final_score = grade_task_result(
            initial_df=self.initial_df,
            final_df=self.df,
            task_id=self.current_task_id,
            step_count=self.step_count
        )
        
        self.episode_phase = EpisodePhase.COMPLETION
        
        return DataCleaningObservation(
            done=True,
            reward=final_score,
            message=f"Episode completed. Final score: {final_score:.4f}",
            step_count=self.step_count,
            task_id=self.current_task_id,
            quality_score=calculate_dataset_quality_score(self.df, self.current_task_id),
            metadata={
                "episode_complete": True,
                "final_score": final_score,
            }
        )

    def _check_early_stopping(self) -> Optional[str]:
        """
        Check if episode should stop early based on performance conditions.

        Returns:
            str describing early stop reason, or None if should continue.
        """
        if not self.config.early_stopping_enabled:
            return None
            
        # Require minimum steps before early stopping
        if self.step_count < 3:
            return None

        # 1. No improvement after multiple steps
        if len(self.quality_history) >= 5:
            recent_quality = self.quality_history[-5:]
            if max(recent_quality) == min(recent_quality):
                return "no_improvement"
        
        # 2. Too many repeated actions
        if self.consecutive_repeated_actions >= 3:
            return "repeated_actions"
            
        # 3. Perfect quality achieved early
        current_quality = self.quality_history[-1] if self.quality_history else 0.0
        if current_quality >= 0.95:
            return "perfect_quality"

        return None