"""Server module for DataQualityGuard-Env."""

import sys
import os

# Add server directory to path for relative imports
_server_dir = os.path.dirname(os.path.abspath(__file__))
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

# Now import from same directory (works for both local and HF Spaces)
from environment import DataCleaningEnvironment
from grader import (
    calculate_reward,
    calculate_dataset_quality_score,
    grade_task_result,
)
from dataset_loader import DatasetGenerator

__all__ = [
    "DataCleaningEnvironment",
    "calculate_reward",
    "calculate_dataset_quality_score",
    "grade_task_result",
    "DatasetGenerator",
]
