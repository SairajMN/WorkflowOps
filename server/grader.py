"""Grading system for AutoClean-Ai data cleaning environment.

Implements deterministic reward calculation, quality scoring, and task grading.
All scores are normalized between 0.0 and 1.0.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import re

from models import CleaningActionType, DataCleaningAction


def calculate_dataset_quality_score(df: pd.DataFrame, task_id: str) -> float:
    """
    Calculate overall dataset quality score from 0.0 (worst) to 1.0 (perfect).
    
    Combines multiple quality metrics:
    - Null value percentage
    - Duplicate row percentage
    - Outlier percentage
    - Email validity (if applicable)
    - Data type correctness
    """
    if df.empty:
        return 0.0
        
    total_rows = len(df)
    quality_components = []
    
    # 1. Null value handling (0.0 = all nulls, 1.0 = no nulls)
    null_percentage = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    null_score = 1.0 - null_percentage
    quality_components.append(null_score)
    
    # 2. Duplicate handling (0.0 = all duplicates, 1.0 = no duplicates)
    duplicate_count = df.duplicated().sum()
    duplicate_percentage = duplicate_count / total_rows if total_rows > 0 else 0.0
    duplicate_score = 1.0 - duplicate_percentage
    quality_components.append(duplicate_score)
    
    # 3. Outlier detection for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outlier_scores = []
    
    for col in numeric_columns:
        if len(df[col].dropna()) < 4:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            continue
            
        outlier_mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        outlier_percentage = outlier_mask.sum() / len(df[col])
        outlier_scores.append(1.0 - outlier_percentage)
    
    if outlier_scores:
        outlier_score = sum(outlier_scores) / len(outlier_scores)
        quality_components.append(outlier_score)
    
    # 4. Email validation if email column exists
    if 'email' in df.columns:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        valid_emails = df['email'].astype(str).str.match(email_pattern, na=False).sum()
        email_validity = valid_emails / len(df['email']) if len(df['email']) > 0 else 0.0
        quality_components.append(email_validity)
    
    # 5. Task specific quality checks
    if task_id == "task_1_basic_cleaning":
        # Basic task only requires null and duplicate handling
        weights = [0.5, 0.5]
        final_score = (null_score * weights[0] + duplicate_score * weights[1])
    elif task_id == "task_2_intermediate_cleaning":
        # Intermediate adds outliers and email validation
        components = [null_score, duplicate_score]
        if outlier_scores:
            components.append(outlier_score)
        if 'email' in df.columns:
            components.append(email_validity)
        final_score = sum(components) / len(components)
    elif task_id == "task_3_full_pipeline":
        # Advanced task uses all components
        final_score = sum(quality_components) / len(quality_components) if quality_components else 0.0
    else:
        # Default average
        final_score = sum(quality_components) / len(quality_components) if quality_components else 0.0
    
    return max(0.0, min(1.0, final_score))


def calculate_reward(
    df: pd.DataFrame,
    initial_df: pd.DataFrame,
    previous_quality: float,
    current_quality: float,
    action: DataCleaningAction,
    task_id: str,
    step_count: int
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate reward for an action. Returns (reward, reward_info).
    
    Reward components:
    - Quality improvement delta
    - Action validity bonus
    - Efficiency bonus
    - Progress bonus
    - Penalties for invalid/ineffective actions
    """
    reward_info = {
        "quality_improvement": 0.0,
        "action_validity": 0.0,
        "efficiency_bonus": 0.0,
        "progress_bonus": 0.0,
        "penalty": 0.0,
        "total": 0.0
    }
    
    # Base reward is quality improvement
    quality_delta = current_quality - previous_quality
    reward = quality_delta * 2.0  # Amplify delta for stronger signal
    reward_info["quality_improvement"] = quality_delta
    
    # Bonus for valid action execution
    reward += 0.05
    reward_info["action_validity"] = 0.05
    
    # Bonus for progress
    if quality_delta > 0:
        progress_bonus = 0.03
        reward += progress_bonus
        reward_info["progress_bonus"] = progress_bonus
    
    # Efficiency bonus (earlier steps give higher reward)
    if step_count < 5 and quality_delta > 0:
        efficiency_bonus = 0.02 * (5 - step_count)
        reward += efficiency_bonus
        reward_info["efficiency_bonus"] = efficiency_bonus
    
    # Penalties
    if quality_delta < -0.01:
        # Negative quality change penalty
        penalty = abs(quality_delta) * 0.5
        reward -= penalty
        reward_info["penalty"] += penalty
    
    # Penalty for dropping too many rows
    if len(df) < len(initial_df) * 0.5:
        row_loss_penalty = 0.1
        reward -= row_loss_penalty
        reward_info["penalty"] += row_loss_penalty
    
    # Clamp final reward between -0.2 and 0.5 per step
    final_reward = max(-0.2, min(0.5, reward))
    reward_info["total"] = final_reward
    
    return final_reward, reward_info


def grade_task_result(
    initial_df: pd.DataFrame,
    final_df: pd.DataFrame,
    task_id: str,
    step_count: int
) -> float:
    """
    Grade final task result. Returns score from 0.0 to 1.0.
    
    Grading criteria per task:
    - task_1_basic_cleaning: 40% null handling, 40% duplicate handling, 20% efficiency
    - task_2_intermediate_cleaning: 25% null, 30% email, 25% outliers, 20% efficiency
    - task_3_full_pipeline: Full weighted criteria
    """
    if final_df.empty:
        return 0.0
        
    final_quality = calculate_dataset_quality_score(final_df, task_id)
    initial_quality = calculate_dataset_quality_score(initial_df, task_id)
    
    quality_improvement = final_quality - initial_quality
    
    # Calculate efficiency score (better score for fewer steps)
    max_steps = 15
    efficiency_score = max(0.0, 1.0 - (step_count / max_steps))
    
    # Task specific grading
    if task_id == "task_1_basic_cleaning":
        # Basic: 40% null, 40% duplicates, 20% efficiency
        null_percentage = final_df.isnull().sum().sum() / (final_df.shape[0] * final_df.shape[1])
        null_score = 1.0 - null_percentage
        
        duplicate_percentage = final_df.duplicated().sum() / len(final_df)
        duplicate_score = 1.0 - duplicate_percentage
        
        score = (null_score * 0.4) + (duplicate_score * 0.4) + (efficiency_score * 0.2)
        
    elif task_id == "task_2_intermediate_cleaning":
        # Intermediate: 25% null, 30% email, 25% outliers, 20% efficiency
        null_percentage = final_df.isnull().sum().sum() / (final_df.shape[0] * final_df.shape[1])
        null_score = 1.0 - null_percentage
        
        email_score = 0.5
        if 'email' in final_df.columns:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            valid_emails = final_df['email'].astype(str).str.match(email_pattern, na=False).sum()
            email_score = valid_emails / len(final_df['email'])
        
        outlier_score = 0.5
        numeric_columns = final_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if len(final_df[col].dropna()) >= 4:
                Q1 = final_df[col].quantile(0.25)
                Q3 = final_df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    outlier_mask = (final_df[col] < (Q1 - 1.5 * IQR)) | (final_df[col] > (Q3 + 1.5 * IQR))
                    outlier_percentage = outlier_mask.sum() / len(final_df[col])
                    outlier_score = 1.0 - outlier_percentage
                    break
        
        score = (null_score * 0.25) + (email_score * 0.30) + (outlier_score * 0.25) + (efficiency_score * 0.20)
        
    elif task_id == "task_3_full_pipeline":
        # Advanced: quality * 0.8 + efficiency * 0.2
        score = (final_quality * 0.8) + (efficiency_score * 0.2)
        
    else:
        # Default grading
        score = final_quality
    
    # Ensure score is in valid range
    final_score = max(0.0, min(1.0, score))
    
    return final_score