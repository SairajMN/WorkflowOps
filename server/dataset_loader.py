"""Dataset generator for AutoClean-Ai data cleaning environment.

Generates realistic datasets with common data quality issues for training
AI agents on data cleaning tasks.
"""

import pandas as pd
import numpy as np
import random
import string
from typing import Dict, Any, List
from faker import Faker

fake = Faker()


class DatasetGenerator:
    """
    Generates realistic datasets with controlled data quality issues.
    
    Supports 3 difficulty levels:
    - Basic: nulls and duplicates only
    - Intermediate: nulls, duplicates, outliers, invalid emails
    - Advanced: full range of data quality problems
    """
    
    def __init__(self):
        self.fake = Faker()
        self.seed = None
    
    def get_total_examples(self):
        """Required method for OpenEnv compatibility."""
        return 3
        
    def generate_dataset(self, task_id: str, seed: int = None) -> pd.DataFrame:
        """Generate dataset for specified task."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            Faker.seed(seed)
            self.seed = seed
            
        if task_id == "task_1_basic_cleaning":
            return self._generate_basic_dataset()
        elif task_id == "task_2_intermediate_cleaning":
            return self._generate_intermediate_dataset()
        elif task_id == "task_3_full_pipeline":
            return self._generate_advanced_dataset()
        else:
            # Default to basic
            return self._generate_basic_dataset()
            
    def _generate_basic_dataset(self) -> pd.DataFrame:
        """Generate basic dataset with nulls and duplicates."""
        n_rows = 100
        
        # Generate base data
        data = {
            'id': [i for i in range(n_rows)],
            'name': [self.fake.name() for _ in range(n_rows)],
            'age': np.random.randint(18, 75, size=n_rows),
            'email': [self.fake.email() for _ in range(n_rows)],
            'salary': np.random.randint(30000, 150000, size=n_rows)
        }
        
        df = pd.DataFrame(data)
        
        # Add null values (15% of rows)
        null_mask = np.random.choice([True, False], size=n_rows, p=[0.15, 0.85])
        df.loc[null_mask, 'age'] = np.nan
        df.loc[null_mask[:n_rows//2], 'salary'] = np.nan
        
        # Add duplicates (10% of rows)
        n_duplicates = int(n_rows * 0.10)
        duplicate_indices = np.random.choice(df.index, size=n_duplicates, replace=False)
        duplicates = df.loc[duplicate_indices].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
        
        # Shuffle
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return df
        
    def _generate_intermediate_dataset(self) -> pd.DataFrame:
        """Generate intermediate dataset with nulls, duplicates, outliers, invalid emails."""
        n_rows = 200
        
        # Generate base data
        data = {
            'id': [i for i in range(n_rows)],
            'name': [self.fake.name() for _ in range(n_rows)],
            'age': np.random.randint(18, 75, size=n_rows),
            'email': [self.fake.email() for _ in range(n_rows)],
            'salary': np.random.randint(30000, 150000, size=n_rows),
            'department': [self.fake.job() for _ in range(n_rows)]
        }
        
        df = pd.DataFrame(data)
        
        # Add null values (20% of rows)
        null_mask = np.random.choice([True, False], size=n_rows, p=[0.20, 0.80])
        df.loc[null_mask, 'age'] = np.nan
        df.loc[df.index[null_mask][:n_rows//2], 'salary'] = np.nan
        df.loc[df.index[null_mask][:n_rows//3], 'department'] = np.nan
        
        # Add duplicates (15% of rows)
        n_duplicates = int(n_rows * 0.15)
        duplicate_indices = np.random.choice(df.index, size=n_duplicates, replace=False)
        duplicates = df.loc[duplicate_indices].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
        
        # Add invalid emails (25% of emails)
        invalid_email_mask = np.random.choice([True, False], size=len(df), p=[0.25, 0.75])
        invalid_count = np.sum(invalid_email_mask)
        invalid_values = [
            [
                self.fake.user_name(),
                'not_an_email',
                'missing@domain',
                'user@.com',
                '@missinguser.com'
            ][np.random.randint(0, 5)] for _ in range(invalid_count)
        ]
        df.loc[invalid_email_mask, 'email'] = invalid_values
        
        # Add outliers in salary column
        outlier_indices = np.random.choice(df.index, size=int(len(df) * 0.10), replace=False)
        df.loc[outlier_indices, 'salary'] = np.random.randint(500000, 1000000, size=len(outlier_indices))
        
        # Shuffle
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return df
        
    def _generate_advanced_dataset(self) -> pd.DataFrame:
        """Generate advanced dataset with full range of data quality issues."""
        n_rows = 500
        
        # Generate base data
        data = {
            'id': [i for i in range(n_rows)],
            'name': [self.fake.name() for _ in range(n_rows)],
            'age': np.random.randint(18, 75, size=n_rows),
            'email': [self.fake.email() for _ in range(n_rows)],
            'salary': np.random.randint(30000, 150000, size=n_rows),
            'department': [self.fake.job() for _ in range(n_rows)],
            'join_date': [self.fake.date_between(start_date='-10y', end_date='today') for _ in range(n_rows)],
            'performance_score': np.random.uniform(0.0, 10.0, size=n_rows)
        }
        
        df = pd.DataFrame(data)
        
        # Convert join_date to string with inconsistent formats
        date_mask = np.random.choice([True, False], size=n_rows, p=[0.3, 0.7])
        df.loc[date_mask, 'join_date'] = df.loc[date_mask, 'join_date'].astype(str)
        
        # Add null values (25% of rows)
        null_mask = np.random.choice([True, False], size=n_rows, p=[0.25, 0.75])
        df.loc[null_mask, 'age'] = np.nan
        df.loc[null_mask[:int(n_rows*0.8)], 'salary'] = np.nan
        df.loc[null_mask[:int(n_rows*0.6)], 'department'] = np.nan
        df.loc[null_mask[:int(n_rows*0.4)], 'performance_score'] = np.nan
        
        # Add duplicates (20% of rows)
        n_duplicates = int(n_rows * 0.20)
        duplicate_indices = np.random.choice(df.index, size=n_duplicates, replace=False)
        duplicates = df.loc[duplicate_indices].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
        
        # Add invalid emails (30% of emails)
        invalid_email_mask = np.random.choice([True, False], size=len(df), p=[0.30, 0.70])
        invalid_formats = [
            self.fake.user_name(),
            'not_an_email',
            'missing@domain',
            'user@.com',
            '@missinguser.com',
            'user@missing..com',
            'user name@domain.com'
        ]
        df.loc[invalid_email_mask, 'email'] = [random.choice(invalid_formats) for _ in range(sum(invalid_email_mask))]
        
        # Add outliers
        outlier_indices = np.random.choice(df.index, size=int(len(df) * 0.15), replace=False)
        df.loc[outlier_indices, 'salary'] = np.random.randint(500000, 2000000, size=len(outlier_indices))
        
        outlier_perf_indices = np.random.choice(df.index, size=int(len(df) * 0.10), replace=False)
        df.loc[outlier_perf_indices, 'performance_score'] = np.random.uniform(15.0, 100.0, size=len(outlier_perf_indices))
        
        # Add inconsistent data types
        type_mask = np.random.choice([True, False], size=len(df), p=[0.10, 0.90])
        df.loc[type_mask, 'age'] = df.loc[type_mask, 'age'].astype(str) + ' years'
        
        # Shuffle
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return df