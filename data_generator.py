"""
Synthetic Data Generator
Generate controlled datasets with known bias patterns for testing
"""

import pandas as pd
import numpy as np
from typing import Optional, List


class SyntheticDataGenerator:
    """
    Generate synthetic datasets with controlled bias patterns
    
    Supports multiple bias types:
    - 'none': No intentional bias (fair baseline)
    - 'gender': Gender-based bias
    - 'age': Age-based bias
    - 'intersectional': Multiple attributes interact
    """
    
    def __init__(self, domain: str):
        """
        Initialize generator for a specific domain
        
        Args:
            domain: One of 'credit_scoring', 'hiring', 'healthcare', etc.
        """
        self.domain = domain
    
    def generate_credit_scoring_data(self, 
                                    n_samples: int = 1000,
                                    bias_type: str = 'none',
                                    random_seed: int = 42) -> pd.DataFrame:
        """
        Generate synthetic credit scoring data
        
        Args:
            n_samples: Number of samples to generate
            bias_type: Type of bias ('none', 'gender', 'age', 'intersectional')
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame with synthetic credit scoring data
        """
        print(f"\nGenerating {n_samples} synthetic credit scoring records...")
        print(f"Bias pattern: {bias_type}")
        
        np.random.seed(random_seed)
        
        # Generate base features
        data = {
            'age': np.random.randint(18, 70, n_samples),
            'sex': np.random.choice(['Male', 'Female'], n_samples),
            'income': np.random.lognormal(10.5, 0.5, n_samples).astype(int),
            'debt_ratio': np.random.uniform(0, 1, n_samples),
            'credit_history_length': np.random.randint(0, 30, n_samples),
            'num_credit_lines': np.random.randint(0, 10, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate approval score based on legitimate factors
        base_score = (
            0.3 * (df['income'] / df['income'].max()) +
            0.3 * (1 - df['debt_ratio']) +
            0.2 * (df['credit_history_length'] / 30) +
            0.2 * (df['num_credit_lines'] / 10)
        )
        
        # Apply bias patterns
        if bias_type == 'gender':
            # Males get 20% boost in approval score
            bias_factor = np.where(df['sex'] == 'Male', 1.2, 1.0)
            base_score = base_score * bias_factor
            print("   ✓ Applied gender bias: Males receive 20% higher scores")
        
        elif bias_type == 'age':
            # Younger applicants (< 30) penalized
            bias_factor = np.where(df['age'] < 30, 0.8, 1.0)
            base_score = base_score * bias_factor
            print("   ✓ Applied age bias: <30 years old penalized by 20%")
        
        elif bias_type == 'intersectional':
            # Young females most disadvantaged
            bias_factor = np.where(
                (df['sex'] == 'Female') & (df['age'] < 30),
                0.7,  # 30% penalty
                np.where(df['sex'] == 'Female', 0.85,  # 15% penalty
                        np.where(df['age'] < 30, 0.85, 1.0))  # 15% penalty
            )
            base_score = base_score * bias_factor
            print("   ✓ Applied intersectional bias: Young females most disadvantaged")
        
        # Convert to binary approval decision
        threshold = base_score.median()
        df['approved'] = (base_score > threshold).astype(int)
        
        # Add some noise (5% random flips)
        noise_indices = np.random.choice(len(df), int(0.05 * len(df)), replace=False)
        df.loc[noise_indices, 'approved'] = 1 - df.loc[noise_indices, 'approved']
        
        # Print summary
        print(f"\n✓ Generated {len(df)} samples")
        print(f"   Overall approval rate: {df['approved'].mean():.2%}")
        print(f"\n   Approval by gender:")
        for sex, rate in df.groupby('sex')['approved'].mean().items():
            print(f"      {sex}: {rate:.2%}")
        
        return df
    
    def generate_hiring_data(self,
                           n_samples: int = 1000,
                           bias_type: str = 'none',
                           random_seed: int = 42) -> pd.DataFrame:
        """
        Generate synthetic hiring data
        
        Args:
            n_samples: Number of samples to generate
            bias_type: Type of bias ('none', 'gender', 'age', 'intersectional')
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame with synthetic hiring data
        """
        print(f"\nGenerating {n_samples} synthetic hiring records...")
        print(f"Bias pattern: {bias_type}")
        
        np.random.seed(random_seed)
        
        # Generate base features
        data = {
            'age': np.random.randint(22, 65, n_samples),
            'sex': np.random.choice(['Male', 'Female'], n_samples),
            'education_level': np.random.choice(['HS', 'Bachelors', 'Masters', 'PhD'], 
                                               n_samples, p=[0.2, 0.5, 0.25, 0.05]),
            'years_experience': np.random.randint(0, 30, n_samples),
            'interview_score': np.random.uniform(0, 100, n_samples),
            'technical_test_score': np.random.uniform(0, 100, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Encode education
        edu_mapping = {'HS': 1, 'Bachelors': 2, 'Masters': 3, 'PhD': 4}
        df['education_encoded'] = df['education_level'].map(edu_mapping)
        
        # Generate hiring decision score
        base_score = (
            0.3 * (df['interview_score'] / 100) +
            0.3 * (df['technical_test_score'] / 100) +
            0.2 * (df['years_experience'] / 30) +
            0.2 * (df['education_encoded'] / 4)
        )
        
        # Apply bias
        if bias_type == 'gender':
            bias_factor = np.where(df['sex'] == 'Male', 1.15, 1.0)
            base_score = base_score * bias_factor
            print("   ✓ Applied gender bias: Males favored by 15%")
        
        elif bias_type == 'age':
            # Older candidates (>50) penalized
            bias_factor = np.where(df['age'] > 50, 0.85, 1.0)
            base_score = base_score * bias_factor
            print("   ✓ Applied age bias: >50 years old penalized by 15%")
        
        # Convert to binary hiring decision (hire top 30%)
        threshold = base_score.quantile(0.7)
        df['hired'] = (base_score > threshold).astype(int)
        
        # Remove temporary column
        df = df.drop('education_encoded', axis=1)
        
        print(f"\n✓ Generated {len(df)} samples")
        print(f"   Overall hiring rate: {df['hired'].mean():.2%}")
        print(f"\n   Hiring by gender:")
        for sex, rate in df.groupby('sex')['hired'].mean().items():
            print(f"      {sex}: {rate:.2%}")
        
        return df
    
    def generate_controlled_dataset(self,
                                   n_samples: int = 1000,
                                   include_features: Optional[List[str]] = None,
                                   exclude_features: Optional[List[str]] = None,
                                   bias_type: str = 'none',
                                   random_seed: int = 42) -> pd.DataFrame:
        """
        Generate dataset with explicit feature inclusion/exclusion
        Useful for testing data minimisation principles
        
        Args:
            n_samples: Number of samples to generate
            include_features: List of features to include (None = all)
            exclude_features: List of features to exclude
            bias_type: Type of bias to introduce
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame with controlled features
        """
        print("\n" + "="*80)
        print("GENERATING CONTROLLED DATASET")
        print("="*80)
        
        # Generate based on domain
        if self.domain == 'credit_scoring':
            df = self.generate_credit_scoring_data(n_samples, bias_type, random_seed)
        elif self.domain == 'hiring':
            df = self.generate_hiring_data(n_samples, bias_type, random_seed)
        else:
            raise ValueError(f"Domain '{self.domain}' not yet supported for synthetic generation")
        
        # Feature selection with rationale
        if exclude_features:
            print(f"\n🔒 Excluding features (Data Minimisation):")
            for feat in exclude_features:
                if feat in df.columns:
                    print(f"   - {feat}: Excluded per GDPR Art. 5(1)(c)")
                    df = df.drop(feat, axis=1)
        
        if include_features:
            print(f"\n✓ Included features:")
            # Keep only specified features
            available_features = [f for f in include_features if f in df.columns]
            df = df[available_features]
            for feat in available_features:
                print(f"   - {feat}")
        
        print(f"\n✓ Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        
        return df
    
    def generate_comparison_datasets(self,
                                    n_samples: int = 1000,
                                    random_seed: int = 42) -> dict:
        """
        Generate multiple datasets for comparison (no bias, with bias)
        
        Args:
            n_samples: Number of samples per dataset
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary of datasets: {'no_bias': df1, 'gender_bias': df2, ...}
        """
        print("\n" + "="*80)
        print("GENERATING COMPARISON DATASETS")
        print("="*80)
        
        datasets = {}
        
        # No bias baseline
        print("\n1. Generating NO BIAS baseline:")
        datasets['no_bias'] = self.generate_credit_scoring_data(n_samples, 'none', random_seed)
        
        # Gender bias
        print("\n2. Generating GENDER BIAS dataset:")
        datasets['gender_bias'] = self.generate_credit_scoring_data(n_samples, 'gender', random_seed)
        
        # Age bias
        print("\n3. Generating AGE BIAS dataset:")
        datasets['age_bias'] = self.generate_credit_scoring_data(n_samples, 'age', random_seed)
        
        print("\n" + "="*80)
        print("✓ Generated 3 datasets for comparison")
        print("="*80)
        
        return datasets
