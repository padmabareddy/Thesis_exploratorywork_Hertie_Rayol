"""
Deep Exploratory Data Analysis (EDA)
Goes beyond standard EDA to uncover bias blind spots and fairness issues
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
from sklearn.preprocessing import LabelEncoder


class DeepEDA:
    """
    Deep Exploratory Data Analysis for bias detection and fairness assessment
    
    Performs 5 key analyses:
    1. Missing sensitive attributes (fairness blind spots)
    2. Under-represented groups (< 5% of dataset)
    3. Missingness patterns as bias (MCAR/MAR/MNAR)
    4. Proxy variables (features correlated with protected attributes)
    5. Statistical disparities in outcomes across groups
    """
    
    def __init__(self, df: pd.DataFrame, metadata: Dict):
        """
        Initialize Deep EDA analyzer
        
        Args:
            df: Input DataFrame
            metadata: Dictionary with keys: 'name', 'domain', 'target', 'protected_attributes'
        """
        self.df = df.copy()
        self.metadata = metadata
        self.findings = {
            'missing_sensitive_attrs': [],
            'underrepresented_groups': [],
            'missingness_bias': {},
            'proxy_variables': [],
            'statistical_disparities': {}
        }
    
    def analyze_missing_sensitive_attributes(self) -> List[str]:
        """
        Identify protected attributes that SHOULD be present but are missing
        Missing attributes = fairness blind spots
        
        Returns:
            List of missing protected attribute names
        """
        print("\n" + "="*80)
        print("ANALYSIS 1: Missing Sensitive Attributes (Fairness Blind Spots)")
        print("="*80)
        
        expected_attrs = set(self.metadata.get('protected_attributes', []))
        actual_attrs = set(col.lower().replace('-', '_').replace(' ', '_') 
                          for col in self.df.columns)
        expected_normalized = set(attr.lower().replace('-', '_').replace(' ', '_') 
                                 for attr in expected_attrs)
        
        missing = expected_normalized - actual_attrs
        
        if missing:
            print("\n⚠️  WARNING: The following protected attributes are MISSING:")
            for attr in missing:
                print(f"   - {attr}")
                self.findings['missing_sensitive_attrs'].append(attr)
            
            print("\n💡 IMPLICATIONS:")
            print("   • Cannot assess fairness for these groups")
            print("   • May violate EU AI Act transparency requirements")
            print("   • Risk of undetected discrimination")
        else:
            print("✓ All expected protected attributes present")
        
        return list(missing)
    
    def analyze_representation(self, min_group_size_pct: float = 5.0) -> List[Dict]:
        """
        Identify under-represented groups with insufficient sample sizes
        Small groups = less statistical power, more variance in predictions
        
        Args:
            min_group_size_pct: Threshold percentage for under-representation
            
        Returns:
            List of dictionaries with under-represented group information
        """
        print("\n" + "="*80)
        print("ANALYSIS 2: Under-Represented Groups")
        print("="*80)
        
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            value_counts = self.df[col].value_counts()
            total = len(self.df)
            percentages = (value_counts / total * 100).round(2)
            
            underrep = percentages[percentages < min_group_size_pct]
            
            if len(underrep) > 0:
                print(f"\n⚠️  Column: {col}")
                print(f"   Under-represented groups (< {min_group_size_pct}%):")
                for group, pct in underrep.items():
                    count = value_counts[group]
                    print(f"      • {group}: {count} samples ({pct}%)")
                    self.findings['underrepresented_groups'].append({
                        'column': col,
                        'group': group,
                        'count': int(count),
                        'percentage': float(pct)
                    })
                
                print("\n   💡 IMPLICATIONS:")
                print("      • Model may not learn patterns for these groups")
                print("      • Higher variance in predictions")
                print("      • Consider: data augmentation, synthetic data, or resampling")
        
        if not self.findings['underrepresented_groups']:
            print("✓ No significantly under-represented groups detected")
        
        return self.findings['underrepresented_groups']
    
    def analyze_missingness_patterns(self, target_col: Optional[str] = None) -> Dict:
        """
        Examine how missingness itself may encode bias
        Tests if missing data correlates with target (MNAR - Missing Not At Random)
        
        Args:
            target_col: Target column name to test correlation with
            
        Returns:
            Dictionary mapping columns to their missingness statistics
        """
        print("\n" + "="*80)
        print("ANALYSIS 3: Missingness as Bias (MCAR, MAR, MNAR Analysis)")
        print("="*80)
        
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df) * 100).round(2)
        missing_summary = pd.DataFrame({
            'Missing_Count': missing_counts,
            'Missing_Percent': missing_pct
        })
        missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values(
            'Missing_Percent', ascending=False
        )
        
        if len(missing_summary) > 0:
            print("\nColumns with missing data:")
            print(missing_summary)
            
            if target_col and target_col in self.df.columns:
                print(f"\n📊 Testing if missingness correlates with target '{target_col}':")
                
                for col in missing_summary.index:
                    if col != target_col:
                        missing_indicator = self.df[col].isnull().astype(int)
                        
                        # Chi-square test for categorical target
                        if self.df[target_col].dtype == 'object' or self.df[target_col].nunique() < 10:
                            try:
                                contingency = pd.crosstab(missing_indicator, self.df[target_col])
                                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                                
                                if p_value < 0.05:
                                    print(f"\n   ⚠️  {col}: Missingness SIGNIFICANTLY correlated with {target_col}")
                                    print(f"      χ² = {chi2:.2f}, p-value = {p_value:.4f}")
                                    print("      → Missingness is NOT random (MNAR)")
                                    print("      → May encode protected attribute information")
                                    
                                    self.findings['missingness_bias'][col] = {
                                        'chi2': float(chi2),
                                        'p_value': float(p_value),
                                        'mechanism': 'MNAR'
                                    }
                            except:
                                pass  # Skip if chi-square test fails
        else:
            print("✓ No missing data found")
        
        return self.findings['missingness_bias']
    
    def detect_proxy_variables(self, 
                              protected_attrs: Optional[List[str]] = None,
                              correlation_threshold: float = 0.7) -> List[Dict]:
        """
        Detect proxy variables highly correlated with protected attributes
        Proxies may violate data minimisation principles (GDPR Art. 5(1)(c))
        
        Args:
            protected_attrs: List of protected attribute names (uses metadata if None)
            correlation_threshold: Minimum correlation to flag as proxy
            
        Returns:
            List of dictionaries with proxy variable information
        """
        print("\n" + "="*80)
        print("ANALYSIS 4: Proxy Variables (Data Minimisation Violations)")
        print("="*80)
        
        if protected_attrs is None:
            protected_attrs = self.metadata.get('protected_attributes', [])
        
        # Find matching columns
        protected_cols = []
        for attr in protected_attrs:
            matching = [col for col in self.df.columns 
                       if attr.lower() in col.lower().replace('-', '_').replace(' ', '_')]
            protected_cols.extend(matching)
        
        if not protected_cols:
            print("⚠️  No protected attributes found in dataset to check for proxies")
            return []
        
        # Encode categorical variables for correlation analysis
        df_encoded = self.df.copy()
        for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        proxies_found = []
        
        for protected_col in protected_cols:
            if protected_col not in df_encoded.columns:
                continue
            
            try:
                correlations = df_encoded.corr()[protected_col].abs().sort_values(ascending=False)
                correlations = correlations[correlations.index != protected_col]
                
                high_corr = correlations[correlations > correlation_threshold]
                
                if len(high_corr) > 0:
                    print(f"\n⚠️  Potential proxies for '{protected_col}':")
                    for var, corr in high_corr.items():
                        print(f"   • {var}: correlation = {corr:.3f}")
                        proxy_info = {
                            'protected_attribute': protected_col,
                            'proxy_variable': var,
                            'correlation': float(corr)
                        }
                        proxies_found.append(proxy_info)
                        self.findings['proxy_variables'].append(proxy_info)
                    
                    print("\n   💡 DATA MINIMISATION IMPLICATIONS:")
                    print("      • These features may reveal protected attribute information")
                    print("      • Consider excluding to comply with GDPR Article 5(1)(c)")
                    print("      • Document rationale if retaining these features")
            except:
                pass  # Skip if correlation computation fails
        
        if not proxies_found:
            print("✓ No high-correlation proxy variables detected")
        
        return proxies_found
    
    def analyze_statistical_disparities(self, 
                                       target_col: str,
                                       protected_attrs: Optional[List[str]] = None) -> Dict:
        """
        Statistical tests for disparities in outcomes across protected groups
        Uses chi-square test to determine if outcomes are independent of protected attributes
        
        Args:
            target_col: Target variable to test for disparities
            protected_attrs: List of protected attributes (uses metadata if None)
            
        Returns:
            Dictionary mapping columns to their disparity statistics
        """
        print("\n" + "="*80)
        print("ANALYSIS 5: Statistical Disparities in Outcomes")
        print("="*80)
        
        if target_col not in self.df.columns:
            print(f"⚠️  Target column '{target_col}' not found")
            return {}
        
        if protected_attrs is None:
            protected_attrs = self.metadata.get('protected_attributes', [])
        
        for attr in protected_attrs:
            matching_cols = [col for col in self.df.columns 
                           if attr.lower() in col.lower().replace('-', '_').replace(' ', '_')]
            
            for col in matching_cols:
                if col not in self.df.columns or self.df[col].nunique() > 20:
                    continue
                
                try:
                    print(f"\n📊 Disparity analysis for '{col}':")
                    
                    # Cross-tabulation
                    crosstab = pd.crosstab(self.df[col], self.df[target_col], normalize='index')
                    print(crosstab.round(3))
                    
                    # Chi-square test
                    contingency = pd.crosstab(self.df[col], self.df[target_col])
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                    
                    print(f"\n   χ² test: χ² = {chi2:.2f}, p-value = {p_value:.4f}")
                    
                    if p_value < 0.05:
                        print("   ⚠️  SIGNIFICANT disparity detected (p < 0.05)")
                        print("   → Outcomes are NOT independent of this protected attribute")
                        print("   → May indicate systemic bias requiring intervention")
                        
                        self.findings['statistical_disparities'][col] = {
                            'chi2': float(chi2),
                            'p_value': float(p_value),
                            'significant': True
                        }
                    else:
                        print("   ✓ No significant disparity")
                except:
                    pass  # Skip if statistical test fails
        
        return self.findings['statistical_disparities']
    
    def run_full_analysis(self, target_col: Optional[str] = None) -> Dict:
        """
        Run all deep EDA analyses
        
        Args:
            target_col: Target column for missingness and disparity analyses
            
        Returns:
            Dictionary containing all findings
        """
        print("\n" + "#"*80)
        print("#" + " "*28 + "DEEP EDA ANALYSIS" + " "*34 + "#")
        print("#"*80)
        
        self.analyze_missing_sensitive_attributes()
        self.analyze_representation()
        self.analyze_missingness_patterns(target_col)
        self.detect_proxy_variables()
        if target_col:
            self.analyze_statistical_disparities(target_col)
        
        print("\n" + "#"*80)
        print("#" + " "*24 + "DEEP EDA ANALYSIS COMPLETE" + " "*29 + "#")
        print("#"*80)
        
        return self.findings
    
    def get_summary(self) -> Dict:
        """Get a summary of all findings"""
        return {
            'missing_sensitive_attrs_count': len(self.findings['missing_sensitive_attrs']),
            'underrepresented_groups_count': len(self.findings['underrepresented_groups']),
            'missingness_bias_count': len(self.findings['missingness_bias']),
            'proxy_variables_count': len(self.findings['proxy_variables']),
            'statistical_disparities_count': len(self.findings['statistical_disparities']),
            'findings': self.findings
        }
