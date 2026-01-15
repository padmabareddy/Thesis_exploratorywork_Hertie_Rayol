"""
Data Minimisation Analyzer
Implements GDPR Article 5(1)(c) - data minimisation principle
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


class DataMinimisationAnalyzer:
    """
    Analyze and enforce data minimisation principles (GDPR Art. 5(1)(c))
    
    Performs 3 key tasks:
    1. Classify features by necessity (Essential/Prohibited/Optional)
    2. Assess feature importance using ML
    3. Create minimal dataset with documented rationale
    """
    
    def __init__(self, df: pd.DataFrame, metadata: Dict):
        """
        Initialize analyzer
        
        Args:
            df: Input DataFrame
            metadata: Dictionary with keys: 'name', 'target', 'protected_attributes'
        """
        self.df = df.copy()
        self.metadata = metadata
        self.target_col = metadata.get('target')
        self.minimisation_report = {
            'essential_features': [],
            'optional_features': [],
            'prohibited_features': [],
            'recommended_exclusions': []
        }
    
    def classify_features(self, protected_attrs: Optional[List[str]] = None) -> Dict:
        """
        Classify features by necessity and compliance
        
        Args:
            protected_attrs: List of protected attributes (uses metadata if None)
            
        Returns:
            Minimisation report dictionary
        """
        print("\n" + "="*80)
        print("DATA MINIMISATION ANALYSIS")
        print("="*80)
        
        if protected_attrs is None:
            protected_attrs = self.metadata.get('protected_attributes', [])
        
        protected_normalized = [attr.lower().replace('-', '_').replace(' ', '_') 
                               for attr in protected_attrs]
        
        for col in self.df.columns:
            col_normalized = col.lower().replace('-', '_').replace(' ', '_')
            
            # Target variable is essential
            if col == self.target_col:
                self.minimisation_report['essential_features'].append({
                    'feature': col,
                    'rationale': 'Target variable - required for supervised learning'
                })
                continue
            
            # Protected attributes should generally be excluded
            if any(prot in col_normalized for prot in protected_normalized):
                self.minimisation_report['prohibited_features'].append({
                    'feature': col,
                    'rationale': 'Protected attribute - may violate anti-discrimination law',
                    'gdpr_ref': 'Art. 9 (Special categories), Art. 5(1)(c)',
                    'eu_ai_act_ref': 'Annex III - High-risk prohibited characteristics'
                })
            else:
                # All other features require justification
                self.minimisation_report['optional_features'].append({
                    'feature': col,
                    'rationale': 'Requires justification of necessity'
                })
        
        # Display classification
        print("\n✓ ESSENTIAL FEATURES (Must Keep):")
        for item in self.minimisation_report['essential_features']:
            print(f"   - {item['feature']}: {item['rationale']}")
        
        print("\n⚠️  PROHIBITED FEATURES (Should Exclude):")
        for item in self.minimisation_report['prohibited_features']:
            print(f"   - {item['feature']}: {item['rationale']}")
            print(f"     Regulatory: {item.get('gdpr_ref', 'N/A')}")
        
        print(f"\n❓ OPTIONAL FEATURES (Justify if Used): {len(self.minimisation_report['optional_features'])} features")
        
        return self.minimisation_report
    
    def assess_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Use ML to assess which features are actually necessary
        Uses Random Forest to compute feature importance scores
        
        Returns:
            DataFrame with features and their importance scores, or None if fails
        """
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ASSESSMENT")
        print("="*80)
        
        if self.target_col not in self.df.columns:
            print("⚠️  Target column not found, skipping importance assessment")
            return None
        
        try:
            # Prepare data
            df_ml = self.df.copy()
            
            # Encode categorical features
            for col in df_ml.select_dtypes(include=['object', 'category']).columns:
                if col != self.target_col:
                    le = LabelEncoder()
                    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            
            # Encode target
            if df_ml[self.target_col].dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(df_ml[self.target_col])
            else:
                y = df_ml[self.target_col].values
            
            # Prepare features
            X = df_ml.drop(self.target_col, axis=1)
            X = X.fillna(X.mean())
            
            # Train Random Forest
            print("\n📊 Training Random Forest to assess feature importance...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
            rf.fit(X, y)
            
            # Get importance scores
            importances = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n✓ Feature Importance Rankings (Top 10):")
            print(importances.head(10).to_string(index=False))
            
            # Identify low-importance features
            low_importance = importances[importances['importance'] < 0.01]['feature'].tolist()
            
            if low_importance:
                print(f"\n💡 RECOMMENDATION: Consider excluding {len(low_importance)} low-importance features:")
                for feat in low_importance[:5]:  # Show first 5
                    imp_val = importances[importances['feature']==feat]['importance'].values[0]
                    print(f"   - {feat} (importance: {imp_val:.4f})")
                
                if len(low_importance) > 5:
                    print(f"   ... and {len(low_importance) - 5} more")
                
                self.minimisation_report['recommended_exclusions'] = low_importance
                
                print("\n   Rationale: Features with negligible importance do not contribute to model")
                print("   Benefit: Reduces processing, improves interpretability, lowers privacy risk")
            else:
                print("\n✓ No low-importance features detected")
            
            return importances
            
        except Exception as e:
            print(f"\n⚠️  Error assessing feature importance: {e}")
            return None
    
    def create_minimal_dataset(self, 
                              exclude_protected: bool = True,
                              exclude_low_importance: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create minimized dataset with documented exclusions
        
        Args:
            exclude_protected: Whether to exclude protected attributes
            exclude_low_importance: Whether to exclude low-importance features
            
        Returns:
            Tuple of (minimal_dataframe, exclusion_log_dataframe)
        """
        print("\n" + "="*80)
        print("CREATING MINIMAL DATASET")
        print("="*80)
        
        df_minimal = self.df.copy()
        exclusion_log = []
        
        # Exclude protected attributes
        if exclude_protected:
            for item in self.minimisation_report['prohibited_features']:
                feat = item['feature']
                if feat in df_minimal.columns:
                    df_minimal = df_minimal.drop(feat, axis=1)
                    exclusion_log.append({
                        'feature': feat,
                        'reason': 'Protected attribute',
                        'regulation': item.get('gdpr_ref', 'GDPR Art. 9')
                    })
                    print(f"✓ Excluded: {feat} (protected attribute)")
        
        # Exclude low-importance features
        if exclude_low_importance and self.minimisation_report['recommended_exclusions']:
            for feat in self.minimisation_report['recommended_exclusions']:
                if feat in df_minimal.columns and feat != self.target_col:
                    df_minimal = df_minimal.drop(feat, axis=1)
                    exclusion_log.append({
                        'feature': feat,
                        'reason': 'Low importance (<0.01)',
                        'regulation': 'GDPR Art. 5(1)(c) - Data minimisation'
                    })
                    print(f"✓ Excluded: {feat} (low importance)")
        
        # Summary
        reduction = self.df.shape[1] - df_minimal.shape[1]
        pct = (reduction/self.df.shape[1]*100) if self.df.shape[1] > 0 else 0
        
        print(f"\n📊 Dataset Reduction:")
        print(f"   Original: {self.df.shape[1]} features")
        print(f"   Minimal: {df_minimal.shape[1]} features")
        print(f"   Reduction: {reduction} features removed ({pct:.1f}%)")
        
        exclusion_df = pd.DataFrame(exclusion_log)
        
        return df_minimal, exclusion_df
    
    def run_full_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Run complete data minimisation analysis
        
        Returns:
            Tuple of (minimal_dataframe, exclusion_log, feature_importance)
        """
        print("\n" + "#"*80)
        print("#" + " "*24 + "DATA MINIMISATION ANALYSIS" + " "*29 + "#")
        print("#"*80)
        
        self.classify_features()
        importances = self.assess_feature_importance()
        df_minimal, exclusion_log = self.create_minimal_dataset()
        
        print("\n" + "#"*80)
        print("#" + " "*23 + "MINIMISATION ANALYSIS COMPLETE" + " "*25 + "#")
        print("#"*80)
        
        return df_minimal, exclusion_log, importances
    
    def get_report(self) -> Dict:
        """Get a summary report of the minimisation analysis"""
        return {
            'essential_features_count': len(self.minimisation_report['essential_features']),
            'prohibited_features_count': len(self.minimisation_report['prohibited_features']),
            'optional_features_count': len(self.minimisation_report['optional_features']),
            'recommended_exclusions_count': len(self.minimisation_report['recommended_exclusions']),
            'report': self.minimisation_report
        }
