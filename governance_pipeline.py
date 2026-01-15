"""
Data Governance Pipeline
Orchestrates all governance checks before model training
"""

import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime

from pii_detector import PIIDetector
from data_minimisation import DataMinimisationAnalyzer
from deep_eda import DeepEDA
from data_auditor import DataAuditor
from fairlearn_metrics import FairlearnEvaluator


class DataGovernancePipeline:
    """
    Complete pre-training governance pipeline
    
    Steps:
    1. PII Detection
    2. Data Minimisation
    3. Deep EDA (5 analyses)
    4. Data Quality Auditing
    5. Representation Analysis
    6. Compliance Assessment
    """
    
    def __init__(self, domain: str):
        """
        Initialize pipeline
        
        Args:
            domain: One of 'credit_scoring', 'hiring', 'healthcare', 'education'
        """
        self.domain = domain
        self.results = {
            'pipeline_start': None,
            'pipeline_end': None,
            'domain': domain,
            'checks': {},
            'compliance': {},
            'status': 'NOT_RUN'
        }
    
    def run_governance_checks(
        self,
        df: pd.DataFrame,
        target_col: str,
        protected_attrs: Optional[list] = None
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Run all governance checks
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            protected_attrs: List of protected attributes (auto-detected if None)
            
        Returns:
            Tuple of (governance_report, cleaned_dataframe)
        """
        self.results['pipeline_start'] = datetime.now().isoformat()
        self.results['status'] = 'IN_PROGRESS'
        
        print("\n" + "="*80)
        print("DATA GOVERNANCE PIPELINE - PRE-TRAINING CHECKS")
        print("="*80)
        print(f"Domain: {self.domain}")
        print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Target: {target_col}")
        print("="*80)
        
        # Auto-detect protected attributes if not provided
        if protected_attrs is None:
            protected_attrs = self._get_protected_attrs_for_domain(self.domain)
        
        # Prepare metadata
        metadata = {
            'name': f'{self.domain}_dataset',
            'domain': self.domain,
            'target': target_col,
            'protected_attributes': protected_attrs
        }
        
        # STEP 1: PII DETECTION
        print("\n\n🔒 STEP 1/6: PII Detection")
        print("-" * 80)
        pii_detector = PIIDetector()
        pii_results = pii_detector.detect_pii(df, sample_size=100)
        self.results['checks']['pii_detection'] = {
            'pii_columns': list(pii_results.keys()),
            'pii_count': len(pii_results),
            'details': pii_results
        }
        
        # STEP 2: DATA MINIMISATION
        print("\n\n🗑️ STEP 2/6: Data Minimisation (GDPR Art. 5(1)(c))")
        print("-" * 80)
        minimiser = DataMinimisationAnalyzer(df, metadata)
        df_minimal, exclusion_log, importances = minimiser.run_full_analysis()
        
        self.results['checks']['data_minimisation'] = {
            'original_features': df.shape[1],
            'minimal_features': df_minimal.shape[1],
            'excluded_count': len(exclusion_log),
            'exclusion_log': exclusion_log.to_dict('records') if len(exclusion_log) > 0 else [],
            'report': minimiser.get_report()
        }
        
        # STEP 3: DEEP EDA
        print("\n\n🔬 STEP 3/6: Deep EDA (Bias Detection)")
        print("-" * 80)
        deep_eda = DeepEDA(df_minimal, metadata)
        eda_findings = deep_eda.run_full_analysis(target_col=target_col)
        
        self.results['checks']['deep_eda'] = deep_eda.get_summary()
        
        # STEP 4: DATA AUDITING
        print("\n\n✅ STEP 4/6: Data Quality Auditing")
        print("-" * 80)
        auditor = DataAuditor(df_minimal, metadata)
        audit_results = auditor.run_full_audit(
            reference_df=None,
            use_great_expectations=True
        )
        
        self.results['checks']['data_auditing'] = audit_results
        
        # STEP 5: REPRESENTATION ANALYSIS
        print("\n\n📊 STEP 5/6: Representation Analysis")
        print("-" * 80)
        rep_analysis = self._analyze_representation(df_minimal, protected_attrs, target_col)
        self.results['checks']['representation'] = rep_analysis
        
        # STEP 6: COMPLIANCE ASSESSMENT
        print("\n\n⚖️ STEP 6/6: Regulatory Compliance Assessment")
        print("-" * 80)
        compliance = self._assess_compliance(self.results['checks'])
        self.results['compliance'] = compliance
        
        # Finalize
        self.results['pipeline_end'] = datetime.now().isoformat()
        self.results['status'] = 'COMPLETED'
        
        self._print_summary()
        
        return self.results, df_minimal
    
    def _get_protected_attrs_for_domain(self, domain: str) -> list:
        """Get default protected attributes for each domain"""
        domain_map = {
            'credit_scoring': ['sex', 'race', 'age', 'marital-status'],
            'hiring': ['sex', 'race', 'age', 'disability'],
            'healthcare': ['sex', 'race', 'age', 'ethnicity'],
            'education': ['sex', 'race', 'age', 'disability']
        }
        return domain_map.get(domain, ['sex', 'race', 'age'])
    
    def _analyze_representation(
        self,
        df: pd.DataFrame,
        protected_attrs: list,
        target_col: str
    ) -> Dict:
        """Analyze representation across protected attributes"""
        rep_results = {}
        
        # Find matching columns
        for attr in protected_attrs:
            matching = [col for col in df.columns 
                       if attr.lower().replace('-', '_') in col.lower().replace('-', '_')]
            
            for col in matching:
                if col in df.columns:
                    # Distribution
                    dist = df[col].value_counts(normalize=True).to_dict()
                    
                    # Outcome rates by group
                    if target_col in df.columns:
                        outcome_by_group = df.groupby(col)[target_col].mean().to_dict()
                    else:
                        outcome_by_group = {}
                    
                    rep_results[col] = {
                        'distribution': dist,
                        'outcome_by_group': outcome_by_group,
                        'total_groups': len(dist),
                        'warnings': []
                    }
                    
                    # Check for under-representation
                    for group, pct in dist.items():
                        if pct < 0.05:
                            rep_results[col]['warnings'].append(
                                f'{group} is under-represented ({pct*100:.1f}%)'
                            )
                    
                    print(f"\n📊 {col} Distribution:")
                    for group, pct in dist.items():
                        print(f"   • {group}: {pct*100:.1f}%")
                    
                    if rep_results[col]['warnings']:
                        for warning in rep_results[col]['warnings']:
                            print(f"   ⚠️  {warning}")
        
        return rep_results
    
    def _assess_compliance(self, checks: Dict) -> Dict:
        """Assess regulatory compliance based on check results"""
        compliance = {
            'gdpr_article_5_1_c': None,  # Data minimisation
            'gdpr_article_5_1_d': None,  # Data accuracy
            'gdpr_article_9': None,      # Special categories
            'eu_ai_act_annex_iii': None, # High-risk AI
            'overall_status': None,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check GDPR Art. 5(1)(c) - Data Minimisation
        if checks['data_minimisation']['excluded_count'] > 0:
            compliance['gdpr_article_5_1_c'] = 'COMPLIANT'
        else:
            compliance['gdpr_article_5_1_c'] = 'WARNING'
            compliance['warnings'].append(
                'No features excluded during data minimisation'
            )
        
        # Check GDPR Art. 5(1)(d) - Data Accuracy
        audit_status = checks['data_auditing'].get('compliance_status', 'UNKNOWN')
        if audit_status == 'COMPLIANT':
            compliance['gdpr_article_5_1_d'] = 'COMPLIANT'
        else:
            compliance['gdpr_article_5_1_d'] = 'NON_COMPLIANT'
            compliance['issues'].append(
                'Data quality checks failed - dataset has accuracy issues'
            )
        
        # Check GDPR Art. 9 - Special Categories
        prohibited_count = checks['data_minimisation']['report']['prohibited_features_count']
        if prohibited_count > 0:
            compliance['gdpr_article_9'] = 'COMPLIANT'
            compliance['warnings'].append(
                f'{prohibited_count} protected attributes identified and handled appropriately'
            )
        else:
            compliance['gdpr_article_9'] = 'WARNING'
        
        # Check EU AI Act Annex III - High-risk AI Systems
        eda = checks['deep_eda']
        issues_found = (
            eda['underrepresented_groups_count'] +
            eda['proxy_variables_count'] +
            eda['statistical_disparities_count']
        )
        
        if issues_found == 0:
            compliance['eu_ai_act_annex_iii'] = 'COMPLIANT'
        elif issues_found < 5:
            compliance['eu_ai_act_annex_iii'] = 'WARNING'
            compliance['warnings'].append(
                f'{issues_found} potential bias issues detected'
            )
        else:
            compliance['eu_ai_act_annex_iii'] = 'NON_COMPLIANT'
            compliance['issues'].append(
                f'{issues_found} significant bias issues detected'
            )
        
        # Overall Status
        non_compliant = [k for k, v in compliance.items() 
                        if v == 'NON_COMPLIANT' and k != 'overall_status']
        
        if non_compliant:
            compliance['overall_status'] = 'NON_COMPLIANT'
        elif compliance['issues']:
            compliance['overall_status'] = 'NON_COMPLIANT'
        elif compliance['warnings']:
            compliance['overall_status'] = 'COMPLIANT_WITH_WARNINGS'
        else:
            compliance['overall_status'] = 'COMPLIANT'
        
        # Recommendations
        if eda['underrepresented_groups_count'] > 0:
            compliance['recommendations'].append(
                'Consider data augmentation or synthetic data for under-represented groups'
            )
        
        if eda['proxy_variables_count'] > 0:
            compliance['recommendations'].append(
                'Review and potentially exclude proxy variables'
            )
        
        if eda['statistical_disparities_count'] > 0:
            compliance['recommendations'].append(
                'Apply bias mitigation techniques before model training'
            )
        
        return compliance
    
    def _print_summary(self):
        """Print governance pipeline summary"""
        print("\n\n" + "="*80)
        print("GOVERNANCE PIPELINE SUMMARY")
        print("="*80)
        
        compliance = self.results['compliance']
        
        print(f"\n📋 Overall Status: {compliance['overall_status']}")
        
        print("\n✅ GDPR Compliance:")
        print(f"   • Art. 5(1)(c) Data Minimisation: {compliance['gdpr_article_5_1_c']}")
        print(f"   • Art. 5(1)(d) Data Accuracy: {compliance['gdpr_article_5_1_d']}")
        print(f"   • Art. 9 Special Categories: {compliance['gdpr_article_9']}")
        
        print(f"\n⚖️ EU AI Act Compliance:")
        print(f"   • Annex III High-risk AI: {compliance['eu_ai_act_annex_iii']}")
        
        if compliance['issues']:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in compliance['issues']:
                print(f"   • {issue}")
        
        if compliance['warnings']:
            print("\n⚠️ WARNINGS:")
            for warning in compliance['warnings']:
                print(f"   • {warning}")
        
        if compliance['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for rec in compliance['recommendations']:
                print(f"   • {rec}")
        
        print("\n" + "="*80)
        
        if compliance['overall_status'] == 'COMPLIANT':
            print("✅ DATASET IS COMPLIANT - Safe to proceed with model training")
        elif compliance['overall_status'] == 'COMPLIANT_WITH_WARNINGS':
            print("⚠️ DATASET IS COMPLIANT WITH WARNINGS - Review warnings before proceeding")
        else:
            print("🚫 DATASET IS NON-COMPLIANT - DO NOT proceed with model training")
            print("   Fix identified issues and re-run governance checks")
        
        print("="*80 + "\n")
    
    def export_report(self, filepath: str = 'governance_report.json'):
        """Export governance report to JSON"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Governance report exported to {filepath}")
    
    def get_results(self) -> Dict:
        """Get governance results"""
        return self.results
