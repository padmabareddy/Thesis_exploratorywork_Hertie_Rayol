"""
Fairness Metrics with Fairlearn
Compute comprehensive fairness metrics including demographic parity, equalized odds, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    selection_rate
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class FairlearnEvaluator:
    """Comprehensive fairness evaluation using Fairlearn"""
    
    def __init__(self):
        self.fairness_results = {}
    
    def compute_metrics(self, y_true, y_pred, sensitive_features, metric_names=None):
        """Compute fairness metrics"""
        print("\n" + "="*80)
        print("FAIRNESS METRICS WITH FAIRLEARN")
        print("="*80)
        
        if metric_names is None:
            metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'selection_rate']
        
        metrics = {
            'accuracy': accuracy_score,
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
            'f1_score': lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
            'selection_rate': selection_rate
        }
        
        metrics_to_compute = {k: v for k, v in metrics.items() if k in metric_names}
        
        print(f"\n[Computing {len(metrics_to_compute)} metrics across {len(sensitive_features.columns)} protected attributes]")
        
        results = {}
        
        for col in sensitive_features.columns:
            print(f"\n📊 Analyzing: {col}")
            
            mf = MetricFrame(
                metrics=metrics_to_compute,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features[col]
            )
            
            results[col] = {
                'by_group': mf.by_group.to_dict(),
                'overall': mf.overall.to_dict(),
                'difference': mf.difference().to_dict(),
                'ratio': mf.ratio().to_dict(),
                'groups': list(mf.by_group.index)
            }
            
            print(f"\n  Performance by Group:")
            print(mf.by_group.round(3))
            
            dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features[col])
            dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features[col])
            
            results[col]['demographic_parity'] = {
                'difference': float(dp_diff),
                'ratio': float(dp_ratio)
            }
            
            print(f"\n  Fairness Metrics:")
            print(f"    Demographic Parity Difference: {dp_diff:.3f}")
            print(f"    Demographic Parity Ratio: {dp_ratio:.3f}")
            
            violations = self._check_violations(col, results[col])
            results[col]['violations'] = violations
            
            if violations:
                print(f"\n  ⚠️  FAIRNESS VIOLATIONS DETECTED:")
                for violation in violations:
                    print(f"      • {violation}")
            else:
                print(f"\n  ✓ No significant fairness violations")
        
        self.fairness_results = results
        
        print("\n" + "="*80)
        print("FAIRNESS ASSESSMENT SUMMARY")
        print("="*80)
        
        self._print_summary(results)
        
        return results
    
    def _check_violations(self, attribute, results):
        """Check for fairness violations"""
        violations = []
        
        dp_ratio = results['demographic_parity']['ratio']
        if dp_ratio < 0.8:
            violations.append(f"Demographic parity violation: ratio = {dp_ratio:.3f} (should be ≥ 0.8)")
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in results['difference']:
                diff = abs(results['difference'][metric])
                if diff > 0.1:
                    violations.append(f"{metric.capitalize()} disparity: {diff:.3f} difference between groups")
        
        return violations
    
    def _print_summary(self, results):
        """Print overall fairness summary"""
        total_violations = sum(len(r['violations']) for r in results.values())
        
        if total_violations == 0:
            print("\n✅ FAIR MODEL")
            print("   No significant fairness violations detected")
        else:
            print(f"\n⚠️  FAIRNESS CONCERNS")
            print(f"   {total_violations} violations detected")
            print("\n📋 RECOMMENDATIONS:")
            print("   • Apply bias mitigation techniques")
            print("   • Re-evaluate after interventions")
    
    def generate_fairness_report(self):
        """Generate fairness report"""
        if not self.fairness_results:
            return pd.DataFrame()
        
        report_data = []
        
        for attr, results in self.fairness_results.items():
            dp_diff = results['demographic_parity']['difference']
            dp_ratio = results['demographic_parity']['ratio']
            
            report_data.append({
                'protected_attribute': attr,
                'metric': 'demographic_parity_difference',
                'value': dp_diff
            })
            
            report_data.append({
                'protected_attribute': attr,
                'metric': 'demographic_parity_ratio',
                'value': dp_ratio
            })
        
        return pd.DataFrame(report_data)


def quick_fairness_check(y_true, y_pred, sensitive_features):
    """Quick fairness check - convenience function"""
    evaluator = FairlearnEvaluator()
    results = evaluator.compute_metrics(y_true, y_pred, sensitive_features)
    report_df = evaluator.generate_fairness_report()
    
    return {
        'metrics': results,
        'report': report_df,
        'evaluator': evaluator
    }
