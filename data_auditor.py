"""
Data Auditor
Formal data auditing with schema validation, quality checks, and drift detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from scipy import stats


class DataAuditor:
    """
    Formal data auditing with three key steps:
    1. Schema validation
    2. Quality checks (duplicates, missing data, outliers)
    3. Drift detection (vs reference distribution)
    """
    
    def __init__(self, df: pd.DataFrame, metadata: Dict):
        """
        Initialize auditor
        
        Args:
            df: Input DataFrame
            metadata: Dictionary with dataset metadata
        """
        self.df = df.copy()
        self.metadata = metadata
        self.audit_results = {
            'schema_validation': {},
            'quality_checks': {},
            'drift_detection': {},
            'compliance_status': 'PENDING'
        }
    
    def validate_schema(self) -> Dict:
        """
        Validate data schema (basic type checking)
        
        Returns:
            Schema validation results
        """
        print("\n" + "="*80)
        print("AUDIT STEP 1: Schema Validation")
        print("="*80)
        
        try:
            # Check for consistent data types
            schema_issues = []
            
            for col in self.df.columns:
                dtype = self.df[col].dtype
                
                # Check for mixed types in object columns
                if dtype == 'object':
                    # Try to detect if numeric data is stored as string
                    non_null = self.df[col].dropna()
                    if len(non_null) > 0:
                        try:
                            pd.to_numeric(non_null.head(100), errors='raise')
                            schema_issues.append(f"{col}: numeric data stored as string")
                        except:
                            pass  # It's truly a string column
            
            if schema_issues:
                print("⚠️  Schema issues detected:")
                for issue in schema_issues:
                    print(f"   - {issue}")
                self.audit_results['schema_validation'] = {
                    'status': 'WARNING',
                    'issues': schema_issues
                }
            else:
                print("✓ Schema validation PASSED")
                print("  All columns have consistent data types")
                self.audit_results['schema_validation'] = {
                    'status': 'PASS',
                    'message': 'All columns conform to expected types'
                }
        
        except Exception as e:
            print(f"✗ Schema validation FAILED: {e}")
            self.audit_results['schema_validation'] = {
                'status': 'FAIL',
                'message': str(e)
            }
        
        return self.audit_results['schema_validation']

    
    def run_quality_checks(self) -> Dict:
        """
        Run comprehensive data quality checks
        
        Returns:
            Quality check results
        """
        print("\n" + "="*80)
        print("AUDIT STEP 2: Data Quality Checks")
        print("="*80)
        
        quality_report = {
            'total_checks': 0,
            'passed': 0,
            'failed': 0,
            'warnings': []
        }
        
        # Optional: Run Great Expectations
        gx_results = self.run_great_expectations()
        quality_report['great_expectations'] = gx_results
        
        # Check 1: No duplicate rows
        duplicates = self.df.duplicated().sum()
        quality_report['total_checks'] += 1
        if duplicates == 0:
            print("✓ No duplicate rows")
            quality_report['passed'] += 1
        else:
            print(f"✗ Found {duplicates} duplicate rows")
            quality_report['failed'] += 1
            quality_report['warnings'].append(f"Duplicate rows: {duplicates}")
        
        # Check 2: Reasonable missing data percentage
        missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        quality_report['total_checks'] += 1
        if missing_pct < 30:
            print(f"✓ Total missing data: {missing_pct:.2f}% (acceptable)")
            quality_report['passed'] += 1
        else:
            print(f"⚠️  Total missing data: {missing_pct:.2f}% (high)")
            quality_report['warnings'].append(f"High missing data: {missing_pct:.2f}%")
        
        # Check 3: Data type consistency
        quality_report['total_checks'] += 1
        type_consistency = True
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check if numeric data stored as string
                try:
                    non_null = self.df[col].dropna()
                    if len(non_null) > 0:
                        pd.to_numeric(non_null.head(100), errors='raise')
                        print(f"⚠️  Column '{col}' contains numeric data stored as string")
                        type_consistency = False
                        quality_report['warnings'].append(f"{col}: numeric as string")
                except:
                    pass
        
        if type_consistency:
            print("✓ Data types are consistent")
            quality_report['passed'] += 1
        else:
            quality_report['failed'] += 1
        
        # Check 4: Outlier detection for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_cols = []
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 3 * IQR)) | (self.df[col] > (Q3 + 3 * IQR))).sum()
            
            if outliers > len(self.df) * 0.05:  # More than 5% outliers
                pct = (outliers / len(self.df)) * 100
                outlier_cols.append(col)
                print(f"⚠️  Column '{col}': {outliers} outliers ({pct:.1f}%)")
                quality_report['warnings'].append(f"{col}: {pct:.1f}% outliers")
        
        # Summary
        print(f"\n📊 Quality Check Summary:")
        print(f"   Total checks: {quality_report['total_checks']}")
        print(f"   Passed: {quality_report['passed']}")
        print(f"   Failed: {quality_report['failed']}")
        print(f"   Warnings: {len(quality_report['warnings'])}")
        
        self.audit_results['quality_checks'] = quality_report
        return quality_report
    
    def detect_drift(self, reference_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Drift detection - compare current data to reference distribution
        
        Args:
            reference_df: Reference DataFrame (e.g., training data)
            
        Returns:
            Drift detection results
        """
        print("\n" + "="*80)
        print("AUDIT STEP 3: Drift Detection")
        print("="*80)
        
        if reference_df is None:
            print("⚠️  No reference data provided, skipping drift detection")
            print("   In production, compare against training/baseline data")
            self.audit_results['drift_detection'] = {'status': 'SKIPPED'}
            return self.audit_results['drift_detection']
        
        # Statistical tests for drift
        drift_detected = []
        
        for col in self.df.columns:
            if col not in reference_df.columns:
                continue
            
            # Only test numeric columns
            if self.df[col].dtype in ['int64', 'float64']:
                try:
                    # Kolmogorov-Smirnov test
                    stat, p_value = stats.ks_2samp(
                        self.df[col].dropna(),
                        reference_df[col].dropna()
                    )
                    
                    if p_value < 0.05:
                        print(f"⚠️  Drift detected in '{col}' (p={p_value:.4f})")
                        drift_detected.append({
                            'column': col,
                            'p_value': float(p_value),
                            'test': 'KS'
                        })
                except:
                    pass  # Skip if test fails
        
        if drift_detected:
            print(f"\n⚠️  Drift detected in {len(drift_detected)} columns")
            self.audit_results['drift_detection'] = {
                'status': 'DRIFT_DETECTED',
                'drifted_columns': drift_detected
            }
        else:
            print("✓ No significant drift detected")
            self.audit_results['drift_detection'] = {'status': 'NO_DRIFT'}
        
        return self.audit_results['drift_detection']
    
    def generate_audit_report(self) -> Dict:
        """
        Generate comprehensive audit report and determine compliance status
        
        Returns:
            Complete audit results
        """
        print("\n" + "="*80)
        print("AUDIT REPORT SUMMARY")
        print("="*80)
        
        # Determine overall compliance status
        schema_pass = self.audit_results['schema_validation'].get('status') in ['PASS', 'WARNING']
        quality_pass = self.audit_results['quality_checks'].get('failed', 0) == 0
        
        if schema_pass and quality_pass:
            self.audit_results['compliance_status'] = 'COMPLIANT'
            print("\n✓ DATA AUDIT: COMPLIANT")
            print("   Dataset meets quality and validation requirements")
            print("   Safe to proceed with model training")
        else:
            self.audit_results['compliance_status'] = 'NON_COMPLIANT'
            print("\n✗ DATA AUDIT: NON-COMPLIANT")
            print("   Dataset has quality issues that must be addressed")
            print("   DO NOT proceed with model training until issues are resolved")
        
        # Display recommendations
        if self.audit_results['quality_checks'].get('warnings'):
            print("\n📋 RECOMMENDATIONS:")
            print("\n   Quality Warnings:")
            for warning in self.audit_results['quality_checks']['warnings']:
                print(f"   - {warning}")
        
        return self.audit_results
    
    def run_great_expectations(self) -> Dict:
        """
        Run Great Expectations validation suite
        
        Returns:
            Great Expectations validation results
        """
        print("\n" + "="*80)
        print("AUDIT STEP 4: Great Expectations Validation")
        print("="*80)
        
        try:
            import great_expectations as gx
            from great_expectations.dataset import PandasDataset
            
            # Convert to GX dataset
            ge_df = PandasDataset(self.df)
            
            print("✓ Great Expectations available")
            
            # Define expectations
            expectations_results = []
            
            # Expectation 1: No null values in critical columns
            if 'target' in self.metadata:
                target = self.metadata['target']
                if target in ge_df.columns:
                    result = ge_df.expect_column_values_to_not_be_null(target)
                    expectations_results.append({
                        'expectation': 'Target not null',
                        'success': result.success,
                        'column': target
                    })
            
            # Expectation 2: Columns exist
            for col in ge_df.columns:
                result = ge_df.expect_column_to_exist(col)
                if not result.success:
                    expectations_results.append({
                        'expectation': 'Column exists',
                        'success': False,
                        'column': col
                    })
            
            # Expectation 3: Row count in range
            row_count = len(ge_df)
            result = ge_df.expect_table_row_count_to_be_between(
                min_value=1,
                max_value=row_count * 2
            )
            expectations_results.append({
                'expectation': 'Row count valid',
                'success': result.success,
                'value': row_count
            })
            
            # Summary
            passed = sum(1 for r in expectations_results if r.get('success', False))
            total = len(expectations_results)
            
            print(f"\n📊 Great Expectations Results:")
            print(f"   Expectations passed: {passed}/{total}")
            
            self.audit_results['great_expectations'] = {
                'status': 'PASS' if passed == total else 'FAIL',
                'passed': passed,
                'total': total,
                'results': expectations_results
            }
            
            return self.audit_results['great_expectations']
            
        except ImportError:
            print("⚠️  Great Expectations not installed")
            print("   Install with: pip install great_expectations")
            self.audit_results['great_expectations'] = {
                'status': 'SKIPPED',
                'message': 'Great Expectations not available'
            }
            return self.audit_results['great_expectations']
        except Exception as e:
            print(f"✗ Error running Great Expectations: {e}")
            self.audit_results['great_expectations'] = {
                'status': 'ERROR',
                'message': str(e)
            }
            return self.audit_results['great_expectations']
    
    def run_full_audit(self, 
                      reference_df: Optional[pd.DataFrame] = None,
                      use_great_expectations: bool = True) -> Dict:
        """
        Run complete data audit
        
        Args:
            reference_df: Optional reference DataFrame for drift detection
            use_great_expectations: Whether to run Great Expectations checks
            
        Returns:
            Complete audit results
        """
        print("\n" + "#"*80)
        print("#" + " "*28 + "FORMAL DATA AUDIT" + " "*34 + "#")
        print("#"*80)
        
        self.validate_schema()
        self.run_quality_checks()
        self.detect_drift(reference_df)
        
        if use_great_expectations:
            self.run_great_expectations()
        
        audit_report = self.generate_audit_report()
        
        print("\n" + "#"*80)
        print("#" + " "*27 + "AUDIT COMPLETE" + " "*38 + "#")
        print("#"*80)
        
        return audit_report
    
    def get_compliance_status(self) -> str:
        """Get the compliance status"""
        return self.audit_results.get('compliance_status', 'UNKNOWN')
