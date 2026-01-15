"""
PII Detection and Anonymization
Uses Microsoft Presidio for detecting and anonymizing personally identifiable information
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


class PIIDetector:
    """
    Detect and anonymize personally identifiable information (PII) in datasets
    
    Uses Microsoft Presidio to detect:
    - PERSON (names)
    - EMAIL_ADDRESS
    - PHONE_NUMBER
    - CREDIT_CARD
    - IBAN_CODE
    - US_SSN
    - And more...
    """
    
    def __init__(self, language: str = 'en'):
        """
        Initialize PII detector
        
        Args:
            language: Language for analysis (default: 'en')
        """
        self.language = language
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.detection_results = {}
    
    def detect_pii(self, df: pd.DataFrame, sample_size: int = 100) -> Dict:
        """
        Detect PII in all text columns of a DataFrame
        
        Args:
            df: Input DataFrame
            sample_size: Number of rows to sample for detection
            
        Returns:
            Dictionary mapping column names to detected PII types
        """
        print("\n" + "="*80)
        print("PII DETECTION WITH PRESIDIO")
        print("="*80)
        
        results = {}
        text_columns = df.select_dtypes(include=['object']).columns
        
        if len(text_columns) == 0:
            print("\n✓ No text columns found - no PII risk")
            return results
        
        print(f"\n[Analyzing {len(text_columns)} text columns]")
        
        for col in text_columns:
            print(f"\n  Checking column: {col}")
            sample_data = df[col].dropna().astype(str).head(sample_size).tolist()
            
            if not sample_data:
                print(f"    └─ Empty column, skipping")
                continue
            
            pii_found = []
            pii_examples = {}
            
            for text in sample_data:
                if len(text) > 0:
                    analysis = self.analyzer.analyze(text=text, language=self.language)
                    
                    if analysis:
                        for item in analysis:
                            pii_type = item.entity_type
                            pii_found.append(pii_type)
                            
                            if pii_type not in pii_examples:
                                pii_examples[pii_type] = {
                                    'text': text,
                                    'start': item.start,
                                    'end': item.end,
                                    'score': item.score
                                }
            
            if pii_found:
                unique_pii = list(set(pii_found))
                pii_counts = {pii: pii_found.count(pii) for pii in unique_pii}
                
                results[col] = {
                    'pii_types': unique_pii,
                    'counts': pii_counts,
                    'total_detections': len(pii_found),
                    'examples': pii_examples
                }
                
                print(f"    ⚠️  PII DETECTED:")
                for pii_type, count in pii_counts.items():
                    print(f"       • {pii_type}: {count} occurrences")
            else:
                print(f"    ✓ No PII detected")
        
        self.detection_results = results
        
        print("\n" + "="*80)
        print("PII DETECTION SUMMARY")
        print("="*80)
        
        if results:
            print(f"\n⚠️  PII found in {len(results)} columns:")
            for col, info in results.items():
                print(f"\n  Column: {col}")
                print(f"  Types: {', '.join(info['pii_types'])}")
                print(f"  Total detections: {info['total_detections']}")
        else:
            print("\n✓ No PII detected")
        
        return results
    
    def anonymize_dataframe(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Anonymize PII in specified columns"""
        print("\n" + "="*80)
        print("PII ANONYMIZATION")
        print("="*80)
        
        df_anon = df.copy()
        
        if columns is None:
            if not self.detection_results:
                print("\n⚠️  Run detect_pii() first")
                return df_anon
            columns = list(self.detection_results.keys())
        
        if not columns:
            print("\n✓ No columns to anonymize")
            return df_anon
        
        print(f"\n[Anonymizing {len(columns)} columns]")
        
        for col in columns:
            if col in df_anon.columns and df_anon[col].dtype == 'object':
                print(f"\n  Anonymizing: {col}")
                df_anon[col] = df_anon[col].apply(lambda x: self._anonymize_text(str(x)) if pd.notna(x) else x)
                print(f"    ✓ Complete")
        
        print("\n✓ ANONYMIZATION COMPLETE")
        return df_anon
    
    def _anonymize_text(self, text: str) -> str:
        """Anonymize a single text string"""
        results = self.analyzer.analyze(text=text, language=self.language)
        if results:
            return self.anonymizer.anonymize(text=text, analyzer_results=results).text
        return text
