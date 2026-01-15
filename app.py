"""
Streamlit Web Interface for Ethical AI Pipeline
Upload data → Run Governance Checks → Download reports
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path

# Import governance pipeline
from governance_pipeline import DataGovernancePipeline

# Page config
st.set_page_config(
    page_title="Ethical AI Governance",
    page_icon="⚖️",
    layout="wide"
)

# Title
st.title("⚖️ Ethical AI Governance Pipeline")
st.markdown("**Upload data → Pre-training governance checks → Compliance report**")
st.markdown("*EU AI Act & GDPR Compliant*")

# Sidebar
st.sidebar.header("Configuration")

# Domain selection
domain = st.sidebar.selectbox(
    "Select Domain",
    options=['credit_scoring', 'hiring', 'healthcare', 'education'],
    help="Choose the high-risk AI domain for your use case"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**About This Tool**")
st.sidebar.info("""
This tool performs pre-training data governance checks:
- 🔒 PII Detection
- 🗑️ Data Minimisation (GDPR)
- 🔬 Deep EDA (Bias Detection)
- ✅ Data Quality Auditing
- 📊 Representation Analysis
- ⚖️ Compliance Assessment

**No model training at this stage.**
""")

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Check", "📋 Governance Report", "📚 Documentation"])

with tab1:
    st.header("Step 1: Upload Your Data")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset for governance checks"
    )
    
    if uploaded_file:
        # Save uploaded file
        upload_path = Path("uploads") / uploaded_file.name
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and preview
        if uploaded_file.name.endswith('.csv'):
            df_preview = pd.read_csv(upload_path)
        else:
            df_preview = pd.read_excel(upload_path)
        
        st.success(f"✓ File uploaded: {uploaded_file.name}")
        
        # Show preview
        st.subheader("Data Preview")
        st.dataframe(df_preview.head(10))
        
        # Show stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df_preview))
        with col2:
            st.metric("Columns", len(df_preview.columns))
        with col3:
            st.metric("Missing %", f"{(df_preview.isnull().sum().sum() / (len(df_preview) * len(df_preview.columns)) * 100):.1f}%")
        
        # Step 2: Configure
        st.header("Step 2: Configure Governance Checks")
        
        target_col = st.selectbox(
            "Select Target Column",
            options=list(df_preview.columns),
            help="Choose the column you want to predict (your outcome variable)"
        )
        
        # Step 3: Run governance checks
        st.header("Step 3: Run Governance Checks")
        
        st.info("⚠️ Note: This will run data governance checks ONLY. No model training will occur.")
        
        if st.button("🔍 Run Governance Checks", type="primary"):
            # Initialize governance pipeline
            governance = DataGovernancePipeline(domain=domain)
            
            # Run with progress updates
            with st.spinner("Running governance checks... This may take a few minutes."):
                try:
                    # Create progress sections
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Load data
                    status_text.text("Loading data...")
                    progress_bar.progress(5)
                    
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(upload_path)
                    else:
                        df = pd.read_excel(upload_path)
                    
                    # Step 1: PII Detection
                    status_text.text("🔒 Step 1/6: PII Detection...")
                    progress_bar.progress(15)
                    
                    # Step 2: Data Minimisation
                    status_text.text("🗑️ Step 2/6: Data Minimisation (GDPR)...")
                    progress_bar.progress(30)
                    
                    # Step 3: Deep EDA
                    status_text.text("🔬 Step 3/6: Deep EDA (Bias Detection)...")
                    progress_bar.progress(50)
                    
                    # Step 4: Data Auditing
                    status_text.text("✅ Step 4/6: Data Quality Auditing...")
                    progress_bar.progress(65)
                    
                    # Step 5: Representation
                    status_text.text("📊 Step 5/6: Representation Analysis...")
                    progress_bar.progress(80)
                    
                    # Step 6: Compliance
                    status_text.text("⚖️ Step 6/6: Compliance Assessment...")
                    progress_bar.progress(90)
                    
                    # Run governance checks
                    report, df_clean = governance.run_governance_checks(
                        df=df,
                        target_col=target_col
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Governance checks complete!")
                    
                    # Store in session state
                    st.session_state['governance_report'] = report
                    st.session_state['df_clean'] = df_clean
                    st.session_state['checks_run'] = True
                    
                    # Show status
                    compliance_status = report['compliance']['overall_status']
                    if compliance_status == 'COMPLIANT':
                        st.success("✅ Governance checks completed - Dataset is COMPLIANT!")
                        st.balloons()
                    elif compliance_status == 'COMPLIANT_WITH_WARNINGS':
                        st.warning("⚠️ Governance checks completed - Dataset is COMPLIANT WITH WARNINGS")
                    else:
                        st.error("❌ Governance checks completed - Dataset is NON-COMPLIANT")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)

with tab2:
    st.header("📋 Governance Report")
    
    if 'checks_run' in st.session_state and st.session_state['checks_run']:
        report = st.session_state['governance_report']
        
        # ========== COMPLIANCE STATUS ==========
        st.subheader("⚖️ Compliance Status")
        compliance = report['compliance']
        
        status = compliance['overall_status']
        if status == 'COMPLIANT':
            st.success(f"✅ Overall Status: **{status}**")
        elif status == 'COMPLIANT_WITH_WARNINGS':
            st.warning(f"⚠️ Overall Status: **{status}**")
        else:
            st.error(f"❌ Overall Status: **{status}**")
        
        # Detailed Compliance Metrics
        st.markdown("---")
        st.subheader("📊 Detailed Compliance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**GDPR Compliance**")
            
            gdpr_5_1_c = compliance['gdpr_article_5_1_c']
            if gdpr_5_1_c == 'COMPLIANT':
                st.success(f"✅ Art. 5(1)(c) Data Minimisation: {gdpr_5_1_c}")
            elif gdpr_5_1_c == 'WARNING':
                st.warning(f"⚠️ Art. 5(1)(c) Data Minimisation: {gdpr_5_1_c}")
            else:
                st.error(f"❌ Art. 5(1)(c) Data Minimisation: {gdpr_5_1_c}")
            
            gdpr_5_1_d = compliance['gdpr_article_5_1_d']
            if gdpr_5_1_d == 'COMPLIANT':
                st.success(f"✅ Art. 5(1)(d) Data Accuracy: {gdpr_5_1_d}")
            elif gdpr_5_1_d == 'WARNING':
                st.warning(f"⚠️ Art. 5(1)(d) Data Accuracy: {gdpr_5_1_d}")
            else:
                st.error(f"❌ Art. 5(1)(d) Data Accuracy: {gdpr_5_1_d}")
            
            gdpr_9 = compliance['gdpr_article_9']
            if gdpr_9 == 'COMPLIANT':
                st.success(f"✅ Art. 9 Special Categories: {gdpr_9}")
            elif gdpr_9 == 'WARNING':
                st.warning(f"⚠️ Art. 9 Special Categories: {gdpr_9}")
            else:
                st.error(f"❌ Art. 9 Special Categories: {gdpr_9}")
        
        with col2:
            st.markdown("**EU AI Act Compliance**")
            
            eu_ai_act = compliance['eu_ai_act_annex_iii']
            if eu_ai_act == 'COMPLIANT':
                st.success(f"✅ Annex III High-risk AI: {eu_ai_act}")
            elif eu_ai_act == 'WARNING':
                st.warning(f"⚠️ Annex III High-risk AI: {eu_ai_act}")
            else:
                st.error(f"❌ Annex III High-risk AI: {eu_ai_act}")
        
        # Issues & Warnings
        if compliance['issues']:
            st.markdown("---")
            st.subheader("🚨 Critical Issues")
            for issue in compliance['issues']:
                st.error(f"• {issue}")
        
        if compliance['warnings']:
            st.markdown("---")
            st.subheader("⚠️ Warnings")
            for warning in compliance['warnings']:
                st.warning(f"• {warning}")
        
        if compliance['recommendations']:
            st.markdown("---")
            st.subheader("💡 Recommendations")
            for rec in compliance['recommendations']:
                st.info(f"• {rec}")
        
        # ========== DETAILED RESULTS ==========
        st.markdown("---")
        st.subheader("📊 Detailed Check Results")
        
        checks = report['checks']
        
        # PII Detection
        with st.expander("🔒 PII Detection Results", expanded=False):
            pii = checks['pii_detection']
            if pii['pii_count'] > 0:
                st.warning(f"⚠️ Found PII in {pii['pii_count']} columns")
                st.write("**Columns with PII:**")
                for col in pii['pii_columns']:
                    st.write(f"• {col}")
                
                if pii['details']:
                    st.write("**Details:**")
                    for col, details in pii['details'].items():
                        st.write(f"**{col}:**")
                        st.write(f"  - Types: {', '.join(details['pii_types'])}")
                        st.write(f"  - Total detections: {details['total_detections']}")
            else:
                st.success("✅ No PII detected")
        
        # Data Minimisation
        with st.expander("🗑️ Data Minimisation Results", expanded=False):
            min_data = checks['data_minimisation']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Features", min_data['original_features'])
            with col2:
                st.metric("Minimal Features", min_data['minimal_features'])
            with col3:
                reduction = min_data['original_features'] - min_data['minimal_features']
                st.metric("Features Excluded", reduction)
            
            if min_data['exclusion_log']:
                st.write("**Exclusion Log:**")
                exclusion_df = pd.DataFrame(min_data['exclusion_log'])
                st.dataframe(exclusion_df)
            else:
                st.info("No features were excluded")
            
            # Show feature classification
            report_data = min_data['report']
            st.write(f"**Feature Classification:**")
            st.write(f"• Essential: {report_data['essential_features_count']}")
            st.write(f"• Prohibited: {report_data['prohibited_features_count']}")
            st.write(f"• Optional: {report_data['optional_features_count']}")
        
        # Deep EDA
        with st.expander("🔬 Deep EDA Findings", expanded=False):
            eda = checks['deep_eda']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Under-represented Groups", eda['underrepresented_groups_count'])
            with col2:
                st.metric("Proxy Variables", eda['proxy_variables_count'])
            with col3:
                st.metric("Statistical Disparities", eda['statistical_disparities_count'])
            
            # Show details
            if eda['underrepresented_groups_count'] > 0:
                st.write("**Under-represented Groups:**")
                for group in eda['findings']['underrepresented_groups']:
                    st.write(f"• {group['column']}: {group['group']} ({group['percentage']:.1f}%, n={group['count']})")
            
            if eda['proxy_variables_count'] > 0:
                st.write("**Potential Proxy Variables:**")
                for proxy in eda['findings']['proxy_variables']:
                    st.write(f"• {proxy['proxy_variable']} → {proxy['protected_attribute']} (r={proxy['correlation']:.3f})")
            
            if eda['statistical_disparities_count'] > 0:
                st.write("**Statistical Disparities Detected:**")
                for attr, disparity in eda['findings']['statistical_disparities'].items():
                    st.write(f"• {attr}: χ²={disparity['chi2']:.2f}, p={disparity['p_value']:.4f}")
        
        # Data Auditing
        with st.expander("✅ Data Quality Audit", expanded=False):
            audit = checks['data_auditing']
            
            status = audit['compliance_status']
            if status == 'COMPLIANT':
                st.success(f"✅ Audit Status: {status}")
            else:
                st.error(f"❌ Audit Status: {status}")
            
            # Schema Validation
            if 'schema_validation' in audit:
                schema = audit['schema_validation']
                st.write(f"**Schema Validation:** {schema['status']}")
            
            # Quality Checks
            if 'quality_checks' in audit:
                quality = audit['quality_checks']
                st.write(f"**Quality Checks:**")
                st.write(f"• Passed: {quality.get('passed', 0)}/{quality.get('total_checks', 0)}")
                
                if quality.get('warnings'):
                    st.write("**Warnings:**")
                    for warning in quality['warnings']:
                        st.warning(warning)
            
            # Great Expectations
            if 'great_expectations' in audit:
                gx = audit['great_expectations']
                if gx.get('status') == 'PASS':
                    st.success(f"✅ Great Expectations: {gx['passed']}/{gx['total']} checks passed")
                elif gx.get('status') == 'SKIPPED':
                    st.info("ℹ️ Great Expectations not available")
        
        # Representation Analysis
        with st.expander("📊 Representation Analysis", expanded=False):
            rep = checks['representation']
            
            if rep:
                for attr, data in rep.items():
                    st.write(f"**{attr} Distribution:**")
                    
                    # Create bar chart
                    dist_df = pd.DataFrame.from_dict(
                        data['distribution'], 
                        orient='index', 
                        columns=['Percentage']
                    )
                    dist_df['Percentage'] = dist_df['Percentage'] * 100
                    st.bar_chart(dist_df)
                    
                    # Show outcome rates by group
                    if data['outcome_by_group']:
                        st.write("**Outcome Rates by Group:**")
                        for group, rate in data['outcome_by_group'].items():
                            st.write(f"• {group}: {rate*100:.1f}%")
                    
                    # Show warnings
                    if data['warnings']:
                        for warning in data['warnings']:
                            st.warning(warning)
                    
                    st.write("---")
            else:
                st.info("No protected attributes found for representation analysis")
        
        # ========== DOWNLOADS ==========
        st.markdown("---")
        st.subheader("📥 Download Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Governance report JSON
            report_json = json.dumps(report, indent=2)
            st.download_button(
                label="📋 Download Governance Report (JSON)",
                data=report_json,
                file_name="governance_report.json",
                mime="application/json",
                help="Complete governance report with all checks and compliance status"
            )
        
        with col2:
            # Cleaned dataset CSV
            df_clean = st.session_state['df_clean']
            csv = df_clean.to_csv(index=False)
            st.download_button(
                label="📊 Download Cleaned Dataset (CSV)",
                data=csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                help="Minimized dataset with prohibited features removed"
            )
        
        with col3:
            # Compliance summary
            compliance_summary = f"""
GOVERNANCE COMPLIANCE SUMMARY
================================

Domain: {report['domain']}
Status: {compliance['overall_status']}
Date: {report['pipeline_end']}

GDPR COMPLIANCE:
- Art. 5(1)(c) Data Minimisation: {compliance['gdpr_article_5_1_c']}
- Art. 5(1)(d) Data Accuracy: {compliance['gdpr_article_5_1_d']}
- Art. 9 Special Categories: {compliance['gdpr_article_9']}

EU AI ACT COMPLIANCE:
- Annex III High-risk AI: {compliance['eu_ai_act_annex_iii']}

CRITICAL ISSUES: {len(compliance['issues'])}
WARNINGS: {len(compliance['warnings'])}
RECOMMENDATIONS: {len(compliance['recommendations'])}

================================
"""
            st.download_button(
                label="📄 Download Compliance Summary (TXT)",
                data=compliance_summary,
                file_name="compliance_summary.txt",
                mime="text/plain",
                help="Human-readable compliance summary"
            )
    
    else:
        st.info("👈 Upload data and run governance checks to see results here")

with tab3:
    st.header("📚 Documentation")
    
    st.markdown("""
    ## How It Works
    
    This pipeline implements comprehensive **pre-training data governance** for ethical AI:
    
    ### Governance Checks (6 Steps)
    
    #### 1. 🔒 PII Detection
    - Uses Microsoft Presidio to detect personally identifiable information
    - Identifies: names, emails, phone numbers, addresses, SSNs, credit cards
    - **Regulation:** GDPR Article 9 (Special categories of data)
    
    #### 2. 🗑️ Data Minimisation
    - Classifies features as Essential/Prohibited/Optional
    - Removes non-essential PII and protected attributes
    - Assesses feature importance using Random Forest
    - **Regulation:** GDPR Article 5(1)(c) - Data minimisation principle
    
    #### 3. 🔬 EDA 
    - **Missing Sensitive Attributes:** Identifies fairness blind spots
    - **Under-represented Groups:** Flags groups with <5% representation
    - **Missingness as Bias:** Tests if missing data correlates with outcomes (MNAR)
    - **Proxy Variables:** Detects features highly correlated with protected attributes
    - **Statistical Disparities:** Chi-square tests for outcome disparities across groups
    - **Regulation:** EU AI Act transparency and fairness requirements
    
    #### 4. ✅ Data Quality Auditing
    - Schema validation (data type consistency)
    - Quality checks (duplicates, missing data, outliers)
    - Drift detection (vs reference distribution)
    - **Regulation:** GDPR Article 5(1)(d) - Data accuracy
    
    #### 5. 📊 Representation Analysis
    - Distribution analysis across protected attributes
    - Outcome rates by demographic group
    - Under-representation warnings
    - **Regulation:** EU AI Act Article 10 - Data governance
    
    #### 6. ⚖️ Compliance Assessment
    - Evaluates GDPR compliance (Articles 5(1)(c), 5(1)(d), 9)
    - Evaluates EU AI Act compliance (Annex III - High-risk AI systems)
    - Identifies critical issues, warnings, and provides recommendations
    - Overall status: COMPLIANT / COMPLIANT_WITH_WARNINGS / NON_COMPLIANT
    
    ---
    
    ## Supported Domains - for later stages - at this stage, only upload option 
    
    - **Credit Scoring**: Loan approval, credit risk assessment
    - **Hiring**: Recruitment, employee screening, promotion decisions
    - **Healthcare**: Medical diagnosis, patient risk assessment, treatment recommendations
    - **Education**: Student assessment, admission decisions, scholarship allocation
    
    ---
    
    ## Regulatory Compliance
    
    ### GDPR (General Data Protection Regulation)
    
    **Article 5(1)(c) - Data Minimisation:**
    - Personal data shall be adequate, relevant and limited to what is necessary
    - Implementation: Automatic feature exclusion, importance-based selection
    
    **Article 5(1)(d) - Accuracy:**
    - Personal data shall be accurate and kept up to date
    - Implementation: Comprehensive quality checks, validation suites
    
    **Article 9 - Special Categories:**
    - Processing of data revealing racial/ethnic origin, political opinions, religious beliefs, etc. is prohibited
    - Implementation: Identification and proper handling of protected attributes
    
    ### EU AI Act
    
    **Annex III - High-risk AI Systems:**
    - Credit scoring, employment, education, law enforcement
    - Requirements: Bias detection, fairness assessment, transparency, data governance
    - Implementation: Deep EDA, representation analysis, compliance reporting
    
    **Article 10 - Data Governance:**
    - Training data must be relevant, representative, and free from errors
    - Implementation: Data quality auditing, drift detection, schema validation
    
    ---
    
      
    ### ⚠️ This Tool Does NOT Train Models at this stage
    
    This governance pipeline performs **pre-training checks only**. It will:
    - ✅ Analyze your data for bias and quality issues
    - ✅ Check regulatory compliance
    - ✅ Generate governance reports
    - ✅ Provide a cleaned, minimized dataset
    
    NOT at this stage:
    - ❌ Train machine learning models
    - ❌ Make predictions
    - ❌ Deploy models to production
    
    ### 🔄 Next Steps After Governance
    
    If your dataset passes governance checks (status = COMPLIANT):
    1. Use the cleaned dataset for model training
    2. Apply bias mitigation techniques during training
    3. Perform post-training fairness evaluation
    4. Document all decisions and interventions
    5. Maintain audit trail for regulatory compliance
    
    If your dataset fails (status = NON_COMPLIANT):
    1. Review critical issues and warnings
    2. Fix identified problems:
       - Collect more data for under-represented groups
       - Remove or engineer proxy variables
       - Improve data quality
    3. Re-run governance checks
    4. Only proceed when compliant
    
    ---
    
    ## Requirements
    
    ### Python Libraries
    ```
    pandas>=2.0.0
    numpy>=1.24.0
    streamlit>=1.28.0
    presidio-analyzer>=2.2.0
    presidio-anonymizer>=2.2.0
    great-expectations>=0.18.0
    fairlearn>=0.9.0
    scikit-learn>=1.3.0
    scipy>=1.10.0
    spacy>=3.0.0
    ```
    
    ### Installation
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_lg
    python -m streamlit run app.py
    ```
    
    ---

    
    ---
    
    **Version:** 1.0.0  
    **Last Updated:** January 2026  
    **Compliance:** GDPR + EU AI Act
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Ethical AI Governance Pipeline v1.0**")
st.sidebar.markdown("🇪🇺 EU AI Act & GDPR Compliant")
st.sidebar.markdown("*Pre-training governance checks only*")
