# Data Governance Pipeline - Setup Instructions

## 📁 Files Provided

### Core Implementation Files
1. **governance_pipeline.py** - Main orchestrator (NEW)
2. **app_updated.py** - Updated Streamlit interface (REPLACES app.py)
3. **requirements.txt** - All dependencies
4. **test_governance.py** - Test script

### Existing Module Files (You Already Have)
- pii_detector.py
- data_minimisation.py
- data_auditor.py
- deep_eda.py
- fairlearn_metrics.py
- data_generator.py

---

## 🚀 Quick Setup (5 Minutes)

### Step 1: Organize Files

Place all files in the same directory:

```
AI_ethical_tool/
├── governance_pipeline.py      
├── app_updated.py              
├── requirements.txt            
├── test_governance.py         
├── pii_detector.py             
├── data_minimisation.py        
├── data_auditor.py             
├── deep_eda.py                 
├── fairlearn_metrics.py       
```




### Step 2:  Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Download spaCy language model (required for Presidio)
python -m spacy download en_core_web_lg
```

### Step 3: Run the Web Interface

```bash
    ### Installation
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_lg
    
# Start Streamlit app
streamlit run app_updated.py
or
python -m streamlit run app.py
```

**Access at:** http://localhost:8501

## 📋 Usage Instructions

### Option 1: Web Interface (Recommended)

1. **Start the app:**
   ```bash
   streamlit run app_updated.py
   ```

2. **Upload your data:**
   - Click "Choose a CSV or Excel file"
   - Select your dataset
   - Preview will show automatically

3. **Configure:**
   - Select domain (credit_scoring, hiring, healthcare, education)
   - Select target column

4. **Run checks:**
   - Click "🔍 Run Governance Checks"
   - Wait for all 6 steps to complete

5. **Review results:**
   - Check compliance status
   - Review detailed findings
   - Download reports

```

