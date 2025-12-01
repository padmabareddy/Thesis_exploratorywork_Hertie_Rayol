# Ethical AI Credit Pipeline

To demonstrates an ethical AI pre-training pipeline on adult income data set

objective:
Regulatory Alignment (high-level)
- GDPR: data minimisation, accuracy, privacy by design, accountability.
- EU AI Act: data governance and bias assessment steps for high-risk decision systems (use case examples, employment, credit, or similar socio-economic use cases).

High level checks on Data Quality Checks
- Basic schema & missingness inspection with pandas.
- Great Expectations checks for missing numeric values and label validity (income in {0,1}).

Fairness & Representation
- Representation analysis by sex, race, and age_group.
- Outcome rate (high-income proportion) by protected group.
- Simple Fairlearn metrics (selection rate, accuracy by group).

flow
- Automatic dataset download via `ucimlrepo`
- PII detection with Microsoft Presidio (best effort)
- Data minimisation (dropping PII / unused columns)
- Data quality checks using Great Expectations
- Fairness analysis of historical labels using Fairlearn

## Setup

```bash
python -m venv .venv
# Activate the venv, then:
pip install -r requirements.txt
```