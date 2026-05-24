# Late Delivery Risk Prediction in Global Supply Chain Operations

**Client:** APL Logistics (KWE Group) | **Platform:** Unified Mentor Private Limited

A machine learning-based predictive intelligence system that flags orders
likely to be delayed **before they are dispatched** — enabling supply chain
operations teams to take proactive action and reduce the cost of reactive
delay management.

---

## Live Demo
-- https://late-delivery-risk-prediction-unifiedmentors.streamlit.app/


---

## Project Overview

In global logistics networks, late deliveries cause SLA breaches, financial
penalties, and customer churn. This project transforms APL Logistics' approach
from reactive delay handling to proactive risk intelligence by predicting
delivery risk at the order level before dispatch.

Each order receives:
- A **Late Delivery Probability Score** (0–100%)
- A **Risk Category** — Low / Medium / High
- **Key Risk Drivers** explaining what is driving the risk
- **Recommended Actions** for operations teams

---

## Model Performance

| Metric | Score |
|---|---|
| Model | XGBoost (Tuned) |
| Accuracy | 97.87% |
| ROC-AUC | 0.9972 |
| Recall | 99.80% |
| Precision | 96.44% |
| F1 Score | 98.09% |
| Orders Analyzed | 180,517 |

---

## Key Findings

- **First Class shipping has a 95.3% late delivery rate** — the highest of all
  shipping modes, counterintuitively making it the riskiest option
- **Shipping Delay Gap** (real days − scheduled days) accounts for **79.38%**
  of the model's decision-making — a single engineered feature
- All global markets show similar delay rates (54.4%–55.2%) — the problem
  is systemic, not geographic
- Transfer payments show notably lower late delivery rates (48.5%) vs other
  payment types (56.6%–57.5%)

---

## Project Structure

```
late-delivery-risk-prediction/
│
├── data/
│   ├── apl_logistics.csv          ← raw dataset (not tracked by git)
│   ├── cleaned_data.csv           ← saved after notebook 02
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_preprocessing_feature_engineering.ipynb
│   ├── 05_modeling_evaluation.ipynb
│   └── 06_risk_scoring.ipynb
│
├── models/
│   ├── best_model.pkl
│   ├── encoders.pkl
│   ├── scaler.pkl
│   ├── preprocessing_config.pkl
│   ├── risk_thresholds.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost_baseline.pkl
│   └── xgboost_tuned.pkl
│
├── reports/
│   └── eda_figures/               ← all EDA plots saved as .png
│
├── outputs/
│   ├── scored_dataset.csv
│   ├── high_risk_orders.csv
│   └── risk_drivers.csv
│
├── src/
│   └── app.py                     ← Streamlit application
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Notebook Pipeline

| Notebook | Description |
|---|---|
| 01_data_understanding | Dataset structure, missing values, target distribution, feature importance (Cramér's V) |
| 02_data_cleaning | Drop leakage columns, PII, ID columns, duplicates — save cleaned_data.csv |
| 03_eda | 11 visualizations saved to reports/eda_figures/ |
| 04_preprocessing_feature_engineering | 6 engineered features, train/test split, encoding, scaling — save pkl files |
| 05_modeling_evaluation | 3 models, 5-fold CV, RandomizedSearchCV, confusion matrices, feature importance |
| 06_risk_scoring | Probability scores, risk categories, high-risk order list — save outputs |

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data Processing | Pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost, Imbalanced-learn |
| Model Persistence | Joblib |
| Web Application | Streamlit |
| Development | Jupyter Notebooks, VS Code |

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/late-delivery-risk-prediction.git
cd late-delivery-risk-prediction

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run src/app.py
```

> **Note:** The raw dataset `apl_logistics.csv` is not tracked by git due to
> its size. To reproduce the full pipeline, place the DataCo Supply Chain
> dataset in the `data/` folder and run the notebooks in order (01 → 06)
> before launching the app.

---

## Streamlit App Pages

| Page | Description |
|---|---|
| Home | Project overview, professional scope, methodology, usage guide |
| Risk Predictor | Enter order details → get probability score, risk category, risk drivers, recommended action |
| Risk Dashboard | Portfolio-level risk distribution and feature importance |
| Operations Panel | Filterable high-risk order table with CSV export |

> This application is designed for **supply chain professionals** — logistics
> analysts, operations managers, and dispatch coordinators. It requires
> operational data available only in internal order management systems.

---

## Project Article

Full technical walkthrough published on Dev.to:

[Late Delivery Risk Prediction in Global Supply Chain Operations](https://dev.to/sugnikm/how-i-built-a-late-delivery-risk-predictor-for-apl-logistics-what-a-95-delay-rate-in-first-class-1d11)

---

## About

This project was completed as part of the **Data Science internship program**
at **Unified Mentor Private Limited**, in collaboration with
**APL Logistics (KWE Group)**.

**Author:** Sugnik Mondal
**Program:** MBA — Data Analytics & Machine Learning, Manipal University Jaipur
