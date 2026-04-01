# 📊 Credit Risk Scoring System

> **Actuarial credit risk platform** — PD · LGD · EAD · Expected Loss · Monte Carlo simulation  
> Built with XGBoost, scikit-learn, and Streamlit

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This project implements a **production-grade credit risk scoring pipeline** that goes beyond typical ML classification work. It integrates actuarial methods (PD/LGD/EAD/EL) and regulatory-aligned risk metrics (VaR, CVaR, Economic Capital) into an interactive Streamlit application.

The system accepts any credit dataset via CSV upload and requires no code changes — columns are mapped through the UI.

**Key result on Give Me Some Credit (Kaggle):** XGBoost AUC-ROC = **0.863** (5-fold CV).

---

## Features

| Layer | What it does |
|---|---|
| **Data Validation** | 7 checks: column types, target encoding, missing values, imbalance, range anomalies, text-to-numeric conversion |
| **Preprocessing** | Median imputation, winsorization (p99), utilization clipping, age correction |
| **Feature Engineering** | 6 derived features: `total_past_due`, `clean_history`, `income_per_dependent`, `high_utilization`, `debt_to_income`, `age_segment` |
| **Model Comparison** | Logistic Regression · Random Forest · XGBoost — all with StratifiedKFold CV (5 folds) |
| **Actuarial Analysis** | PD × LGD × EAD = Expected Loss per borrower; LGD calibrated to Basel II retail bands |
| **Monte Carlo** | 5,000 vectorized scenarios → Loss distribution, VaR 95/99%, CVaR, Economic Capital |
| **Export** | 4-sheet Excel report: Scoring, Model Comparison, Actuarial Summary, Portfolio Segmentation |

---

## Methodology

### Credit Risk Components (Basel II — Retail Portfolio)

```
Expected Loss (EL) = PD × LGD × EAD
```

| Component | Definition | Source in this project |
|---|---|---|
| **PD** (Probability of Default) | XGBoost predicted probability | Model output |
| **LGD** (Loss Given Default) | % of EAD lost if borrower defaults | Basel II bands: 35%–80% by risk segment |
| **EAD** (Exposure at Default) | Estimated outstanding balance at default | Monthly income × exposure months |
| **EL** (Expected Loss) | Average loss expected from the portfolio | PD × LGD × EAD |

### Monte Carlo Simulation

The portfolio loss distribution is estimated by simulating 5,000 independent scenarios using vectorized `numpy.random.binomial`. Each scenario draws defaults for every approved borrower based on their predicted PD.

From the resulting distribution:
- **VaR 99%** — maximum loss not exceeded in 99% of scenarios
- **CVaR 99%** — average loss in the worst 1% of scenarios (tail risk)
- **Economic Capital** — VaR 99% minus Expected Loss (unexpected loss buffer)

This follows the conceptual framework of the **Basel II Internal Ratings-Based (IRB)** approach for retail credit portfolios.

---

## Project Structure

```
credit_risk_app/
│
├── app.py                  # Streamlit UI and pipeline orchestration
├── requirements.txt        # Pinned dependencies
│
└── pipeline/
    ├── __init__.py
    ├── validation.py       # Data quality checks (7 rules)
    ├── preprocessing.py    # Cleaning and feature engineering
    ├── modeling.py         # Model training, CV, feature importance
    └── actuarial.py        # PD/LGD/EAD/EL, Monte Carlo, Excel export
```

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/cesarcruz21/credit-risk-scoring.git
cd credit-risk-scoring

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.  
Click **"🎲 Usar dataset de demo"** in the sidebar to run a full analysis immediately — no CSV needed.

---

## Dataset

The default column mapping targets **[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)** (Kaggle, 150k rows).

Any credit CSV works. Required columns (names configurable in the UI):

| Internal name | Default column name | Description |
|---|---|---|
| target | `SeriousDlqin2yrs` | 1 = defaulted within 2 years |
| ingreso | `MonthlyIncome` | Monthly income |
| edad | `age` | Borrower age |
| utilizacion | `RevolvingUtilizationOfUnsecuredLines` | Credit utilization ratio [0–1] |
| deuda_ratio | `DebtRatio` | Debt-to-income ratio |
| dependientes | `NumberOfDependents` | Number of financial dependents |
| atraso_30 | `NumberOfTime30-59DaysPastDueNotWorse` | 30–59 day delinquencies |
| atraso_60 | `NumberOfTime60-89DaysPastDueNotWorse` | 60–89 day delinquencies |
| atraso_90 | `NumberOfTimes90DaysLate` | 90+ day delinquencies |
| lineas_abiertas | `NumberOfOpenCreditLinesAndLoans` | Open credit lines |

---

## Results (Give Me Some Credit)

| Model | AUC-ROC (CV 5-fold) |
|---|---|
| XGBoost | **0.8630 ± 0.0021** |
| Random Forest | 0.8401 ± 0.0034 |
| Logistic Regression | 0.7892 ± 0.0018 |

Top predictive features: `utilizacion`, `atraso_90`, `total_past_due`, `clean_history`, `deuda_ratio`

---

## Author

**Cesar Cruz**  
Actuarial Science student · Data Science portfolio  
[GitHub](https://github.com/cesarcruz21) · [LinkedIn](https://linkedin.com/in/cesarcruz21)

---

## License

MIT — free to use, modify, and distribute with attribution.
