# Diabetes Readmission Prediction (End-to-End ML)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)]()
[![Flask](https://img.shields.io/badge/Flask-Deployment-green)]()

Interactive demo: **https://ml-diabetic-readmission-30dmittance-1.onrender.com/**  
Author: **Sagun Shrestha** · 📧 fssagunshrestha@gmail.com · 📞 (818)-930-0018 · GitHub: https://github.com/slavdefense

---

## 🧭 Project Overview

This project builds an **end-to-end machine learning pipeline** to predict the risk of **30-day hospital readmission** in diabetic patients. It covers **EDA, data cleaning, feature engineering, model training/tuning**, and **deployment** as a Flask web app.

- **Dataset:** Diabetes 130-US Hospitals (1999–2008), **101,766 records / 47 features**  
- **Models tried:** Logistic Regression, Random Forest, XGBoost  
- **Imbalance handling:** **SMOTE** and **`scale_pos_weight`**  
- **Priority metric:** **Recall** for the minority class (readmitted)  
- **Deployment:** Flask app for interactive predictions

> ⚕️ **Note:** This is a technical project. It is **not medical advice** and should not be used for clinical decision-making.

---

## 📑 Table of Contents
- [Project Objective & Significance](#project-objective--significance)
- [What’s In/Out of Scope](#whats-inout-of-scope)
- [Data](#data)
- [EDA Highlights](#eda-highlights)
- [Data Prep & Feature Engineering](#data-prep--feature-engineering)
- [Modeling](#modeling)
- [Results](#results)
- [Run Locally](#run-locally)
- [Project Structure](#project-structure)
- [Screenshots / Figures](#screenshots--figures)
- [Contact](#contact)
- [License](#license)

---

## 🎯 Project Objective & Significance

Hospital readmissions are stressful for patients and costly for hospitals. Using demographics, admission/discharge info, labs, and medication histories, this project builds tools to help identify **patients at higher risk of 30-day readmission** and enable earlier interventions.

---

## 🧪 What’s In/Out of Scope

**In scope**
- EDA, data preparation, cleaning
- Feature engineering and statistical tests
- Training and comparing multiple ML models
- Class imbalance mitigation
- Model evaluation and **Flask** deployment

**Out of scope**
- Real-time data ingestion
- Live integration with EHR systems
- Clinical validation or treatment recommendations

---

## 📊 Data

- **Source:** UCI Machine Learning Repository — *Diabetes 130-US Hospitals (1999–2008)*  
  https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- **Records/Features:** 101,766 rows / 47 features
- **Feature types:** demographics (race, sex, age), admission/discharge, labs (A1C, glucose), diabetes medications, outcomes  
- **Target:** `readmitted` (1 = within 30 days, 0 = otherwise)

---

## 🔍 EDA Highlights

- Cleaned and validated feature lists per docs; corrected “unknown” values that Pandas flagged as missing (e.g., A1C, max_glu_serum → “not measured”).
- Outliers retained (e.g., time in hospital, #lab procedures) as they may reflect real complications.
- Scatter plots and correlation heat maps showed **generally weak linear correlations** (max ≈ 0.46 between `time_in_hospital` and `num_medication`).
- Categorical significance: many features had **low p-values** (chi-square) but **weak practical effect sizes** (Cramér’s V), so kept for modeling rather than drop.

---

## 🧱 Data Prep & Feature Engineering

- **Age**: raw ranges → midpoints, then grouped (young/middle-aged/senior).  
- **ICD-9 decoding**: custom function to map codes to disease categories (more interpretable).  
- **New feature**: `time_spent_hospital_with_med = time_in_hospital × num_medication`.  
- **Missing/unknowns**: normalized categories (e.g., admission source “Unknown/Not Mapped”).  
- **Discharge disposition**: 30 raw IDs → 5 interpretable groups.  
- **Encoding & scaling**: `ColumnTransformer` with:
  - Numeric: `SimpleImputer(mean)` → log-transform → `RobustScaler`
  - Categorical: `SimpleImputer(most_frequent)` → `OneHotEncoder(drop="first", handle_unknown="ignore")`

---

## 🤖 Modeling

- **Train/test split:** 80/20
- **Models:** Logistic Regression (baseline), Random Forest, **XGBoost**
- **Imbalance:** **SMOTE** + `scale_pos_weight` (ratio of negatives to positives)
- **Tuning:** `RandomizedSearchCV` and `GridSearchCV` (focus on **F1/Recall** for minority)
- **Final choice:** **XGBoost** with SMOTE + log-transform + class weighting

**Key hyperparameters (final XGBoost):**


---

## ✅ Results

| Model                                   | Accuracy | Recall (minority) | ROC-AUC |
|-----------------------------------------|:--------:|:-----------------:|:-------:|
| Logistic Regression (baseline)          | 0.88     | **0.00**          | –       |
| Random Forest (best variant)            | 0.88     | 0.02–0.03         | –       |
| XGBoost + class weight                  | 0.59     | 0.65              | 0.65    |
| **XGBoost + SMOTE + class weight (final)** | **0.58** | **0.67**          | **0.65** |

> **Interpretation:** Optimizing for **recall** intentionally trades overall accuracy to better **catch high-risk patients** (fewer false negatives), which is often more appropriate in healthcare triage settings.

- Feature importance examined with **SHAP**.

---

## 🏃 Run Locally

### 1) Clone & set up
```bash
git clone https://github.com/slavdefense/ml-diabetic-readmission-30dmittance.git
cd ml-diabetic-readmission-30dmittance
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/train.py
# saves best model artifact e.g. models/xgb_readmission.pkl
export FLASK_APP=app.py
flask run
# or: python app.py

.
├─ README.md
├─ data/                 # (optional) raw/processed data or a download script
├─ notebooks/            # EDA, experiments
├─ src/                  # training/processing code (pipelines, utils)
├─ models/               # saved model artifacts (.pkl)
├─ app.py                # Flask app
├─ requirements.txt
└─ docs/                 # figures, screenshots, report PDF
📬 Contact

Sagun Shrestha
📧 fssagunshrestha@gmail.com
 · 📞 (818)-930-0018 · GitHub: https://github.com/slavdefense
