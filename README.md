# Loan Default Risk Prediction

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-scikit--learn-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An end-to-end credit risk analytics project that predicts loan default probability from applicant demographics, income, loan details, and credit history. The project includes synthetic data generation, exploratory analysis, feature engineering, model comparison, exported charts, and a notebook validation script.

## Project Highlights

- Builds a realistic synthetic lending dataset with missing values and duplicate rows for cleaning practice.
- Performs EDA on default rates, income, loan amount, education, credit history, and feature correlations.
- Engineers credit-risk features such as `TotalIncome`, `LoanIncomeRatio`, `EMI_Ratio`, and `RiskScore`.
- Compares Logistic Regression, Decision Tree, and Random Forest classifiers.
- Saves all major visualizations to `images/`.
- Includes scripts to regenerate data, train models, rebuild the notebook, and validate notebook code cells from the terminal.

## Project Structure

```text
Loan Defaulter/
|-- data/
|   `-- loan_data.csv
|-- images/
|   |-- applicant_income_distribution.png
|   |-- confusion_matrices.png
|   |-- correlation_matrix.png
|   |-- credit_history_vs_default.png
|   |-- default_distribution.png
|   |-- education_vs_default.png
|   |-- feature_importance.png
|   |-- income_vs_loan_amount.png
|   |-- loan_amount_distribution.png
|   `-- roc_curves.png
|-- notebooks/
|   `-- Loan_Default_Prediction.ipynb
|-- reports/
|   `-- model_metrics.csv
|-- src/
|   |-- build_notebook.py
|   |-- generate_data.py
|   |-- train_model.py
|   `-- verify_notebook.py
|-- LICENSE
|-- Loan_Default_Prediction.ipynb
|-- README.md
`-- requirements.txt
```

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Regenerate the project artifacts:

```bash
python src/generate_data.py
python src/train_model.py
python src/build_notebook.py
python src/verify_notebook.py
```

Optional data generation arguments:

```bash
python src/generate_data.py --samples 1200 --seed 42 --output data/loan_data.csv
```

Save model metrics to a custom location:

```bash
python src/train_model.py --output reports/model_metrics.csv
```

## Workflow

1. **Data generation** creates a reproducible synthetic credit dataset with realistic distributions.
2. **Data cleaning** removes duplicates and imputes missing categorical and numeric fields.
3. **EDA** explores class balance and key risk drivers.
4. **Feature engineering** creates income, leverage, repayment burden, and composite risk indicators.
5. **Preprocessing** encodes categorical variables, creates train/test splits, and scales numeric fields.
6. **Modeling** trains Logistic Regression, Decision Tree, and Random Forest models.
7. **Evaluation** compares Accuracy, Precision, Recall, F1-score, and ROC-AUC.

## Key Features

| Feature | Description |
| :--- | :--- |
| `TotalIncome` | Applicant plus co-applicant monthly income |
| `LoanIncomeRatio` | Loan amount divided by annual household income |
| `EMI` | Estimated monthly installment |
| `EMI_Ratio` | Monthly installment divided by household income |
| `RiskScore` | Composite rule-based score using credit history, leverage, and education |

## Model Summary

The notebook benchmarks three models:

| Model | Strength |
| :--- | :--- |
| Logistic Regression | Interpretable baseline often preferred in regulated credit environments |
| Decision Tree | Transparent nonlinear rules |
| Random Forest | Stronger ensemble model for nonlinear interactions |

The strongest model can vary slightly by generated sample and split. With the default seed, Logistic Regression and Random Forest both perform competitively, with ROC-AUC around `0.79` to `0.80`.

## Business Interpretation

The analysis supports practical credit policy ideas:

- Treat weak credit history as the strongest risk signal.
- Monitor loan-to-income and EMI-to-income ratios to avoid over-leveraging borrowers.
- Use model scores for risk-based pricing, underwriting review queues, or early warning systems.
- Retrain and monitor the model periodically because borrower behavior and macroeconomic conditions can drift.

## Notes

This is a portfolio and learning project. The dataset is synthetic and should not be used for real lending decisions without real-world validation, compliance review, fairness analysis, and production-grade model monitoring.
