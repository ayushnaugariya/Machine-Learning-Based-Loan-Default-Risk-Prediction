import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df_cleaned = df.drop_duplicates().copy()

    for col in ["LoanAmount", "Loan_Amount_Term"]:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    for col in ["Credit_History", "Gender", "Married", "Dependents", "Self_Employed"]:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

    return df_cleaned


def engineer_features(df):
    df_fe = df.copy()
    df_fe["TotalIncome"] = df_fe["ApplicantIncome"] + df_fe["CoapplicantIncome"]
    df_fe["LoanIncomeRatio"] = (df_fe["LoanAmount"] * 1000) / (df_fe["TotalIncome"] * 12)
    df_fe["EMI"] = (df_fe["LoanAmount"] * 1000) / df_fe["Loan_Amount_Term"]
    df_fe["EMI_Ratio"] = df_fe["EMI"] / df_fe["TotalIncome"]

    credit_risk_flag = (1 - df_fe["Credit_History"]) * 5
    lti_risk_flag = (df_fe["LoanIncomeRatio"] > 2.5).astype(int) * 3
    edu_risk_flag = (df_fe["Education"] == "Not Graduate").astype(int) * 2
    df_fe["RiskScore"] = credit_risk_flag + lti_risk_flag + edu_risk_flag

    return df_fe


def prepare_features(df):
    df_ml = df.drop(columns=["Loan_ID"]).copy()
    df_ml["Gender"] = df_ml["Gender"].map({"Male": 1, "Female": 0})
    df_ml["Married"] = df_ml["Married"].map({"Yes": 1, "No": 0})
    df_ml["Education"] = df_ml["Education"].map({"Graduate": 1, "Not Graduate": 0})
    df_ml["Self_Employed"] = df_ml["Self_Employed"].map({"Yes": 1, "No": 0})
    df_ml["Dependents"] = df_ml["Dependents"].map({"0": 0, "1": 1, "2": 2, "3+": 3})
    df_ml = pd.get_dummies(df_ml, columns=["Property_Area"], drop_first=True)

    bool_cols = df_ml.select_dtypes(include="bool").columns
    df_ml[bool_cols] = df_ml[bool_cols].astype(int)

    X = df_ml.drop(columns=["Default"])
    y = df_ml["Default"]
    return X, y


def split_and_scale(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    num_cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "TotalIncome",
        "LoanIncomeRatio",
        "EMI",
        "EMI_Ratio",
        "RiskScore",
    ]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def train_models(X_train, X_test, y_train, y_test, seed):
    models = {
        "Logistic Regression": LogisticRegression(random_state=seed, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=seed, max_depth=5, min_samples_leaf=5),
        "Random Forest": RandomForestClassifier(
            random_state=seed,
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=4,
        ),
    }

    rows = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({"model": model_name, **metrics})

    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare loan default classifiers.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "loan_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "model_metrics.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_data(args.data)
    df_cleaned = clean_data(df)
    df_features = engineer_features(df_cleaned)
    X, y = prepare_features(df_features)
    X_train, X_test, y_train, y_test = split_and_scale(X, y, args.seed)

    results = train_models(X_train, X_test, y_train, y_test, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    print(results.round(4).to_string(index=False))
    print(f"\nSaved metrics to: {args.output}")


if __name__ == "__main__":
    main()
