import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_credit_data(num_samples=1000, random_seed=42):
    rng = np.random.RandomState(random_seed)
    
    # 1. Generate core demographic features
    loan_ids = [f"LP{i:06d}" for i in range(1001, 1001 + num_samples)]
    
    genders = rng.choice(["Male", "Female"], size=num_samples, p=[0.78, 0.22])
    married = rng.choice(["Yes", "No"], size=num_samples, p=[0.65, 0.35])
    dependents = rng.choice(["0", "1", "2", "3+"], size=num_samples, p=[0.58, 0.17, 0.15, 0.10])
    education = rng.choice(["Graduate", "Not Graduate"], size=num_samples, p=[0.78, 0.22])
    self_employed = rng.choice(["Yes", "No"], size=num_samples, p=[0.14, 0.86])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], size=num_samples, p=[0.33, 0.38, 0.29])
    
    # 2. Generate financial features with realistic distributions (log-normal for incomes)
    # Income in USD (monthly)
    applicant_income = rng.lognormal(mean=8.4, sigma=0.5, size=num_samples).astype(int)
    # Ensure income is in a realistic range ($1,500 to $25,000)
    applicant_income = np.clip(applicant_income, 1500, 25000)
    
    # Coapplicant income: 45% have coapplicants
    has_coapplicant = rng.choice([True, False], size=num_samples, p=[0.45, 0.55])
    coapplicant_income = np.zeros(num_samples)
    coapplicant_income[has_coapplicant] = rng.lognormal(mean=7.5, sigma=0.6, size=sum(has_coapplicant)).astype(int)
    coapplicant_income = np.clip(coapplicant_income, 0, 15000).astype(int)
    
    total_income = applicant_income + coapplicant_income
    
    # Loan amount (in thousands of dollars)
    # Loan amount is typically correlated with total income
    base_loan = total_income * rng.normal(loc=0.025, scale=0.005, size=num_samples)
    loan_amount = np.clip(base_loan, 15, 600).astype(int)
    
    # Loan term in months
    loan_term = rng.choice([120, 180, 240, 360], size=num_samples, p=[0.02, 0.06, 0.04, 0.88])
    
    # Credit history: 1 = Good, 0 = Bad. 85% have good credit history.
    credit_history = rng.choice([1.0, 0.0], size=num_samples, p=[0.85, 0.15])
    
    # 3. Define target variable (Default) based on features
    # Log odds of defaulting
    # High credit history score (1.0) reduces default odds drastically
    # High loan-to-income ratio increases default odds
    # Low income increases default odds
    # Rural areas have slightly higher defaults
    # Not Graduate has slightly higher defaults
    # Self-employed has slightly higher defaults
    # Calculate annualized Loan-to-Income (LTI) ratio
    lti = (loan_amount * 1000) / (total_income * 12)
    
    log_odds = (
        -1.8  # baseline for Married=Yes, Graduate, Semiurban, Credit_History=1
        - 3.5 * (credit_history - 0.85)  # If credit history is 0, this adds +2.975. If 1, it subtracts -0.525.
        + 1.5 * (lti - 2.0)  # LTI above 2.0 adds risk.
        + 0.5 * (self_employed == "Yes")
        + 0.4 * (education == "Not Graduate")
        + 0.3 * (property_area == "Rural")
        - 0.3 * (property_area == "Semiurban")
        + 0.2 * (dependents == "3+")
        + 0.3 * (married == "No")
    )
    
    p_default = 1 / (1 + np.exp(-log_odds))
    default = rng.binomial(1, p_default)
    
    # Create DataFrame
    df = pd.DataFrame({
        "Loan_ID": loan_ids,
        "Gender": genders,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Default": default
    })
    
    # 4. Introduce artificial missing values (NaNs) to simulate real-world data issues
    missing_configs = [
        ("Gender", 0.02),
        ("Married", 0.01),
        ("Dependents", 0.03),
        ("Self_Employed", 0.05),
        ("LoanAmount", 0.04),
        ("Loan_Amount_Term", 0.02),
        ("Credit_History", 0.08)
    ]
    
    for col, rate in missing_configs:
        mask = rng.rand(num_samples) < rate
        df.loc[mask, col] = np.nan
        
    # 5. Inject a few duplicates (e.g., 6 rows)
    duplicate_count = min(6, num_samples)
    duplicate_indices = rng.choice(num_samples, size=duplicate_count, replace=False)
    duplicates = df.iloc[duplicate_indices].copy()
    # Change Loan_ID slightly or keep same to detect absolute row duplicates
    # We will keep absolute duplicates to show df.drop_duplicates() working
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a reproducible synthetic loan default dataset.")
    parser.add_argument("--samples", type=int, default=1200, help="Number of base loan applications to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible output.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "loan_data.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = generate_credit_data(num_samples=args.samples, random_seed=args.seed)
    output_path = args.output
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated and saved to: {output_path}")
    print(f"Dataset Shape: {df.shape}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print(f"Duplicates Count: {df.duplicated().sum()}")
    print(f"Default rate: {df['Default'].mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
