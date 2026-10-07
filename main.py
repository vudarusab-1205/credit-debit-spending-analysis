
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95
ALPHA = 1 - CONFIDENCE_LEVEL

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(RANDOM_SEED)


def load_transactions():
    """
    Load credit_debit_transactions.csv if available.
    Otherwise, generate reproducible demonstration data.
    """

    csv_path = Path("Credit card transactions - India - Simple.csv.csv")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print("Loaded credit_debit_transactions.csv")
    else:
        print("No CSV found. Generating simulated demo data.")

        n_credit = 1000
        n_debit = 1000

        credit_amounts = np.random.lognormal(
            mean=5.0, sigma=0.65, size=n_credit
        )

        debit_amounts = np.random.lognormal(
            mean=4.7, sigma=0.65, size=n_debit
        )

        df = pd.DataFrame({
            "payment_method": (
                ["Credit"] * n_credit + ["Debit"] * n_debit
            ),
            "amount": np.concatenate([
                credit_amounts, debit_amounts
            ])
        })

   
    df.columns = df.columns.str.strip().str.lower()
    df["payment_method"] = (
        df["payment_method"].astype(str).str.strip().str.title()
    )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    df = df.dropna(subset=["payment_method", "amount"])
    df = df[df["payment_method"].isin(["Credit", "Debit"])]
    df = df[df["amount"] > 0]

    if df["payment_method"].nunique() != 2:
        raise ValueError(
            "The dataset must contain both Credit and Debit transactions."
        )

    return df


df = load_transactions()

credit = df.loc[
    df["payment_method"] == "Credit", "amount"
].to_numpy()

debit = df.loc[
    df["payment_method"] == "Debit", "amount"
].to_numpy()

if len(credit) < 2 or len(debit) < 2:
    raise ValueError("Each group needs at least two valid transactions.")

print("\nDATASET OVERVIEW")
print("-" * 40)
print(f"Total transactions: {len(df)}")
print(f"Credit transactions: {len(credit)}")
print(f"Debit transactions: {len(debit)}")
print(f"Missing values after cleaning: {df.isna().sum().sum()}")


def summarize(values):
    return {
        "Count": len(values),
        "Mean": np.mean(values),
        "Median": np.median(values),
        "Standard deviation": np.std(values, ddof=1),
        "Minimum": np.min(values),
        "Maximum": np.max(values)
    }


summary = pd.DataFrame({
    "Credit": summarize(credit),
    "Debit": summarize(debit)
}).T

print("\nDESCRIPTIVE STATISTICS")
print(summary.round(2))

summary.to_csv(OUTPUT_DIR / "summary_statistics.csv")


mean_credit = np.mean(credit)
mean_debit = np.mean(debit)

var_credit = np.var(credit, ddof=1)
var_debit = np.var(debit, ddof=1)

n_credit = len(credit)
n_debit = len(debit)

standard_error = np.sqrt(
    var_credit / n_credit + var_debit / n_debit
)

t_statistic = (mean_credit - mean_debit) / standard_error

numerator = (
    var_credit / n_credit + var_debit / n_debit
) ** 2

denominator = (
    (var_credit / n_credit) ** 2 / (n_credit - 1)
    + (var_debit / n_debit) ** 2 / (n_debit - 1)
)

degrees_freedom = numerator / denominator

p_value = 2 * stats.t.sf(
    abs(t_statistic), df=degrees_freedom
)

critical_t = stats.t.ppf(
    1 - ALPHA / 2, df=degrees_freedom
)

difference = mean_credit - mean_debit

ci_lower = difference - critical_t * standard_error
ci_upper = difference + critical_t * standard_error

print("\nHYPOTHESIS TEST RESULTS")
print("-" * 40)
print(f"Confidence level: {CONFIDENCE_LEVEL:.0%}")
print(f"Mean difference (Credit - Debit): {difference:.2f}")
print(f"Standard error: {standard_error:.2f}")
print(f"t-statistic: {t_statistic:.4f}")
print(f"Degrees of freedom: {degrees_freedom:.2f}")
print(f"p-value: {p_value:.6g}")
print(f"95% CI for mean difference: [{ci_lower:.2f}, {ci_upper:.2f}]")

if p_value < ALPHA:
    print("\nDecision: Reject the null hypothesis.")
    print("The mean transaction amounts differ significantly.")
else:
    print("\nDecision: Fail to reject the null hypothesis.")
    print("There is insufficient evidence of a mean difference.")


scipy_result = stats.ttest_ind(
    credit,
    debit,
    equal_var=False,
    alternative="two-sided"
)

print("\nSCIPY VERIFICATION")
print(f"Scipy t-statistic: {scipy_result.statistic:.4f}")
print(f"Scipy p-value: {scipy_result.pvalue:.6g}")

assert np.isclose(t_statistic, scipy_result.statistic)
assert np.isclose(p_value, scipy_result.pvalue)


means = [mean_credit, mean_debit]
labels = ["Credit Card", "Debit Card"]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, means)

plt.title("Average Transaction Amount by Payment Method")
plt.ylabel("Transaction amount")
plt.xlabel("Payment method")

for bar, value in zip(bars, means):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "average_spending.png", dpi=150)
plt.show()


plt.figure(figsize=(9, 5))

plt.hist(credit, bins=40, alpha=0.6, label="Credit")
plt.hist(debit, bins=40, alpha=0.6, label="Debit")

plt.title("Transaction Amount Distributions")
plt.xlabel("Transaction amount")
plt.ylabel("Number of transactions")
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "spending_distributions.png", dpi=150)
plt.show()


plt.figure(figsize=(8, 5))

plt.boxplot(
    [credit, debit],
    tick_labels=["Credit", "Debit"],
    showfliers=False
)

plt.title("Spending Variation by Payment Method")
plt.ylabel("Transaction amount")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "spending_boxplot.png", dpi=150)
plt.show()

results = pd.DataFrame([{
    "confidence_level": CONFIDENCE_LEVEL,
    "credit_count": n_credit,
    "debit_count": n_debit,
    "credit_mean": mean_credit,
    "debit_mean": mean_debit,
    "mean_difference": difference,
    "standard_error": standard_error,
    "t_statistic": t_statistic,
    "degrees_freedom": degrees_freedom,
    "p_value": p_value,
    "ci_lower": ci_lower,
    "ci_upper": ci_upper,
    "reject_null": p_value < ALPHA
}])

results.to_csv(OUTPUT_DIR / "hypothesis_test_results.csv", index=False)

print("\nProject completed successfully!")
print(f"Check the '{OUTPUT_DIR}' folder for saved results and charts.")
