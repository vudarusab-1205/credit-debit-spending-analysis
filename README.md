Credit vs. Debit Card Spending Pattern Analysis
Overview

A Python-based statistical analysis project comparing transaction amounts across credit and debit card payments. The project applies hypothesis testing to investigate whether average transaction amounts differ between the two payment methods.

Objectives
1. Compare average and median transaction amounts.
2. Calculate the Welch independent-samples t-statistic using NumPy.
3. Evaluate statistical significance at a 95% confidence level.
4. Estimate a confidence interval for the difference in population means.
5. Visualize spending distributions and variability.


Technologies
1. Python
2. NumPy
3. Pandas
4. Matplotlib
5. SciPy


Methodology
1. Load and clean transaction data.
2. Separate credit and debit transactions.
3. Calculate descriptive statistics.
4. Formulate null and alternative hypotheses.
5. Calculate the Welch t-statistic and degrees of freedom.
6. Calculate the two-sided p-value and 95% confidence interval.
7. Verify the manual calculations with SciPy.
8. Generate visualizations and export results.


Hypotheses

H0: The population mean transaction amounts are equal.

H1: The population mean transaction amounts are different.

Significance level: 0.05.

Results

Add the actual sample sizes, mean transaction amounts, t-statistic, p-value, confidence interval, and statistical conclusion after running the analysis.

Limitations

The initial demonstration dataset is simulated and must not be presented as real customer behavior. Conclusions from real data depend on data quality, sampling, and the assumptions of the statistical test.

Future Improvements
Analyze spending by merchant category and time of day.
Compare transaction medians and use robust statistical methods.
Add effect sizes and power analysis.
Build an interactive spending dashboard.