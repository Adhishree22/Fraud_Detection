

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def m_column_analysis(dataframe, columns):

    total_fraud = dataframe["isFraud"].sum()

    final_summary = []

    for col in columns:

        summary = (dataframe.groupby(col)["isFraud"].agg(["count", "sum", "mean"]).reset_index())
        summary = summary.rename(columns={"count": "Transactions","sum": "Fraud_Transactions","mean": "Fraud_Rate"})
        summary["Fraud_Rate_Percent"] = (summary["Fraud_Rate"] * 100).round(2)
        summary["Fraud_Contribution_Percent"] = (summary["Fraud_Transactions"] / total_fraud * 100).round(2)
        summary["Feature"] = col
        summary = summary.rename(columns={col: "Feature_Value"})
        summary = summary[["Feature", "Feature_Value", "Transactions", "Fraud_Transactions", "Fraud_Rate_Percent", "Fraud_Contribution_Percent"]]
        summary = summary.sort_values(by=["Fraud_Contribution_Percent", "Fraud_Rate_Percent"],ascending=False)
        final_summary.append(summary)

    final_summary = pd.concat(final_summary,ignore_index=True)

    print("M Column Verification Analysis")
    display(final_summary)

    return final_summary


def fraud_concentration_analysis(dataframe, column):

    summary = (dataframe.groupby(column)["isFraud"].agg(["count", "sum", "mean"]).reset_index())
    summary = summary.rename(columns={"count": "Transactions","sum": "Fraud_Transactions","mean": "Fraud_Rate"})
    summary["Fraud_Rate_Percent"] = (summary["Fraud_Rate"] * 100).round(2)
    total_fraud = summary["Fraud_Transactions"].sum()
    summary["Fraud_Contribution_Percent"] = (summary["Fraud_Transactions"] /total_fraud * 100).round(2)
    summary = summary.sort_values(by="Fraud_Contribution_Percent",ascending=False)
    summary["Cumulative_Fraud_Contribution_Pct"] = (summary["Fraud_Contribution_Percent"].cumsum().round(2))

    print(f"{column} Fraud Concentration Analysis")
    display(summary)

    return summary
