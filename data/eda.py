
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew


def numeric_summary_analysis(dataframe, columns, analysis_name):

    print(f"{analysis_name.upper()} SUMMARY")
    summary = (dataframe[columns].describe().round(2))

    display(summary)

    return summary

def fraud_behavior_summary(dataframe, columns, analysis_name):

    summary_rows = []

    for col in columns:
        non_fraud_mean = round(dataframe[dataframe["isFraud"] == 0][col].mean(), 2)
        fraud_mean = round(dataframe[dataframe["isFraud"] == 1][col].mean(), 2)
        difference = round(fraud_mean - non_fraud_mean, 2)

        summary_rows.append({
            "Feature": col,
            "Non_Fraud_Mean": non_fraud_mean,
            "Fraud_Mean": fraud_mean,
            "Difference": difference
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="Difference", ascending=False)

    print(f"{analysis_name.upper()} FRAUD COMPARISON")

    display(summary_df)

    return summary_df

def distribution_plot(dataframe, column, bins=50):

    plt.figure(figsize=(12, 6))

    sns.histplot(dataframe[column], bins=bins, kde=True)

    plt.title(f"{column} Distribution")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.show()

def fraud_distribution_plot(dataframe, column):

    plt.figure(figsize=(12, 6))

    sns.boxplot(x="isFraud", y=column, data=dataframe)

    plt.title(f"{column} Fraud Comparison")
    plt.xlabel("Fraud Label")
    plt.ylabel(column)

    plt.show()

def outlier_analysis(dataframe, column):

    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)    ]
    outlier_pct = round((len(outliers) / len(dataframe)) * 100, 2)

    print(f"{column.upper()} OUTLIER ANALYSIS")

    print("Lower Bound:", round(lower_bound, 2))
    print("Upper Bound:", round(upper_bound, 2))

    print("Outlier Count:", len(outliers))
    print("Outlier Percentage:", outlier_pct, "%")

    return outliers


def skewness_analysis(dataframe,columns):

    skew_results = []

    for col in columns:
        skew_value = round(skew(dataframe[col].dropna()), 2)
        skew_results.append({
            "Feature": col,
            "Skewness": skew_value
        })

    skew_df = pd.DataFrame(skew_results)
    skew_df = skew_df.sort_values(by="Skewness", ascending=False)

    print("SKEWNESS ANALYSIS")

    display(skew_df)

    return skew_df


def percentile_analysis(dataframe, column):

    percentiles = [ 0.50, 0.75, 0.90, 0.95, 0.99 ]
    percentile_values = {
        f"{int(p * 100)}th Percentile": round(dataframe[column].quantile(p), 2)
        for p in percentiles
    }

    percentile_df = pd.DataFrame({
        "Percentile": percentile_values.keys(),
        "Value": percentile_values.values()
    })

    print(f"{column.upper()} PERCENTILE ANALYSIS")

    display(percentile_df)

    return percentile_df


def correlation_analysis(dataframe, columns):

    correlation_matrix = (dataframe[columns].corr().round(2))

    plt.figure(figsize=(12, 8))

    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")

    plt.title("Correlation Matrix")
    plt.show()

    return correlation_matrix

def fraud_rate_by_bucket(dataframe):

    summary = (dataframe.groupby("Amount_Bucket")["isFraud"].agg(["count", "mean"]).reset_index())

    summary["Fraud_Rate_Percent"] = (summary["mean"] * 100).round(2)
    summary = summary.rename(columns={"count": "Transactions"})
    summary = summary[["Amount_Bucket", "Transactions", "Fraud_Rate_Percent"]]
    summary = summary.sort_values(by="Fraud_Rate_Percent", ascending=False)

    print("FRAUD RATE BY AMOUNT BUCKET")

    display(summary)

    return summary
