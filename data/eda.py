
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew


def numeric_summary_analysis(dataframe, columns, analysis_name):

  print(f"{analysis_name} Summary")
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

  print(f"{analysis_name} Fraud Comparison")
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

  outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
  outlier_pct = round((len(outliers) / len(dataframe)) * 100, 2)

  print(f"{column} Outlier Analysis")

  print("Lower Bound:", round(lower_bound, 2))
  print("Upper Bound:", round(upper_bound, 2))

  print("Outlier Count:", len(outliers))
  print("Outlier Percentage:", outlier_pct, "%")

  return outliers


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

  print(f"{column} Percentile Analysis")
  display(percentile_df)

  return percentile_df


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

  print("Skewness Analysis")
  display(skew_df)

  return skew_df


def correlation_analysis(dataframe, columns):

  correlation_matrix = (dataframe[columns].corr().round(2))

  plt.figure(figsize=(12, 8))

  sns.heatmap(correlation_matrix, annot=True, cmap="RdBu")

  plt.title("Correlation Matrix")
  plt.show()

  return correlation_matrix


def fraud_correlation_analysis(dataframe, columns):

  corr = dataframe[columns + ["isFraud"]].corr()

  fraud_corr = (corr["isFraud"].drop("isFraud").sort_values(key=abs,ascending=False).reset_index())

  fraud_corr.columns = ["Feature","Correlation_With_Fraud"]

  print("\nFraud Correlation Analysis")
  display(fraud_corr)

  return fraud_corr


def fraud_lift_analysis(dataframe, columns):

  results = []

  overall_fraud_rate = ( dataframe["isFraud"].mean())

  for col in columns:
    threshold = dataframe[col].median()

    high_group = dataframe[dataframe[col] >= threshold]

    fraud_rate = (high_group["isFraud"].mean())

    lift = fraud_rate / overall_fraud_rate

    results.append({"Feature": col,"Fraud_Lift": round(lift, 2)})

  result_df = pd.DataFrame(results)

  result_df = result_df.sort_values( by="Fraud_Lift", ascending=False)

  print("\nFraud Lift Analysis")
  display(result_df)

  return result_df

#Categorical Columns
def distribution_analysis(dataframe, column_mapping, top_n=20 ):

    for col, label in column_mapping.items():

        print(f"\n{label} Distribution")

        counts = dataframe[col].value_counts(dropna=False).head(top_n)
        percentages = ( dataframe[col].value_counts(normalize=True, dropna=False ).head(top_n) * 100).round(2)

        summary = pd.DataFrame({
            "Count": counts,
            "Percentage": percentages
        })

        display(summary)

    return


def missing_signal_fraud_analysis(df, target_col='isFraud'):

    results = []

    features = [
        'TransactionAmt', 'ProductCD',
        'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2',
        'dist1', 'dist2',
        'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
    ]

    total_rows = len(df)

    for col in features:

        total_fraud = df[target_col].sum()

        missing_mask = df[col].isna()
        present_mask = ~missing_mask

        missing_count = missing_mask.sum()
        present_count = present_mask.sum()

        missing_fraud_transactions = (df.loc[missing_mask, target_col].sum() )

        missing_pct = round((missing_count / total_rows) * 100, 2)

        present_pct = round((present_count / total_rows) * 100, 2)

        fraud_contribution_pct = round((missing_fraud_transactions / total_fraud) * 100, 2 )

        fraud_rate_missing = (df.loc[missing_mask, target_col].mean()if missing_count > 0 else 0)

        fraud_rate_present = (df.loc[present_mask, target_col].mean()if present_count > 0 else 0)

        fraud_rate_missing_pct = (round(fraud_rate_missing * 100, 2)if missing_count > 0 else 0)

        fraud_rate_present_pct = (round(fraud_rate_present * 100, 2)if present_count > 0 else 0)

        fraud_lift = (round(fraud_rate_missing / fraud_rate_present, 2)if (missing_count > 0 and fraud_rate_present > 0)else 0)

        results.append({
            'Feature': col,
            'Missing_Count': missing_count,
            'Present_Count': present_count,
            'Missing_Pct': missing_pct,
            'Present_Pct': present_pct,
            'Missing_Fraud_Transactions': missing_fraud_transactions,
            'Fraud_Contribution_Pct': fraud_contribution_pct,
            'Fraud_Rate_Missing_Pct': fraud_rate_missing_pct,
            'Fraud_Rate_Present_Pct': fraud_rate_present_pct,
            'Fraud_Lift': fraud_lift
        })

    results_df = pd.DataFrame(results).sort_values(by='Fraud_Lift', ascending=False, na_position='last').reset_index(drop=True)

    return results_df
