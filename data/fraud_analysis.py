

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Basic Fraud Rate Analysis
def fraud_rate_analysis( dataframe, column_mapping, top_n=20, min_transactions=0):

  fraud_results = {}

  for col, label in column_mapping.items():
    summary = ( dataframe .groupby(col)["isFraud"] .agg([ "count", "sum", "mean" ]) .reset_index() )
    summary = summary.rename(columns={ "count": "Total_Transactions", "sum": "Fraud_Transactions", "mean": "Fraud_Rate" })
    summary["Non_Fraud_Transactions"] = (summary["Total_Transactions"] - summary["Fraud_Transactions"])
    summary["Fraud_Rate_Percent"] = (summary["Fraud_Rate"] * 100).round(2)
    if min_transactions > 0:
      summary = summary[summary["Total_Transactions"] >= min_transactions]
    summary = summary.sort_values(by="Fraud_Rate_Percent", ascending=False)
    summary = summary.head(top_n)

    print(f"{label} Fraud Analysis")
    display(summary)
    fraud_results[col] = summary

  return fraud_results


# Advanced Fraud Segmentation
def advanced_fraud_segmentation( dataframe, column_mapping, min_transactions=50, top_n=20 ):

    overall_fraud_count = dataframe["isFraud"].sum()

    final_results = {}

    for col, label in column_mapping.items():
      
      summary = ( dataframe .groupby(col)["isFraud"] .agg([ "count", "sum", "mean" ]) .reset_index())
      summary = summary.rename(columns={ "count": "Total_Transactions", "sum": "Fraud_Transactions", "mean": "Fraud_Rate"     })
      summary["Non_Fraud_Transactions"] = ( summary["Total_Transactions"] - summary["Fraud_Transactions"]     )
      summary["Fraud_Rate_Percent"] = ( summary["Fraud_Rate"] * 100     ).round(2)
      summary["Fraud_Contribution_Percent"] = ( summary["Fraud_Transactions"] / overall_fraud_count * 100     ).round(2)
      summary = summary[ summary["Total_Transactions"] >= min_transactions     ]
      summary = summary.sort_values( by=[ "Fraud_Contribution_Percent", "Fraud_Rate_Percent" ], ascending=False     )
      summary = summary.head(top_n)

      print(f"{label} Advanced Fraud Segmentation")
      display(summary)

      final_results[col] = summary

    return final_results

# M Column Analysis

def m_column_analysis(dataframe, columns ):

    final_summary = []

    for col in columns:

        summary = (dataframe.groupby(col)["isFraud"].agg(["count", "sum", "mean"]).reset_index())
        summary = summary.rename(columns={"count": "Transactions", "sum": "Fraud_Transactions", "mean": "Fraud_Rate"})
        summary["Fraud_Rate_Percent"] = (summary["Fraud_Rate"] * 100).round(2)
        summary["Feature"] = col
        summary = summary.rename(columns={col: "Feature_Value"})
        summary = summary[["Feature", "Feature_Value", "Transactions", "Fraud_Transactions", "Fraud_Rate_Percent"]]
        summary = summary.sort_values(by="Fraud_Rate_Percent", ascending=False)
        final_summary.append(summary)

    final_summary = pd.concat(final_summary,ignore_index=True     )

    print("M Column Verification Analysis")
    display(final_summary)

    return final_summary

def transaction_hour_fraud_analysis( dataframe ):

    summary = ( dataframe .groupby("TransactionHour")["isFraud"] .agg([ "count","sum", "mean" ]) .reset_index() )
    summary = summary.rename(columns={"count": "Transactions", "sum": "Fraud_Transactions", "mean": "Fraud_Rate"})
    summary["Fraud_Rate_Percent"] = (summary["Fraud_Rate"] * 100).round(2)

    print("Transaction Hour Fraud Analysis")
    display(summary)

    plt.figure(figsize=(12, 6))
    sns.lineplot( data=summary, x="TransactionHour", y="Fraud_Rate_Percent" )
    plt.title("Fraud Rate by Transaction Hour")
    plt.show()

    return summary

# High Risk Entity Detection

def high_risk_entity_detection(dataframe,column,min_transactions=100,top_n=15 ):
  
  summary = (dataframe.groupby(column)["isFraud"].agg(["count","sum","mean"]).reset_index())
  summary = summary.rename(columns={"count": "Transactions","sum": "Fraud_Transactions","mean": "Fraud_Rate"})
  summary["Fraud_Rate_Percent"] = (summary["Fraud_Rate"] * 100).round(2)
  summary = summary[summary["Transactions"]>= min_transactions]
  summary = summary.sort_values(by=["Fraud_Rate_Percent","Fraud_Transactions"],ascending=False)
  summary = summary.head(top_n)

  print(f"High Risk {column} Detection")
  display(summary)

  return summary

def missing_flag_analysis( dataframe, flag_columns ):

    final_summary = []

    for col in flag_columns:
      summary = ( dataframe .groupby(col)["isFraud"] .agg([ "count","sum", "mean" ]) .reset_index() )
      summary = summary.rename(columns={"count": "Transactions", "sum": "Fraud_Transactions", "mean": "Fraud_Rate"})
      summary["Fraud_Rate_Percent"] = (summary["Fraud_Rate"] * 100).round(2)
      summary["Flag"] = col
      final_summary.append(summary)

    final_summary = pd.concat( final_summary, ignore_index=True )

    print("Missing Signal Fraud Analysis")

    display(final_summary)

    return final_summary
