

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def fraud_rate_analysis( dataframe, column_mapping, top_n=20 ):
  
  fraud_results = {}
  
  for col, label in column_mapping.items():
    summary = ( dataframe .groupby(col)["isFraud"] .agg([ "count", "mean" ]) .reset_index() )
    summary["Fraud_Rate_Percent"] = ( summary["mean"] * 100 ).round(2)
    summary = summary.rename(columns={ "count": "Transactions" })
    summary = summary[[ col, "Transactions", "Fraud_Rate_Percent" ]]
    summary = summary.sort_values( by="Fraud_Rate_Percent", ascending=False )
    summary = summary.head(top_n)
    
    print(f"{label.upper()} FRAUD ANALYSIS")
    display(summary)
    fraud_results[col] = summary
    
  return fraud_results

def m_column_analysis( dataframe, columns ):

    final_summary = []

    for col in columns:
      summary = ( dataframe .groupby(col)["isFraud"] .agg([ "count", "mean" ]) .reset_index() )
      summary["Fraud_Rate_Percent"] = ( summary["mean"] * 100 ).round(2)
      summary["Feature"] = col
      summary = summary.rename(columns={ col: "Feature_Value", "count": "Transactions" })
      summary = summary[[ "Feature", "Feature_Value", "Transactions", "Fraud_Rate_Percent" ]]
      summary = summary.sort_values( by="Fraud_Rate_Percent", ascending=False )
      final_summary.append(summary)

    final_summary = pd.concat( final_summary, ignore_index=True )

    print("M COLUMN VERIFICATION ANALYSIS")

    display(final_summary)

    return final_summary

def transaction_hour_fraud_analysis( dataframe ):

    summary = ( dataframe .groupby("TransactionHour")["isFraud"] .agg([ "count", "mean" ]) .reset_index() )
    summary["Fraud_Rate_Percent"] = ( summary["mean"] * 100 ).round(2)
    summary = summary.rename(columns={ "count": "Transactions" })

    print("TRANSACTION HOUR FRAUD ANALYSIS")
    display(summary)

    plt.figure(figsize=(12, 6))
    sns.lineplot( data=summary, x="TransactionHour", y="Fraud_Rate_Percent" )
    plt.title("Fraud Rate by Transaction Hour")
    plt.show()

    return summary


def missing_flag_analysis( dataframe, flag_columns ):

    final_summary = []

    for col in flag_columns:
      summary = ( dataframe .groupby(col)["isFraud"] .agg([ "count", "mean" ]) .reset_index() )
      summary["Fraud_Rate_Percent"] = ( summary["mean"] * 100 ).round(2)
      summary["Flag"] = col
      summary = summary.rename(columns={ "count": "Transactions" })
      final_summary.append(summary)

    final_summary = pd.concat( final_summary, ignore_index=True )

    print("MISSING SIGNAL FRAUD ANALYSIS")

    display(final_summary)

    return final_summary
