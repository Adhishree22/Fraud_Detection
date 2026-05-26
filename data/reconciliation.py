
import pandas as pd

def merge_datasets(transactions_df, identity_df):

    merged_df = transactions_df.merge(identity_df, on="TransactionID", how="left")

    print("DATASET MERGE COMPLETED")
    print("Merged Shape:", merged_df.shape)

    return merged_df

def reconciliation_summary(dataframe):

    identity_columns = ["DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"]

    missing_identity = (dataframe[identity_columns].isnull().all(axis=1).sum())
    missing_pct = round((missing_identity / len(dataframe)) * 100, 2)

    print("RECONCILIATION SUMMARY")
    print("Fully Missing Identity Rows:", missing_identity)
    print("Missing Identity Percentage:", missing_pct, "%")


def field_reconciliation_summary(dataframe):

    reconciliation_columns = {
    "DeviceType": "Device Type",
    "DeviceInfo": "Device Info",
    "id_30": "Operating System",
    "id_31": "Browser",
    "id_33": "Screen Resolution"
    }

    reconciliation_results = []

    for col, label in reconciliation_columns.items():
        missing_count = dataframe[col].isnull().sum()
        missing_pct = round((missing_count / len(dataframe)) * 100, 2)

        reconciliation_results.append({
            "Field": label,
            "Missing_Count": missing_count,
            "Missing_Percentage": missing_pct
        })

    reconciliation_df = pd.DataFrame(reconciliation_results)

    print("FIELD-LEVEL RECONCILIATION")
    display(reconciliation_df)

    return reconciliation_df


def create_missing_flags(dataframe):

    missing_flag_columns = {
    "DeviceType": "Missing_DeviceType_Flag",
    "DeviceInfo": "Missing_DeviceInfo_Flag",
    "id_30": "Missing_OS_Flag",
    "id_31": "Missing_Browser_Flag",
    "id_33": "Missing_Resolution_Flag"
}

    for original_col, flag_col in missing_flag_columns.items():
        dataframe[flag_col] = (dataframe[original_col].isnull().astype(int))

    print("MISSING FLAGS CREATED")
    display(dataframe[list(missing_flag_columns.values())].head())

    return dataframe

def identity_completeness_score(dataframe):

    identity_columns = ["DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"]

    dataframe["Identity_Completeness_Score"] = (dataframe[identity_columns].notnull().sum(axis=1))

    print("IDENTITY COMPLETENESS SCORE")
    print(dataframe["Identity_Completeness_Score"].value_counts().sort_index())

    return dataframe


def classify_identity_status(dataframe):

    def classify_identity(score):
        if score == 0:
            return "Fully Missing Identity"
        elif score < 5:
            return "Partial Identity"
        else:
            return "Complete Identity"

    dataframe["Identity_Status"] = (dataframe["Identity_Completeness_Score"].apply(classify_identity))

    print("IDENTITY STATUS DISTRIBUTION")
    print(dataframe["Identity_Status"].value_counts())

    return dataframe


def fraud_reconciliation_analysis(dataframe):

    summary = (dataframe.groupby("Identity_Status")["isFraud"].agg(["count", "mean"]).reset_index())
    summary["Fraud_Rate_Percent"] = (summary["mean"] * 100).round(2)

    summary = summary.rename(columns={
        "count": "Transactions"
    })

    summary = summary[["Identity_Status", "Transactions", "Fraud_Rate_Percent"]]
    summary = summary.sort_values(by="Fraud_Rate_Percent", ascending=False)

    print("FRAUD RECONCILIATION ANALYSIS")
    display(summary)

    return summary
