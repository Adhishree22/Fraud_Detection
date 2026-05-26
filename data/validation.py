
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

def dataset_overview(dataframe, dataset_name):

    print("\n")
    print("=" * 80)
    print(f"{dataset_name.upper()} OVERVIEW")
    print("=" * 80)

    print("Shape:", dataframe.shape)

    display(dataframe.head())

def schema_validation(dataframe, dataset_name):

    print("\n")
    print("=" * 80)
    print(f"{dataset_name.upper()} SCHEMA VALIDATION")
    print("=" * 80)

    schema_df = pd.DataFrame({

        "Column": dataframe.columns,

        "DataType": dataframe.dtypes.astype(str),

        "Missing_Count": dataframe.isnull().sum().values,

        "Missing_Percentage": (
            dataframe.isnull().mean() * 100
        ).round(2).values
    })

    display(schema_df)

    return schema_df


def duplicate_validation(
    dataframe,
    subset_column,
    dataset_name
):

    duplicate_count = dataframe.duplicated(
        subset=subset_column
    ).sum()

    print("\n")
    print("=" * 80)
    print(f"{dataset_name.upper()} DUPLICATE VALIDATION")
    print("=" * 80)

    print(f"Duplicate Count: {duplicate_count}")

    return duplicate_count

def ingestion_summary(
    transactions_df,
    identity_df
):

    summary_df = pd.DataFrame({

        "Dataset": [
            "Transactions",
            "Identity"
        ],

        "Rows": [
            len(transactions_df),
            len(identity_df)
        ],

        "Columns": [
            transactions_df.shape[1],
            identity_df.shape[1]
        ]
    })

    print("\n")
    print("=" * 80)
    print("INGESTION QUALITY SUMMARY")
    print("=" * 80)

    display(summary_df)

    return summary_df
