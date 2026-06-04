
import pandas as pd
import numpy as np

def dataset_overview(dataframe, dataset_name):

    print(f"\n{'='*60}")
    print(f"{dataset_name} Overview")
    print(f"{'='*60}")

    print("Rows:", dataframe.shape[0])
    print("Columns:", dataframe.shape[1])

    display(dataframe.head())

    return pd.DataFrame({
        "Dataset": [dataset_name],
        "Rows": [dataframe.shape[0]],
        "Columns": [dataframe.shape[1]]
    })


def dataset_summary(dataframe, dataset_name):

    summary = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Missing Cells",
            "Duplicate Rows"
        ],
        "Value": [
            dataframe.shape[0],
            dataframe.shape[1],
            dataframe.isnull().sum().sum(),
            dataframe.duplicated().sum()
        ]
    })

    print(f"\n{dataset_name} Summary")

    display(summary)

    return summary


def schema_validation(dataframe, dataset_name):

    print(f"\n{dataset_name} Schema Validation")

    schema_df = pd.DataFrame({
        "Column": dataframe.columns,
        "DataType": dataframe.dtypes.astype(str),
        "Missing_Count": dataframe.isnull().sum().values,
        "Missing_Percentage": (
            dataframe.isnull().mean() * 100
        ).round(2).values
    })

    schema_df = schema_df.sort_values(
        by="Missing_Percentage",
        ascending=False
    )

    display(schema_df)

    return schema_df


def missing_value_analysis(dataframe):

    missing_df = pd.DataFrame({
        "Column": dataframe.columns,
        "Missing_Count": dataframe.isnull().sum().values,
        "Missing_Percentage": (
            dataframe.isnull().mean() * 100
        ).round(2).values
    })

    missing_df = missing_df[
        missing_df["Missing_Count"] > 0
    ]

    missing_df = missing_df.sort_values(
        by="Missing_Percentage",
        ascending=False
    )

    print("\nMissing Value Analysis")

    display(missing_df)

    return missing_df


def duplicate_validation(
    dataframe,
    subset_column,
    dataset_name
):

    duplicate_count = dataframe.duplicated(
        subset=subset_column
    ).sum()

    result = pd.DataFrame({
        "Dataset": [dataset_name],
        "Duplicate_Count": [duplicate_count]
    })

    print(f"\n{dataset_name} Duplicate Validation")

    display(result)

    return result


def fraud_distribution(dataframe):

    summary = (
        dataframe["isFraud"]
        .value_counts()
        .reset_index()
    )

    summary.columns = [
        "Fraud_Label",
        "Count"
    ]

    summary["Percentage"] = (
        summary["Count"]
        / len(dataframe)
        * 100
    ).round(2)

    print("\nFraud Distribution")

    display(summary)

    return summary


def numeric_overview(dataframe):

    summary = (
        dataframe
        .describe()
        .transpose()
        .round(2)
    )

    print("\nNumeric Feature Overview")

    display(summary)

    return summary

def cardinality_overview(dataframe):

    cardinality = pd.DataFrame({"Column": dataframe.columns,"Unique_Values": [dataframe[col].nunique(dropna=True) for col in dataframe.columns]})

    cardinality = cardinality.sort_values(by="Unique_Values",ascending=False)

    print("Cardinality Overview")
    display(cardinality)

    return cardinality
