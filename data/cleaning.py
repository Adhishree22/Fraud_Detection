
import pandas as pd

def fill_categorical_missing(dataframe, categorical_columns, fill_value="Unknown"):

    for col in categorical_columns:
        dataframe[col] = (dataframe[col].fillna(fill_value))

    print("CATEGORICAL MISSING VALUES FILLED")

    return dataframe


def fill_numeric_missing(dataframe, numeric_columns, fill_value=0):

    for col in numeric_columns:
        dataframe[col] = (dataframe[col].fillna(fill_value))

    print("NUMERIC MISSING VALUES FILLED")

    return dataframe

def fill_address_missing(dataframe):

    dataframe["addr1"] = (dataframe["addr1"].fillna(-1))
    dataframe["addr2"] = (dataframe["addr2"].fillna(-1))

    print("ADDRESS FIELDS CLEANED")

    return dataframe

def standardize_email_domains(dataframe, email_columns):

    for col in email_columns:
        dataframe[col] = (dataframe[col].str.lower().str.strip())

    print("EMAIL DOMAINS STANDARDIZED")

    return dataframe


def clean_identity_columns(dataframe, identity_columns):

    for col in identity_columns:
        dataframe[col] = (dataframe[col].fillna("Unknown").astype(str).str.strip())

    print("IDENTITY COLUMNS STANDARDIZED")

    return dataframe


def generate_cleaning_summary(dataframe):

    summary = pd.DataFrame({
        "Missing_Count": dataframe.isnull().sum(),
        "Missing_Percentage": (
            dataframe.isnull().mean() * 100
        ).round(2)
    })

    summary = summary.sort_values(
        by="Missing_Percentage",
        ascending=False
    )

    print("POST-CLEANING SUMMARY")

    display(summary)

    return summary
