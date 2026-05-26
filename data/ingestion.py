
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

def load_transaction_data(file_path, selected_columns):

    dataframe = pd.read_csv(
        file_path,
        usecols=selected_columns
    )

    return dataframe

def load_identity_data(file_path, selected_columns):

    dataframe = pd.read_csv(
        file_path,
        usecols=selected_columns
    )

    return dataframe

def create_transaction_sample(
    dataframe,
    sample_size=100000,
    random_state=42
):

    sample_df = dataframe.sample(
        n=sample_size,
        random_state=random_state
    )

    return sample_df

def filter_identity_sample(
    identity_df,
    transaction_sample_df
):

    filtered_df = identity_df[
        identity_df["TransactionID"].isin(
            transaction_sample_df["TransactionID"]
        )
    ].copy()

    return filtered_df

def remove_duplicates(dataframe, subset_column):

    dataframe = dataframe.drop_duplicates(
        subset=subset_column
    )

    return dataframe


def save_dataframe(dataframe, output_path):

    dataframe.to_csv(
        output_path,
        index=False
    )

    print(f"Dataset saved successfully: {output_path}")
