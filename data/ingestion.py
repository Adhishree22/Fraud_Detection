
import pandas as pd
import numpy as np


def load_dataset(file_path, selected_columns):

    dataframe = pd.read_csv(file_path, usecols=selected_columns)

    return dataframe

def create_sample(transactions, identity, sample_size=100000, random_state=42):

    transactions_sample = transactions.sample(n=sample_size, random_state=random_state    )

    identity_sample = identity[identity["TransactionID"].isin(transactions_sample["TransactionID"])].copy()

    print("\nSampled Transactions:", transactions_sample.shape)
    print("Filtered Identity:", identity_sample.shape)

    return transactions_sample, identity_sample

def remove_duplicates(dataframe, subset_column):

    dataframe = dataframe.drop_duplicates(subset=subset_column)

    return dataframe
