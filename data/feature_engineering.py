
import pandas as pd


def create_amount_buckets(dataframe):

    dataframe["Amount_Bucket"] = pd.cut(
        dataframe["TransactionAmt"],
        bins=[0, 50, 100, 250, 500, 1000, 5000, 50000],
        labels=[ "0-50", "50-100", "100-250", "250-500", "500-1000", "1000-5000", "5000+" ])

    print("Amount Bucket Feature Created")
    print(dataframe["Amount_Bucket"].value_counts())

    return dataframe


def create_transaction_hour(dataframe):

    dataframe["TransactionHour"] = (dataframe["TransactionDT"] // 3600) % 24

    print("\nTransaction Hour Feature Created")
    print(dataframe["TransactionHour"].value_counts().sort_index())

    return dataframe
