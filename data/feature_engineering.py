
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


def create_transaction_amount_zscore(dataframe):

    dataframe = dataframe.copy()

    amount_mean = dataframe["TransactionAmt"].mean()

    amount_std = dataframe["TransactionAmt"].std()

    dataframe["TransactionAmt_ZScore"] = ((dataframe["TransactionAmt"] - amount_mean) / amount_std).round(2)

    print("\nTransaction Amount ZScore Created")
    print(dataframe["TransactionAmt_ZScore"].value_counts().sort_index())

    return dataframe


def create_card_fraud_rate_feature(dataframe):

    dataframe = dataframe.copy()

    fraud_rate_by_card = (dataframe.groupby("card1")["isFraud"].mean())

    dataframe["Card_Historical_Fraud_Rate"] = (dataframe["card1"].map(fraud_rate_by_card) * 100).round(2)

    print("\nCard Historical Fraud Rate Created")
    print(dataframe["Card_Historical_Fraud_Rate"].value_counts().sort_index())

    return dataframe


def create_transaction_time_category(dataframe):

    dataframe = dataframe.copy()

    def categorize_hour(hour):
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        elif 21 <= hour <= 23:
            return "Night"
        else:
            return "Late Night"

    dataframe["Transaction_Time_Category"] = (dataframe["TransactionHour"].apply(categorize_hour))

    print("\nTime Category Created")
    print(dataframe["Transaction_Time_Category"].value_counts().sort_index())

    return dataframe
