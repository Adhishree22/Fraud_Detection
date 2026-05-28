
import pandas as pd

# Card Transaction Velocity
def card_transaction_velocity(dataframe):
    velocity = (dataframe.groupby("card1")["TransactionID"].transform("count"))
    dataframe["Card_Transaction_Velocity"] = velocity

    print("Card Transaction Velocity Created")

    return dataframe

# Address Transaction Velocity
def address_transaction_velocity(dataframe):
    velocity = (dataframe.groupby("addr1")["TransactionID"].transform("count"))
    dataframe["Address_Transaction_Velocity"] = velocity

    print("Address Transaction Velocity Created")

    return dataframe

# Device Transaction Velocity
def device_transaction_velocity(dataframe):
    velocity = (dataframe.groupby("DeviceInfo")["TransactionID"].transform("count"))
    dataframe["Device_Transaction_Velocity"] = velocity

    print("Device Transaction Velocity Created")

    return dataframe

# Transaction Hour Velocity
def hourly_transaction_velocity(dataframe):
    velocity = (dataframe.groupby("TransactionHour")["TransactionID"].transform("count"))
    dataframe["Hourly_Transaction_Velocity"] = velocity

    print("Hourly Transaction Velocity Created")

    return dataframe

# Card Average Amount
def card_average_transaction_amount(dataframe):

    avg_amount = (dataframe.groupby("card1")["TransactionAmt"].transform("mean"))
    dataframe["Card_Avg_TransactionAmt"] = (avg_amount.round(2))

    print("Card Average Transaction Amount Created")

    return dataframe

# Amount Deviation Feature
def transaction_amount_deviation(dataframe):

    dataframe["Transaction_Amount_Deviation"] = (dataframe["TransactionAmt"] - dataframe["Card_Avg_TransactionAmt"]).round(2)

    print("Transaction Amount Deviation Created")

    return dataframe

# Frequency Encoding
def frequency_encoding(dataframe, columns):

    for col in columns:
        frequency = (dataframe[col].value_counts())
        dataframe[f"{col}_Frequency"] = (dataframe[col].map(frequency))

    print("Frequency Features Created")

    return dataframe

# High-Risk Hour Flag
def high_risk_hour_flag(dataframe, high_risk_hours=[0, 1, 2, 3, 4]):

    dataframe["High_Risk_Hour_Flag"] = (dataframe["TransactionHour"].isin(high_risk_hours).astype(int))

    print("High Risk Hour Flag Created")

    return dataframe

# High Amount Flag
def high_amount_flag(dataframe, threshold=500 ):

    dataframe["High_Amount_Flag"] = (dataframe["TransactionAmt"] > threshold).astype(int)

    print("High Amount Flag Created")

    return dataframe
