

import pandas as pd

 
# CARD TRANSACTION VELOCITY
 
def card_transaction_velocity(
    dataframe
):
    velocity = (
        dataframe
        .groupby("card1")["TransactionID"]
        .transform("count")
    )
    dataframe["Card_Transaction_Velocity"] = velocity
 
    print("CARD TRANSACTION VELOCITY CREATED")
 
    return dataframe
 
# ADDRESS TRANSACTION VELOCITY
 
def address_transaction_velocity(
    dataframe
):
    velocity = (
        dataframe
        .groupby("addr1")["TransactionID"]
        .transform("count")
    )
    dataframe["Address_Transaction_Velocity"] = velocity
 
    print("ADDRESS TRANSACTION VELOCITY CREATED")
 
    return dataframe
 
# DEVICE TRANSACTION VELOCITY
 
def device_transaction_velocity(
    dataframe
):
    velocity = (
        dataframe
        .groupby("DeviceInfo")["TransactionID"]
        .transform("count")
    )
    dataframe["Device_Transaction_Velocity"] = velocity
 
    print("DEVICE TRANSACTION VELOCITY CREATED")
 
    return dataframe
 
# TRANSACTION HOUR VELOCITY
 
def hourly_transaction_velocity(
    dataframe
):
    velocity = (
        dataframe
        .groupby("TransactionHour")["TransactionID"]
        .transform("count")
    )
    dataframe["Hourly_Transaction_Velocity"] = velocity
 
    print("HOURLY TRANSACTION VELOCITY CREATED")
 
    return dataframe
 
# CARD AVERAGE AMOUNT
 
def card_average_transaction_amount(
    dataframe
):
    avg_amount = (
        dataframe
        .groupby("card1")["TransactionAmt"]
        .transform("mean")
    )
    dataframe["Card_Avg_TransactionAmt"] = (
        avg_amount.round(2)
    )
 
    print("CARD AVERAGE TRANSACTION AMOUNT CREATED")
 
    return dataframe
 
# AMOUNT DEVIATION FEATURE
 
def transaction_amount_deviation(
    dataframe
):
    dataframe["Transaction_Amount_Deviation"] = (
        dataframe["TransactionAmt"]
        -
        dataframe["Card_Avg_TransactionAmt"]
    ).round(2)
 
    print("TRANSACTION AMOUNT DEVIATION CREATED")
 
    return dataframe
 
# FREQUENCY ENCODING
 
def frequency_encoding(
    dataframe,
    columns
):
    for col in columns:
        frequency = (
            dataframe[col]
            .value_counts()
        )
        dataframe[f"{col}_Frequency"] = (
            dataframe[col]
            .map(frequency)
        )
 
    print("FREQUENCY FEATURES CREATED")
 
    return dataframe
 
# HIGH-RISK HOUR FLAG
 
def high_risk_hour_flag(
    dataframe,
    high_risk_hours=[0, 1, 2, 3, 4]
):
    dataframe["High_Risk_Hour_Flag"] = (
        dataframe["TransactionHour"]
        .isin(high_risk_hours)
        .astype(int)
    )
 
    print("HIGH RISK HOUR FLAG CREATED")
 
    return dataframe
 
# HIGH AMOUNT FLAG
 
def high_amount_flag(
    dataframe,
    threshold=500
):
    dataframe["High_Amount_Flag"] = (
        dataframe["TransactionAmt"] > threshold
    ).astype(int)
 
    print("HIGH AMOUNT FLAG CREATED")
 
    return dataframe
