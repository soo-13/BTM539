##### Online Retail II Data Analysis on UCI Machine Learning Repository #####

import datetime as dt
import os
import pandas as pd
from tabulate import tabulate

### Study the dataset
# read excel dataset with two sheets 
df0910 = pd.read_excel(os.path.join("data", "online_retail_II.xlsx"), sheet_name=0) # 2009-2010
df1011 = pd.read_excel(os.path.join("data", "online_retail_II.xlsx"), sheet_name=1) # 2010-2011
df = pd.concat([df0910, df1011], axis=0) # merge data 
df.to_csv(path_or_buf=os.path.join("data", "online_retail_II.csv")) # save the df as a csv file

### Data preprocessing 1
df = pd.read_csv(os.path.join("data", "online_retail_II.csv")) # read the csv file
print(df.dtypes) # checking data types
df = df.astype({"Customer ID": "float64"}) # convert customer id into dtype float64
print(df.isna().sum()) # checking missing values for each variable
# sub-setting rows with NaN
df_nan = df[df.isna().sum(axis=1)>0] 
df_val = df[df.isna().sum(axis=1)==0] 
print(df_val.isna().sum()) # removing missing values and checking if all NaNs are removed
print(df_val.shape) # check 824,364 observations
df_val = df_val.astype({"Customer ID": "Int32"}) # change customer ID dtypes to Int32

### Data preprocessing 2
df_val['InvoiceDate'] = pd.to_datetime(df_val['InvoiceDate'])# convert the invoice date and time varables (from string to datetime64)
# extract year, month, day, hour, minute, second, and weekday
df_val['Year'] = df_val['InvoiceDate'].dt.year
df_val['Month'] = df_val['InvoiceDate'].dt.month
df_val['Day'] = df_val['InvoiceDate'].dt.day
df_val['Hour'] = df_val['InvoiceDate'].dt.hour
df_val['Minute'] = df_val['InvoiceDate'].dt.minute
df_val['Second'] = df_val['InvoiceDate'].dt.second
df_val['Weekday'] = df_val['InvoiceDate'].dt.weekday # 0 Monday - 6 Sunday
df_val = df_val.astype({"Year": "Int32", "Month": "Int32", "Day": "Int32", "Hour": "Int32", "Minute": "Int32", "Second": "Int32", "Weekday": "Int32"})
print(df_val.dtypes)
print(df_val.head())
# creating a variable to measure recency of transactions
present = dt.datetime(2011,12,31)
df_val['Recency'] = present - df_val.groupby(['Customer ID'])['InvoiceDate'].transform(max)

### Data preprocessing 3
df_val.describe() # summary statistics

# clean negative quantities and zero price
