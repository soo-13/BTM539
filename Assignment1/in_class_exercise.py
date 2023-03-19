##### Online Retail II Data Analysis on UCI Machine Learning Repository #####

import datetime as dt
import matplotlib.pyplot as plt
import os
import pandas as pd

### Study the dataset
# read excel dataset with two sheets 
'''
df0910 = pd.read_excel(os.path.join("data", "online_retail_II.xlsx"), sheet_name=0) # 2009-2010
df1011 = pd.read_excel(os.path.join("data", "online_retail_II.xlsx"), sheet_name=1) # 2010-2011
df = pd.concat([df0910, df1011], axis=0) # merge data 
df.to_csv(path_or_buf=os.path.join("data", "online_retail_II.csv")) # save the df as a csv file
'''
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
df_val['Month'] = df_val['InvoiceDate'].dt.month_name()
df_val['Day'] = df_val['InvoiceDate'].dt.day
df_val['Hour'] = df_val['InvoiceDate'].dt.hour
df_val['Minute'] = df_val['InvoiceDate'].dt.minute
df_val['Second'] = df_val['InvoiceDate'].dt.second
df_val['Weekday'] = df_val['InvoiceDate'].dt.strftime("%A") 
df_val = df_val.astype({"Year": "Int32", "Day": "Int32", "Hour": "Int32", "Minute": "Int32", "Second": "Int32"})
# creating a variable to measure recency of transactions
present = dt.datetime(2012,1,1)
df_val['Recency'] = (present - df_val.groupby(['Customer ID'])['InvoiceDate'].transform(max)).dt.days

### Data preprocessing 3
print(df_val.describe()) # summary statistics
df_val.drop(df_val[df_val['Quantity'] < 0].index, inplace=True) # clean negative quantities 
df_val.drop(df_val[df_val['Price'] == 0].index, inplace=True)  # clean zero price
df_val.to_csv(path_or_buf=os.path.join("/Users", "yeonsoo", "Desktop", "YS", "2023spring", "BTM539", "HW", "Assignment2", "data", "online_retail_II_preprocessed.csv"), index=False) # save the df as a csv file
### Exploratory Data Analysis (EDA)
# unique invoices (transactions), products, and customers & their distributions
df_val = df_val.astype({"Customer ID": "O"})
print(df_val.describe(include=['O'])) # 36969 unique invoices, 3631 unique products, 5878 unique customers
invoice_cnt = df_val.groupby('Invoice').count()
print(invoice_cnt['StockCode'].describe()) # descriptive statistics on number of products ordered per transaction
invoice_cnt.hist(column='StockCode', bins=50)  # histogram of number of products ordered per transaction
plt.show()
product_cnt = df_val.groupby('StockCode').count()
print(product_cnt['Quantity'].describe()) # descriptive statistics on how many times each product is ordered
product_cnt.hist(column='Quantity', bins=50)  # histogram of number of orders per product
plt.show()
customer_cnt = df_val.groupby('Customer ID').count()
print(customer_cnt['Quantity'].describe()) # descriptive statistics on number of purchases per customer (at product level)
customer_cnt.hist(column='Quantity', bins=50)  # histogram of number of purchases per customer (at product level)
plt.show()
# who buys what, how often, how much, and when?
# composition of baskets (invoices)
# graphical presentations