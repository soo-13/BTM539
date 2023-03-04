import pandas as pd


#df0910 = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx", sheet_name=0)
#df1011 = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx", sheet_name=1)
df0910 = pd.read_excel("online_retail_II.xlsx", sheet_name=0)
df1011 = pd.read_excel("online_retail_II.xlsx", sheet_name=1)

print(df0910.shape)
print(df0910.columns)
print(df0910.head())
print(df1011.shape)
print(df1011.columns)
print(df1011.head())