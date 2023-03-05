##############################################################################################################################################################
### [First Project]                                                                                                                                        ### 
### Project report topic: What business implications can we derive from online retail data?                                                                ###   
###                                                                                                                                                        ###
### [Your jobs to do for the coming week]                                                                                                                  ### 
### 1.	Read the dataset: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II                                                                         ###
### 2.	With Python, perform exploratory data analysis (EDA) to understand the data.                                                                       ###
###     Here, EDA includes reporting the description of variables, how they are measured, how often they are measured, and their descriptive statistics.   ###
### 3.	Think about what business analysis you can do with the data. And then, write your idea on a half page.                                             ###
###     Then under this half page, add what you did at 2.                                                                                                  ###
##############################################################################################################################################################

from tabulate import tabulate
import pandas as pd


# Read the online retail data
df0910 = pd.read_excel("online_retail_II.xlsx", sheet_name=0) # 2009-2010
df1011 = pd.read_excel("online_retail_II.xlsx", sheet_name=1) # 2010-2011
#df0910 = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx", sheet_name=0) 
#df1011 = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx", sheet_name=1) 

# explore the data
print(df0910.head())
print(df1011.head())

# merge data 
df = pd.concat([df0910, df1011], axis=0)

### descriptive statistics
print("DESCRIPTIVE STATISTICS")
# numeric variables
pd.options.display.float_format = '{:.4f}'.format # display up to four decimals 
desc0910 = df0910.describe()
desc1011 = df1011.describe()
desc = df.describe()
desc_qnt = pd.concat([desc0910['Quantity'], desc1011['Quantity'], desc['Quantity']], axis=1) # quantity descriptive statistucs
desc_prc = pd.concat([desc0910['Price'], desc1011['Price'], desc['Price']], axis=1) # unit price descriptive statistics
print("<Quantity>")
print(tabulate(desc_qnt, tablefmt='psql', numalign="right", headers=["2009-2010", "2010-2011", "all"], floatfmt=".4f"))
print("<Price>")
print(tabulate(desc_prc, tablefmt='psql', numalign="right", headers=["2009-2010", "2010-2011", "all"], floatfmt=".4f"))
# others
df = df.astype({"Customer ID": "O", "InvoiceDate": "O"})
df0910 = df0910.astype({"Customer ID": "O", "InvoiceDate": "O"})
df1011 = df1011.astype({"Customer ID": "O", "InvoiceDate": "O"})
print(tabulate(df.describe(include=['O']), tablefmt='psql', numalign="right", headers="keys"))
print(tabulate(df0910.describe(include=['O']), tablefmt='psql', numalign="right", headers="keys"))
print(tabulate(df1011.describe(include=['O']), tablefmt='psql', numalign="right", headers="keys"))
