##### RFM Anallysis #####
import matplotlib.pyplot as plt
import os 
import pandas as pd


df = pd.read_csv(os.path.join("data", "online_retail_II_preprocessed.csv")) # read the csv file

### measure variables - recency, frequency, and monetary value
rec = df.groupby(['Customer ID']).mean()['Recency']
freq = df.groupby(['Customer ID']).count()['Invoice'] # frequency number of purchases since the first purchase 
# monetary value average spending per order
df['Spending'] = df['Price']*df['Quantity']
mon =  df.groupby(['Customer ID']).mean()['Spending'] # monetary value average spending per order

### for each dimension, divide all customers into three groups evenly. 
seg = [1]*(len(rec)//3) + [2]*(len(rec)- len(rec)//3*2) + [3]*(len(rec)//3) # score: 1 for lowest, 2 for medium, 3 for highest
rec = rec.sort_values(ascending=False).to_frame()
freq = freq.sort_values().to_frame("Frequency")
mon = mon.sort_values().to_frame("MonetaryValue")
rec['Recency_score'] = seg
freq['Freq_score'] = seg
mon['MV_score'] = seg
df['R_score'] = [1]*len(df.index)
df['F_score'] = [1]*len(df.index)
df['M_score'] = [1]*len(df.index)
df.loc[df['Customer ID'].isin(rec[rec['Recency_score']==2].index),'R_score'] = 2
df.loc[df['Customer ID'].isin(rec[rec['Recency_score']==3].index),'R_score'] = 3
df.loc[df['Customer ID'].isin(freq[freq['Freq_score']==2].index),'F_score'] = 2
df.loc[df['Customer ID'].isin(freq[freq['Freq_score']==3].index),'F_score'] = 3
df.loc[df['Customer ID'].isin(mon[mon['MV_score']==2].index),'M_score'] = 2
df.loc[df['Customer ID'].isin(mon[mon['MV_score']==3].index),'M_score'] = 3
rfm = df.groupby(['R_score', 'F_score', 'M_score'])

### Summarize their features
pd.options.display.float_format = '{:.2f}'.format
print("number of customers in each group")
print(rfm.nunique()['Customer ID']) 
print(rfm.nunique()['Customer ID']/rfm.nunique()['Customer ID'].sum()*100) # proportion 
rfm.nunique()['Customer ID'].plot.bar()
plt.show()

print("number of countries in each group")
print(rfm.nunique()['Country']) 
tmp = df.groupby('Customer ID').max().get(['R_score', 'F_score', 'M_score', 'Country'])
print(tmp.groupby(['R_score', 'F_score', 'M_score'])['Country'].apply(pd.Series.mode)) # which countries are the customers from
#for i in range(3):
#    for j in range(3):
#        for k in range(3):
#            print("Group{}{}{}: {}".format(i+1,j+1,k+1,df[df['R_score']==i+1][df['F_score']==j+1][df['M_score']==k+1]['Country'].unique().tolist()))

print("which weekday were the purchases made most frequently?")
tmp = df.groupby('Invoice').max().get(['R_score', 'F_score', 'M_score', 'Weekday'])
print(tmp.groupby(['R_score', 'F_score', 'M_score'])['Weekday'].apply(pd.Series.mode))

### each group's contribution to 
print("contribution in aggrgated sales")
contrib = rfm.sum()['Spending']
print(contrib)
print(contrib/contrib.sum()*100)
contrib.plot.bar()



