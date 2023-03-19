##### RFM Anallysis #####

import os 
import pandas as pd


df = pd.read_csv(os.path.join("data", "online_retail_II_preprocessed.csv")) # read the csv file

### measure variables - recency, frequency, and monetary value
rec = df.groupby(['Customer ID']).mean()['Recency']
freq = df.groupby(['Customer ID']).count()['Invoice'] # frequency number of purchases since the first purchase 
# monetary value average spending per order
df['Spending'] = df['Price']*df['Quantity']
mon =  df.groupby(['Customer ID']).sum()['Spending'] # monetary value average spending per order

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
import pdb; pdb.set_trace()
print(1)

