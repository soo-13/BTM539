import os
import pandas as pd


### combine data
dPATH = os.path.join("Project2", "data", "Pjt2")
datatitle = os.listdir(dPATH) # list of excel file titles
info = []
for fname in datatitle:
    with open(os.path.join(dPATH, fname),mode="rb") as excel_file:
        df = pd.read_excel(os.path.join(dPATH, fname))
    info.append(df.get(['Article Title', 'Author Keywords', 'Keywords Plus', 'Abstract', 'Affiliations', 'Publication Year', 'Journal Abbreviation']))
df = pd.concat(info) 
print(df.head())
print(df.shape) # check 11920 = 2302 + 1784 + 6018 + 1806

### check for missing data
print(df.isna().sum()) # checking missing values for each variable
df_val = df[df.isna().sum(axis=1)==0] # only include rows that do not have missing value in any column (without missing values)
print(df_val.isna().sum()) # should be all zeroes
df_val.to_csv(os.path.join("Project2", "data", "agg_article_info.csv"), index=False) # save final data as csv file