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
df.loc[df['Journal Abbreviation'] == 'STRATEG MANAGE J', 'Journal Abbreviation'] = 'STRATEGIC MANAGE J' # unify the abbreviation term for SMJ

### check for missing data
print(df.isna().sum()) # checking missing values for each variable
df_val = df[df.isna().sum(axis=1)==0] # only include rows that do not have missing value in any column (without missing values)
# for AMJ, permit having missing value on column 'Author Keywords'
check_cols =  df.columns.tolist() # list of all columns except 'Author Keywords'
check_cols.remove('Author Keywords')
tmp = df.get(check_cols)
amj1 = df[df['Journal Abbreviation'].str.startswith('ACAD')] # amj journal
amj2 = df[tmp.isna().sum(axis=1)==0] # missing value at 'Author Keywords' only
amj = amj1.merge(amj2, how='inner', on=df.columns.tolist()) # intersection of amj1 and amj2 are AMJ with missing value at 'Author Keywords' only 
print("Checking for missing values in df_val")
print(df_val.isna().sum()) # should be all zeroes
print("Checking for missing values in amj")
print(amj.isna().sum()) # should be all zeroes except for author keywords

### merge and save data
df_val = df_val.merge(amj, how='outer', on=df.columns.tolist()) # merge df_val and amj
df_val.to_csv(os.path.join("Project2", "data", "agg_article_info.csv"), index=False) # save final data as csv file

### make a table showing observations for each Journal
import pdb; pdb.set_trace()
print(df_val.groupby('Journal Abbreviation').head())
print(df_val.groupby('Journal Abbreviation').count())