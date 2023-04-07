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
df = df.rename({'Article Title': 'Title', 'Author Keywords': 'Akwds', 'Keywords Plus': 'Kplus', 'Abstract': 'Abst',
                'Affiliations': 'Affil', 'Publication Year': 'Year', 'Journal Abbreviation': 'Jabb'}, axis=1) # rename for conveneince
print(df.head())
print(df.shape) # check 11920 = 2302 + 1784 + 6018 + 1806
df.loc[df['Jabb'] == 'STRATEG MANAGE J', 'Jabb'] = 'STRATEGIC MANAGE J' # unify the abbreviation term for SMJ

### check for missing data
df['Year'].fillna(2023, inplace=True) # publication year is 2023 if missing (recent papers)
df_val = df[df['Abst'].notna()] # remove if abstract is nan
df_val = df_val.drop_duplicates(subset='Title') # check duplicates in title
df_val = df_val.drop_duplicates(subset='Abst', keep=False) # remove different titles with same abstract (warning sign in place of abstract)
print(df_val.isna().sum()) # checking missing values for each variable
print(df_val.groupby('Jabb')['Title'].nunique())
df_val.to_csv(os.path.join("Project2", "data", "agg_article_info.csv"), index=False) # save final data as csv file