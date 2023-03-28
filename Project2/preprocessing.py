import os
import pandas as pd

datatitle = os.listdir(os.path.join("Project2", "Pjt2"))
info = []
for fname in datatitle:
    with open(os.path.join("Project2", "Pjt2", fname),mode="rb") as excel_file:
        df = pd.read_excel(os.path.join("Project2", "Pjt2", fname))
    info.append(df.get(['Article Title', 'Author Keywords', 'Keywords Plus', 'Abstract', 'Affiliations', 'Publication Year', 'Journal Abbreviation']))
df = pd.concat(info)
print(df.head())
print(df.shape)