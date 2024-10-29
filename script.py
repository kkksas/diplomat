import pandas as pd
import numpy as np
df = pd.read_excel('./spreadcheek.xlsx')
print(df['тип отдыха'])
a = []
for i in df['тип отдыха'].astype(str):
    for j in i.split(','):
        a.append(j.strip().lower())
print(set(a))