import pandas as pd
import os

df = pd.read_csv('data/prabayar.csv')
for daya, group in df.groupby('Daya_Listrik_Rumah_VA'):
    clean_daya = str(daya).replace(' ', '_').replace('/', '_')
    filename = f'data/prabayar_{clean_daya}.csv'
    group.to_csv(filename, index=False)
    print(f"Created {filename}")
