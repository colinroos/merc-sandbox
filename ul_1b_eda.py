import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load cleaned data table
df = pd.read_pickle('out/merged_master.pkl')

print(df.head())

kad_001 = df[df.Hole == 'KAD17-001']

print(kad_001.info())

kad_001.plot('RQD_m', 'From_m',figsize=(2, 6), legend=False)
plt.xlabel('RQD')
plt.ylabel(None)
plt.show()

kad_001.plot('Avg_Reading', 'From_m', figsize=(2, 6), legend=False)
plt.xlabel('Mag. Sus.')
plt.yticks([], [])
plt.show()



