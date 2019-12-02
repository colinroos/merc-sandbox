import datetime
print(datetime.datetime.now())

# import packages
import pandas as pd
import os
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from sklearn.cluster import KMeans
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score, silhouette_samples
print('import done.')

# read in files
os.chdir('data')
path1 = '2017/GCL_Geotech.csv'
path2 = '2017/GCL_Altn.csv'
path3 = '2017/GCL_Magsus.csv'
path4 = '2017/GCL_Struct.csv'
path5 = '2017/GCL_Veins.csv'
path6 = '2017/GCL_StructDomains.csv'
path7 = '2017/GCL_Assays_Complete.csv'
path8 = '2017/GCL_Lith.csv'
path9 = '2017/GCL_Minzn.csv'

ul1_Geo17 = pd.read_csv(path1, encoding='latin1', low_memory=True)
ul2_Alt17 = pd.read_csv(path2, encoding='latin1', low_memory=True)
ul3_Mag17 = pd.read_csv(path3, encoding='latin1', low_memory=True)
ul4_Stu17 = pd.read_csv(path4, encoding='latin1', low_memory=True)
ul5_Vei17 = pd.read_csv(path5, encoding='latin1', low_memory=True)
ul6_Std17 = pd.read_csv(path6, encoding='latin1', low_memory=True)
ul7_Ass17 = pd.read_csv(path7, encoding='latin1', low_memory=False)
ul8_Lit17 = pd.read_csv(path8, encoding='latin1', low_memory=True)
ul9_Min17 = pd.read_csv(path9, encoding='latin1', low_memory=True)

ul1_Geo17 = ul1_Geo17[ul1_Geo17['Project'] == 'GCL']
ul2_Alt17 = ul2_Alt17[ul2_Alt17['Project'] == 'GCL']
ul3_Mag17 = ul3_Mag17[ul3_Mag17['Project'] == 'GCL']
ul4_Stu17 = ul4_Stu17[ul4_Stu17['Project'] == 'GCL']
ul5_Vei17 = ul5_Vei17[ul5_Vei17['Project'] == 'GCL']
ul6_Std17 = ul6_Std17[ul6_Std17['Project'] == 'GCL']
ul7_Ass17 = ul7_Ass17[ul7_Ass17['Project'] == 'GCL']
ul8_Lit17 = ul8_Lit17[ul8_Lit17['Project'] == 'GCL']
ul9_Min17 = ul9_Min17[ul9_Min17['Project'] == 'GCL']

# drop non-useful columns
ul1_drop = ['Project', 'Relog', 'Logger', 'Interval_Length', 'Recovery_m', 'Recovery_Pct']
ul2_drop = ['Project', 'Relog', 'Logger', 'Timestamp', 'Alt5', 'Alt5_Int']
ul3_drop = ['Project', 'Relog', 'Logger', 'Interval_Length', 'Instrument',
            'Units', 'Reading1', 'Reading2', 'Reading3', 'Comments', 'Timestamp']
ul4_drop = ['Project', 'Relog', 'Logger', 'Struct1_Dip_Direction', 'Struct1_Dip',
            'Alpha', 'Beta', 'Gamma', 'Comments', 'Timestamp']
ul5_drop = ['Project', 'Relog', 'Logger', 'Timestamp']
ul6_drop = ['Project', 'Relog', 'Logger', 'Timestamp']
ul7_keep = ['Hole', 'From_m', 'To_m', 'Au_Best_ppm', 'Ag_Best_ppm', 'Cu_Best_ppm',
            'Pb_Best_ppm', 'Zn_Best_ppm']
ul8_drop = ['Project', 'Relog', 'Logger', 'Timestamp', 'Lith1_Volc_Txt1', 'Lith1_Volc_Txt2',
            'ImagePath']
ul9_drop = ['Project', 'Relog', 'Logger', 'Timestamp', 'Min4', 'Min5']

ul1_Geo17.drop(ul1_drop, axis=1, inplace=True)
ul2_Alt17.drop(ul2_drop, axis=1, inplace=True)
ul3_Mag17.drop(ul3_drop, axis=1, inplace=True)
ul4_Stu17.drop(ul4_drop, axis=1, inplace=True)
ul5_Vei17.drop(ul5_drop, axis=1, inplace=True)
ul6_Std17.drop(ul6_drop, axis=1, inplace=True)
ul8_Lit17.drop(ul8_drop, axis=1, inplace=True)
ul9_Min17.drop(ul9_drop, axis=1, inplace=True)
ul7_Ass17 = ul7_Ass17[ul7_keep]

print('Geo17')
print(ul1_Geo17.info())
print('Alt17')
print(ul2_Alt17.info())
print('Mag17')
print(ul3_Mag17.info())
print('Stu17')
print(ul4_Stu17.info())
print('Vei17')
print(ul5_Vei17.info())

# ul1_Geo17['Timestamp'] = ul1_Geo17['Timestamp'].astype('datetime64')
# toConvert = ['Hardness', 'Weathering', 'Fracture_Infill', 'Roughness', 'Shape']
# ul1_Geo17[toConvert] = ul1_Geo17[toConvert].astype('category')
# print(ul1_Geo17.info())
#
# # hash holeid, from and to depth togther
# ul1_Geo17['hex'] = ul1_Geo17['Hole'] + ul1_Geo17['From_m'].astype(str) + ul1_Geo17['To_m'].astype(str)
# ul1_Geo17['hex'] = ul1_Geo17['hex'].apply(hash)
# ul1_Geo17.set_index('hex', drop=True, inplace=True)
# print(ul1_Geo17.head())
