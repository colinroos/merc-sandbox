from tqdm import tqdm
import pandas as pd
from file_utils import *
import pickle

# read in files
target = 'data/2017/'
data = []
include_tables = ['GCL_Altn', 'GCL_Assays_Complete', 'GCL_Geotech', 'GCL_Lith', 'GCL_Magsus', 'GCL_Minzn', 'GCL_Stuct',
                  'GCL_StructDomains', 'GCL_Veins']

for file in find_files_glob(target, extension="*.csv"):
    name = file[file.find('GCL'):-4]
    if name in include_tables:
        d = pd.read_csv(file, encoding='latin1', low_memory=False)
        d.index.name = name
        data.append(d)

# drop unneeded columns
drop = {'GCL_Altn': ['Project', 'Relog', 'Logger', 'Timestamp', 'Alt4', 'Alt4_Int', 'Alt5', 'Alt5_Int'],
        'GCL_Assays_Complete': ['Hole', 'From_m', 'To_m', 'Au_Best_ppm', 'Ag_Best_ppm', 'Cu_Best_ppm', 'Pb_Best_ppm',
                                'Zn_Best_ppm'],
        'GCL_Geotech': ['Project', 'Relog', 'Logger', 'Interval_Length', 'Recovery_m', 'Recovery_Pct', 'RQD_Pct',
                        'Hardness', 'Roughness', 'Timestamp'],
        'GCL_Lith': ['Project', 'Relog', 'Logger', 'Timestamp', 'Lith1_Volc_Txt1', 'Lith1_Volc_Txt2', 'ImagePath'],
        'GCL_Magsus': ['Project', 'Relog', 'Logger', 'Interval_Length', 'Instrument', 'Units', 'Reading1', 'Reading2',
                       'Reading3', 'Comments', 'Timestamp'],
        'GCL_Minzn': ['Project', 'Relog', 'Logger', 'Timestamp', 'Min4', 'Min5'],
        'GCL_Struct': ['Project', 'Relog', 'Logger', 'Thickness_m', 'Struct1_Dip_Direction', 'Struct1_Dip', 'Alpha',
                       'Beta', 'Gamma', 'Comments', 'Timestamp'],
        'GCL_StructDomains': ['Project', 'Relog', 'Logger', 'Timestamp'],
        'GCL_Veins': ['Project', 'Relog', 'Logger', 'Timestamp', 'Vein1_Grain_Size']}

categorical = {'GCL_Altn': ['Alt_Group', 'Alt_Intensity', 'Alt1', 'Alt1_Int', 'Alt2', 'Alt2_Int', 'Alt3', 'Alt3_Int'],
               'GCL_Assays_Complete': None,
               'GCL_Geotech': ['Weathering', 'Shape'],
               'GCL_Lith': ['Colour', 'Oxidation', 'Foliation', 'QtzCarb_Replacement_Pct', 'Lith1_Group', 'Lith1',
                            'Lith1_Texture', 'Lith1_Bedding'],
               'GCL_Magsus': None,
               'GCL_Minzn': ['Min_Intensity', 'Min_Style', 'VG', 'Min1', 'Min2', 'Min3'],
               'GCL_Struct': ['Struct1', 'Struct1_Confidence', 'Struct1_Int'],
               'GCL_StructDomains': ['Bedding_Confidence', 'Bedding_Style', 'Bedding_Bed_Thickness',
                                     'Bedding_Younging_Direction', 'Bedding_Cleavage_Angle', 'Fabric1_Feature',
                                     'Fabric1_Confidence', 'Fabric1_Intensity', 'Fabric2_Feature', 'Fabric2_Confidence',
                                     'Fabric2_Intensity'],
               'GCL_Veins': ['Vein1', 'Vein1_Style', 'Vein1_Form', 'Vein1_Internal_Structure', 'Vein1_VG', 'Vein2',
                             'Vein2_Style', 'Vein2_Form', 'Vein2_Internal_Structure', 'Vein2_VG', 'Vein3',
                             'Vein3_Style', 'Vein3_Form', 'Vein3_Internal_Structure', 'Vein3_VG']}

for i, d in enumerate(data):
    name = d.index.name
    if i == 1:
        # Columns to keep
        d = d[drop[name]]
    else:
        # Columns to drop
        d = d.drop(drop[name], axis=1)
    if categorical[name]:
        d[categorical[name]] = d[categorical[name]].astype('category')
    data[i] = d

# Join all tables on Hole ID and Depth
joins = ['Hole', 'From_m']

df = data[0]
for i, d in enumerate(data):
    if i > 0:
        df = pd.merge(df, data[i], how='outer', on=joins, suffixes=('_' + data[i-1].index.name, '_' + d.index.name))

# Sort merged table by hole and depth
df = df.sort_values(['Hole', 'From_m'])
df = df.reset_index(drop=True)
df.ffill(inplace=True)

df.to_pickle('out/merged_master.pkl')
df.to_csv('out/merged_master.csv')
