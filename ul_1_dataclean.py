import datetime
import pandas as pd
from util.findFiles import findfiles

print(datetime.datetime.now())

# import packages

print('import done.')

# read in files
target = 'data/2017'
data = []
names = ['Alterations', 'Assays', 'Geotech', 'Lithography', 'Magsus', 'Mineralization', 'Structures',
         'StructuralDomains', 'Veins']

for file in findfiles(target):
    d = pd.read_csv(file, encoding='latin1', low_memory=False)
    data.append(d)

# data[1] = data[1].rename(columns={'DHProject': 'Project'})


# drop non-useful columns
# drop = [['Project', 'Relog', 'Logger', 'Timestamp', 'Alt5', 'Alt5_Int'],
#         ['Hole', 'From_m', 'To_m', 'Au_Best_ppm', 'Ag_Best_ppm', 'Cu_Best_ppm',
#          'Pb_Best_ppm', 'Zn_Best_ppm'], ['Project', 'Relog', 'Logger', 'Interval_Length', 'Recovery_m', 'Recovery_Pct'],
#         ['Project', 'Relog', 'Logger', 'Timestamp', 'Lith1_Volc_Txt1', 'Lith1_Volc_Txt2',
#          'ImagePath'], ['Project', 'Relog', 'Logger', 'Interval_Length', 'Instrument',
#                         'Units', 'Reading1', 'Reading2', 'Reading3', 'Comments', 'Timestamp'],
#         ['Project', 'Relog', 'Logger', 'Timestamp', 'Min4', 'Min5'],
#         ['Project', 'Relog', 'Logger', 'Struct1_Dip_Direction', 'Struct1_Dip',
#          'Alpha', 'Beta', 'Gamma', 'Comments', 'Timestamp'], ['Project', 'Relog', 'Logger', 'Timestamp'],
#         ['Project', 'Relog', 'Logger', 'Timestamp']]

# for i, d in enumerate(data):
#     if i == 1:
#         d = d[drop[i]]
#     else:
#         d = d.drop(drop[i], axis=1, inplace=True)

joins = ['Hole', 'From_m', 'To_m']

for d in data:
    print(d[joins].head())


df = pd.merge(data[0], data[1], how='left', on=joins)
df = pd.merge(df, data[2], how='left', on=joins)
df = pd.merge(df, data[3], how='left', on=joins)
df = pd.merge(df, data[4], how='left', on=joins)
df = pd.merge(df, data[5], how='left', on=joins)
df = pd.merge(df, data[6], how='left', on=joins)
df = pd.merge(df, data[7], how='left', on=joins)

# df.to_csv('out/merged.csv')

print(df.head())

# ul1_Geo17.drop(ul1_drop, axis=1, inplace=True)
# ul2_Alt17.drop(ul2_drop, axis=1, inplace=True)
# ul3_Mag17.drop(ul3_drop, axis=1, inplace=True)
# ul4_Stu17.drop(ul4_drop, axis=1, inplace=True)
# ul5_Vei17.drop(ul5_drop, axis=1, inplace=True)
# ul6_Std17.drop(ul6_drop, axis=1, inplace=True)
# ul8_Lit17.drop(ul8_drop, axis=1, inplace=True)
# ul9_Min17.drop(ul9_drop, axis=1, inplace=True)
# ul7_Ass17 = ul7_Ass17[ul7_keep]
# top = ['Hole', 'From_m', 'To_m']
# print('Geo17')
# print(ul1_Geo17[top].head())
# print('Alt17')
# print(ul2_Alt17[top].head())
# print('Mag17')
# print(ul3_Mag17[top].head())
# print('Stu17')
# print(ul4_Stu17.head())
# print('Vei17')
# print(ul5_Vei17[top].head())
# print('Std17')
# print(ul6_Std17[top].head())
# print('Ass17')
# print(ul7_Ass17[top].head())
# print('Lit17')
# print(ul8_Lit17[top].head())
# print('Min17')
# print(ul9_Min17[top].head())
#
# # Convert features to categorical
# ul1_Geo17['Timestamp'] = ul1_Geo17['Timestamp'].astype('datetime64')
# ul1_cat = ['Hardness', 'Weathering', 'Fracture_Infill', 'Roughness', 'Shape']
# ul2_cat = ['Alt_Group', 'Alt_Style', 'Alt1', 'Alt2', 'Alt3', 'Alt4']
# ul4_cat = ['Struct1', 'Struct1_Confidence']
# ul5_cat = ['Vein1', 'Vein1_Style', 'Vein1_Form', 'Vein1_Internal_Structure', 'Vein2', 'Vein2_Style', 'Vein2_Form',
#            'Vein2_Internal_Structure', 'Vein3', 'Vein3_Style', 'Vein3_Form', 'Vein3_Internal_Structure']
# ul6_cat = ['Bedding_Confidence', 'Bedding_Style', 'Bedding_Bed_Thickness', 'Bedding_Younging_Direction',
#            'Bedding_Cleavage_Angle', 'Fabric1_Feature', 'Fabric1_Confidence', 'Fabric2_Feature', 'Fabric2_Confidence']
# ul8_cat = ['Colour', 'Oxidation', 'Foliation', 'Lith1_Group', 'Lith1', 'Lith1_Texture', 'Lith1_Bedding']
# ul9_cat = ['Min_Style', 'Min1', 'Min2', 'Min3']
#
# ul1_Geo17[ul1_cat] = ul1_Geo17[ul1_cat].astype('category')
# ul2_Alt17[ul2_cat] = ul2_Alt17[ul2_cat].astype('category')
# ul4_Stu17[ul4_cat] = ul4_Stu17[ul4_cat].astype('category')
# ul5_Vei17[ul5_cat] = ul5_Vei17[ul5_cat].astype('category')
# ul6_Std17[ul6_cat] = ul6_Std17[ul6_cat].astype('category')
# ul8_Lit17[ul8_cat] = ul8_Lit17[ul8_cat].astype('category')
# ul9_Min17[ul9_cat] = ul9_Min17[ul9_cat].astype('category')
#
# # Create Hex
# # Hash-able tables
# # 1,2,3,5,6,7,8,9
# # hash holeid, from and to depth together
# ul1_Geo17['hex'] = ul1_Geo17['Hole'] + ul1_Geo17['From_m'].astype(str) + ul1_Geo17['To_m'].astype(str)
# ul2_Alt17['hex'] = ul2_Alt17['Hole'] + ul2_Alt17['From_m'].astype(str) + ul2_Alt17['To_m'].astype(str)
# ul3_Mag17['hex'] = ul3_Mag17['Hole'] + ul3_Mag17['From_m'].astype(str) + ul3_Mag17['To_m'].astype(str)
# ul5_Vei17['hex'] = ul5_Vei17['Hole'] + ul5_Vei17['From_m'].astype(str) + ul5_Vei17['To_m'].astype(str)
# ul6_Std17['hex'] = ul6_Std17['Hole'] + ul6_Std17['From_m'].astype(str) + ul6_Std17['To_m'].astype(str)
# ul7_Ass17['hex'] = ul7_Ass17['Hole'] + ul7_Ass17['From_m'].astype(str) + ul7_Ass17['To_m'].astype(str)
# ul8_Lit17['hex'] = ul8_Lit17['Hole'] + ul8_Lit17['From_m'].astype(str) + ul8_Lit17['To_m'].astype(str)
# ul9_Min17['hex'] = ul9_Min17['Hole'] + ul9_Min17['From_m'].astype(str) + ul9_Min17['To_m'].astype(str)
#
# # Apply the hash function
# ul1_Geo17['hex'] = ul1_Geo17['hex'].apply(hash)
# ul2_Alt17['hex'] = ul2_Alt17['hex'].apply(hash)
# ul3_Mag17['hex'] = ul3_Mag17['hex'].apply(hash)
# ul5_Vei17['hex'] = ul5_Vei17['hex'].apply(hash)
# ul6_Std17['hex'] = ul6_Std17['hex'].apply(hash)
# ul7_Ass17['hex'] = ul7_Ass17['hex'].apply(hash)
# ul8_Lit17['hex'] = ul8_Lit17['hex'].apply(hash)
# ul9_Min17['hex'] = ul9_Min17['hex'].apply(hash)
#
# # Set table index to the new hash value
# ul1_Geo17.set_index('hex', drop=True, inplace=True)
# ul2_Alt17.set_index('hex', drop=True, inplace=True)
# ul3_Mag17.set_index('hex', drop=True, inplace=True)
# ul5_Vei17.set_index('hex', drop=True, inplace=True)
# ul6_Std17.set_index('hex', drop=True, inplace=True)
# ul7_Ass17.set_index('hex', drop=True, inplace=True)
# ul8_Lit17.set_index('hex', drop=True, inplace=True)
# ul9_Min17.set_index('hex', drop=True, inplace=True)
#
# # Join Left all tables
# df = ul1_Geo17.join(ul2_Alt17, how='left', rsuffix='_2')
# df = df.join(ul3_Mag17, how='left', rsuffix='_3')
# df = df.join(ul5_Vei17, how='left', rsuffix='_5')
# df = df.join(ul6_Std17, how='left', rsuffix='_6')
# df = df.join(ul7_Ass17, how='left', rsuffix='_7')
# df = df.join(ul8_Lit17, how='left', rsuffix='_8')
# df = df.join(ul9_Min17, how='left', rsuffix='_9')
#
# print(df.head())
# print(df.info())
#
# os.chdir('..')
# df.to_csv('out/cleaned.csv')
