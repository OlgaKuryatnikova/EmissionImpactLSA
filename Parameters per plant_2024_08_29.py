# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:54:55 2022

@author: 78160cdu
"""
import pandas as pd
path_file_source= "C:\\Users\\78160cdu\Dropbox (Erasmus Universiteit Rotterdam)\\EnergyMarket\\Data\\source\\"


#Fuel cost and ETS
P_coal_2019 = pd.read_csv(f"{path_file_source}Coal_12_31_19-01_02_19.csv")
P_coal_2019["Date"] = pd.to_datetime(P_coal_2019["Date"],format = "%m/%d/%y")
P_coal_2019['Close'].mean()

P_coal_2022 = pd.read_csv(f"{path_file_source}Coal_12_30_22-01_03_22.csv")
P_coal_2022["Date"] = pd.to_datetime(P_coal_2022["Date"],format = "%m/%d/%y")
P_coal_2022['Close'].mean()

P_TTF = pd.read_csv(f"{path_file_source}Dutch TTF Natural Gas Futures Historical Data.csv")
P_TTF["Date"] = pd.to_datetime(P_TTF["Date"],format = "%m/%d/%Y")
P_TTF.drop(columns = ['Vol.', 'Change %'], inplace = True)
TTF = P_TTF.groupby(P_TTF['Date'].dt.year).mean()['Price']

P_ETS = pd.read_csv(f"{path_file_source}Carbon Emissions Futures Historical Data.csv")
P_ETS["Date"] = pd.to_datetime(P_ETS["Date"],format = "%m/%d/%Y")
P_ETS.drop(columns = ['Vol.', 'Change %'], inplace = True)
ETS = P_ETS.groupby(P_ETS['Date'].dt.year).mean()['Price']

parameters_techno = pd.DataFrame({
  "O&M cost (eur/Mwh)": [1.43, 5.71, 3.47],
  "Fuel cost 2019 (eur/Mwh)": [float(TTF.loc[[2019]]), P_coal_2019['Close'].mean()*0.12, 5.1], 
  "Fuel cost 2022 (eur/Mwh)": [float(TTF.loc[[2022]]), P_coal_2022['Close'].mean()*0.12, 7.64], 
  "ETS price 2019 (eur/t)":[float(ETS.loc[[2019]]), float(ETS.loc[[2019]]), float(ETS.loc[[2019]])],
  "ETS price 2022 (eur/t)":[float(ETS.loc[[2022]]), float(ETS.loc[[2022]]), float(ETS.loc[[2022]])]
}, index = ["Fossil Gas", "Fossil Hard coal", "Nuclear"])
#For nuclear it is already out, while for coal and gas it is before. 

#Marginal cost and emission of plant
Performance_s = pd.read_csv(f"{path_file_source}JRC_OPEN_PERFORMANCE.csv", index_col=1)
Temporal_s = pd.read_csv(f"{path_file_source}JRC_OPEN_TEMPORAL.csv")
Units_s = pd.read_csv(f"{path_file_source}JRC_OPEN_UNITS.csv", index_col=1)

df_units = Units_s.drop(Units_s[Units_s.country != 'Netherlands'].index)
df_units.loc[df_units[df_units.name_g == 'Hemweg 9'].index,'type_g'] = ['Fossil Gas']
df_units.loc['49W000000000057R',['year_decommissioned', 'status_g']]= [2015, 'DECOMISSIONED']


df_units = df_units.drop(df_units[df_units.year_decommissioned < 2019].index)
df_units.loc[df_units[df_units.name_g == 'Hemweg 9'].index,'type_g'] = 'Fossil Gas'

df_units.drop(["year_commissioned", "year_decommissioned", "lon", 'lat', 'status_g',  'NUTS2', 'eic_p', 'capacity_p'], axis=1, inplace=True)
df_units = df_units.drop(df_units[ df_units.type_g == 'Wind Onshore'].index)
df_units.index.name = 'eic_g'

df_plants = df_units.merge(Performance_s, how='left', on='eic_g')

df_emission = Temporal_s
df_emission['Co2/Mwh (kg)']= df_emission['co2emitted'] / df_emission['Generation'] 


df_emission = df_emission.groupby(["eic_g"]).max()
#df_emission_2 = df_emission.sort_values('cyear', ascending=False).drop_duplicates(['eic_g'])# Both works

# Merging the Co2/Mwh (kg) column
df_plants = df_plants.merge(df_emission['Co2/Mwh (kg)'], how='left', on='eic_g')

# Creating new rows as DataFrames
new_rows = [
    pd.DataFrame({'name_g': ['Bergum 10'], 'capacity_g': [72], 'type_g': ['Fossil Gas'], 'eff': [0.31], 'best_source': ['eia'],  'Co2/Mwh (kg)': [483]}, index=['49W0000000001233']),
    pd.DataFrame({'name_g': ['Bergum 20'], 'capacity_g': [72], 'type_g': ['Fossil Gas'], 'eff': [0.31], 'best_source': ['eia'],  'Co2/Mwh (kg)': [483]}, index=['49W000000000135X']),
    pd.DataFrame({'name_g': ['Leiden'], 'capacity_g': [85], 'type_g': ['Fossil Gas'], 'eff': [0.31], 'best_source': ['eia'], 'Co2/Mwh (kg)': [483]}, index=['XXXX'])
]

# Concatenating the new rows to the existing DataFrame
df_plants = pd.concat([df_plants] + new_rows, axis=0)

# Ensure that the indices are preserved or reset as needed
df_plants.index.name = 'eic_g'


#imput missing values
frames = []
for i in list(set(df_plants['type_g'])):
    df_type = df_plants[df_plants['type_g']== i]
    df_type['eff'].fillna(df_type['eff'].mean(),inplace = True)
    df_type['min_load'].fillna(df_type['min_load'].mean(),inplace = True)
    df_type['Co2/Mwh (kg)'].fillna(df_type['Co2/Mwh (kg)'].mean(),inplace = True)
    df_type['ramp_up'].fillna(df_type['ramp_up'].mean(),inplace = True)
    df_type['ramp_down'].fillna(df_type['ramp_down'].mean(),inplace = True)
    frames.append(df_type)
    final_df = pd.concat(frames)
    
#to put together plants close to each other 
frames = []
for i in list(set(final_df['type_g'])):
    df_type = final_df[final_df['type_g']== i]
    df_type = df_type.sort_values(['eff',"Co2/Mwh (kg)"])
    df_type['group_eff'] = df_type['eff'].sort_values().diff().gt(0.01).cumsum()
    df_type['group_co2'] = df_type['Co2/Mwh (kg)'].sort_values().diff().gt(5).cumsum()
    df_type = df_type.groupby(["group_co2", 'group_eff'], as_index=False).agg({'capacity_g': 'sum', 'min_load': 'mean', 'ramp_up': 'mean', 'ramp_down': 'mean', 'minimum_up_time': 'mean','minimum_down_time': 'mean', 'eff': 'mean', 'Co2/Mwh (kg)': 'mean', 'name_g' : 'size'})
    df_type['type_g']=i
    frames.append(df_type.iloc[:,2:12])
final_df2 = pd.concat(frames)
final_df2 = final_df2.reset_index()
final_df2 = final_df2.drop('index', axis = 1)
final_df2 = final_df2.rename(columns={"name_g": "nb_plants"})
final_df2['min_load_div'] = final_df2['min_load'] / final_df2['nb_plants'] 


#For getting the total df
#final_df2 = final_df.copy()
final_df2['operating_cost_2019'] = 0
final_df2['emission_cost_2019'] = 0  
final_df2['operating_cost_2022'] = 0
final_df2['emission_cost_2022'] = 0 


frames = []
for i in list(set(final_df2['type_g'])):
    df_type = final_df2[final_df2['type_g']== i]
    if i != 'Nuclear':
        df_type['operating_cost_2019']=parameters_techno.loc[i,'O&M cost (eur/Mwh)']+parameters_techno.loc[i,'Fuel cost 2019 (eur/Mwh)']/df_type['eff']
        df_type['emission_cost_2019']=parameters_techno.loc[i,'ETS price 2019 (eur/t)']*df_type['Co2/Mwh (kg)']/1000
        df_type['operating_cost_2022']=parameters_techno.loc[i,'O&M cost (eur/Mwh)']+parameters_techno.loc[i,'Fuel cost 2022 (eur/Mwh)']/df_type['eff']
        df_type['emission_cost_2022']=parameters_techno.loc[i,'ETS price 2022 (eur/t)']*df_type['Co2/Mwh (kg)']/1000
    else :
        df_type['operating_cost_2019']=parameters_techno.loc[i,'O&M cost (eur/Mwh)']+parameters_techno.loc[i,'Fuel cost 2019 (eur/Mwh)']
        df_type['operating_cost_2022']=parameters_techno.loc[i,'O&M cost (eur/Mwh)']+parameters_techno.loc[i,'Fuel cost 2022 (eur/Mwh)']
        
    frames.append(df_type)
    final_df3 = pd.concat(frames)

final_df3['marginal_cost_2019']= final_df3['operating_cost_2019'] +final_df3['emission_cost_2019'] 
final_df3['marginal_cost_2022']= final_df3['operating_cost_2022'] +final_df3['emission_cost_2022'] 

#Waste is considered as Baseload like renewable, I find it by taking the max usage in 2019 (Maybe mean would make more sense ?)


final_df3.to_csv("C:\\Users\\78160cdu\\OneDrive - Erasmus University Rotterdam\Documents\\Research\\Project 1\\Data\\Parameters_plants_NL.csv")

