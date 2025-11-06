# -*- coding: utf-8 -*-
"""
This file is to create the file parameters_plant.csv
"""
import pandas as pd
import numpy as np
path_file_source= "C:\\Users\\...\\"


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
  "O&M cost (eur/Mwh)": [1.43, 1.43, 5.71, 3.47],
  "Fuel cost 2019 (eur/Mwh)": [float(TTF.loc[[2019]]),float(TTF.loc[[2019]]), P_coal_2019['Close'].mean()*0.12, 5.1], 
  "Fuel cost 2022 (eur/Mwh)": [float(TTF.loc[[2022]]), float(TTF.loc[[2022]]), P_coal_2022['Close'].mean()*0.12, 7.64], 
  "ETS price 2019 (eur/t)":[float(ETS.loc[[2019]]), float(ETS.loc[[2019]]), float(ETS.loc[[2019]]), float(ETS.loc[[2019]])],
  "ETS price 2022 (eur/t)":[float(ETS.loc[[2022]]), float(ETS.loc[[2022]]), float(ETS.loc[[2022]]), float(ETS.loc[[2022]])],
  'Start-up cost (eur)':[(20+50)/2*0.95, (50+150)/2*0.95, (100+250)/2*0.95, 1000*0.95] #reference is IEA The power of transformation (Table B.4), exchange rate dollar to eur 0.95
}, index = ["Gas peaker", "Gas_cc", "Coal", "Nuclear"])
#For nuclear it is already out, while for coal and gas it is before. 

#Marginal cost and emission of plant
Performance_s = pd.read_csv(f"{path_file_source}JRC_OPEN_PERFORMANCE.csv", index_col=1)
Temporal_s = pd.read_csv(f"{path_file_source}JRC_OPEN_TEMPORAL.csv")
Units_s = pd.read_csv(f"{path_file_source}JRC_OPEN_UNITS.csv", index_col=1)

df_units = Units_s.drop(Units_s[Units_s.country != 'Netherlands'].index)
df_units.loc[df_units[df_units.name_g == 'Hemweg 9'].index,'type_g'] = ['Fossil Gas']
df_units.loc['49W0000000000059',['year_decommissioned']]= [np.nan] #claus C worked was replacing claus B

df_units = df_units.drop(df_units[df_units.year_decommissioned < 2019].index)
df_units.loc[df_units[df_units.name_g == 'Hemweg 9'].index,'type_g'] = 'Fossil Gas'

df_units.drop(["year_commissioned", "year_decommissioned", "lon", 'lat', 'status_g',  'NUTS2', 'eic_p', 'capacity_p'], axis=1, inplace=True)
df_units = df_units.drop(df_units[ df_units.type_g == 'Wind Onshore'].index)
df_units.index.name = 'eic_g'

df_plants = df_units.merge(Performance_s, how='left', on='eic_g')


#back to original code 
df_emission = Temporal_s
df_emission['Co2/Mwh (kg)']= df_emission['co2emitted'] / df_emission['Generation'] 


df_emission = df_emission.groupby(["eic_g"]).max()
#df_emission_2 = df_emission.sort_values('cyear', ascending=False).drop_duplicates(['eic_g'])# Both works

# Merging the Co2/Mwh (kg) column
df_plants = df_plants.merge(df_emission['Co2/Mwh (kg)'], how='left', on='eic_g')

#removing plants that are not producing 
df_plants.drop(['49W0000000000253', '49W0000000000229', '49W000000000057R'], axis = 0, inplace = True)
#Merwedekanaal 11 does not have any production in both years in ENTSOE
#Velsen 24 is a backup for Velsen 25 so they almost never produce at the same time
#Rijnmond 1 is twice in the data set


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
    

#Change the name of type_g:
condition = (final_df['type_g'] == 'Fossil Gas') & (final_df['Co2/Mwh (kg)'] < 450)
final_df.loc[condition, 'type_g'] = 'CCGT'

final_df.loc[final_df['type_g'] == 'Fossil Gas', 'type_g'] = 'Gas peaker'
final_df.loc[final_df['type_g'] == 'Fossil Hard coal', 'type_g'] = 'Coal'
    
#Add O&M cost:
frames = []
for i in list(set(final_df['type_g'])):
    df_type = final_df[final_df['type_g']== i]
    df_type['O & M cost']=parameters_techno.loc[i,'O&M cost (eur/Mwh)']
    df_type['Start-up cost']=parameters_techno.loc[i,'Start-up cost (eur)']
    frames.append(df_type)
final_df2 = pd.concat(frames)


#remove the plants specifically per year 
df_2019 = final_df2.copy()
df_2019.drop(['49W000000000089E', '49W000000000087I', '49W0000000000059'], axis = 0, inplace = True)


df_2022 = final_df2.copy()
df_2022.drop(['49W000000000038V'], axis = 0, inplace = True) #Hemweg 8 was stopped before 2022


'''change of min_load and ramp constraint with data analysis from ENTSOE Data'''


import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta

#download all the data of the generation per plant
API_template = "https://web-api.tp.entsoe.eu/api?securityToken=5bc1d6dc-704f-4344-ae56-596f12576409&documentType=A73&processType=A16&in_Domain=10YNL----------L&periodStart={}&periodEnd={}&PsrType={}"
URL = API_template.format('201901010000', '201901020000')
response_API = requests.get(URL)
data = response_API.text


ns = {'ns': 'urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0'}


Production_df = pd.DataFrame()
for PsrType in ['B05', 'B20']: #'B04'
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2019, 12, 31)
    all_dataframes = []
    
    current_date = start_date
    while current_date <= end_date:
        period_start = current_date.strftime('%Y%m%d%H%M')
        period_end = (current_date + timedelta(days=1)).strftime('%Y%m%d%H%M')
        
        # Format the URL for the current day
        url = API_template.format(period_start, period_end, PsrType)
        
        # Request data from the API
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch data for {current_date.date()}. Skipping...")
            current_date += timedelta(days=1)
            continue
        
        # Parse the XML response
        root = ET.fromstring(response.text)
        
        # Extract data for the day
        plants = []
        hourly_data = []
        
        for timeseries in root.findall('ns:TimeSeries', ns):
            plant_name = timeseries.find('.//ns:name', ns).text  # Plant name
            periods = timeseries.findall('.//ns:Point', ns)  # Hourly points
            
            plant_data = []
            for point in periods:
                quantity = point.find('ns:quantity', ns).text  # Production quantity
                plant_data.append(float(quantity))
            
            plants.append(plant_name)
            hourly_data.append(plant_data)
        
        # Create a DataFrame for the day
        daily_data = {plants[i]: hourly_data[i] for i in range(len(plants))}
        daily_df = pd.DataFrame(daily_data)
        daily_df['Hour'] = [i+1 for i in range(len(daily_df))]
        daily_df['Date'] = current_date.date()
        
        # Add the daily DataFrame to the list
        all_dataframes.append(daily_df)
        
        # Increment the day
        current_date += timedelta(days=1)
    
    # Concatenate all the DataFrames
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    Production_df = pd.concat([Production_df, final_df], axis = 1) 

Production_df  = Production_df.loc[:, ~Production_df.columns.duplicated()]

Production_df_both_year = pd.concat([Production_df_2019, Production_df_2022], ignore_index = True)
df_simple = Production_df_both_year.drop(columns=['Hour', 'Date'])
df_simple_2019 = Production_df_2019.drop(columns=['Hour', 'Date'])
df_simple_2022 = Production_df_2022.drop(columns=['Hour', 'Date'])

# Drop non-production columns (Hour and Date) to focus on production data
#create a common dataframe to merge the data
df_Entsoe = final_df2.drop(['name_p', 'country', 'eic_p'], axis = 1)
#change the names of plants to be the same in both file
updates = {
    1: 'Eems 20',
    2: 'Centrale Merwedekanaal 12',
    3: 'Eems 3',
    4: 'Eems 7',
    5: 'Eems 6',
    6: 'Eems 5',
    7: 'Eems 4',
    9: 'FLEVO 4',
    10: 'FLEVO 5',
    11: 'Sloecentrale 20',
    12: 'Rijnmond 1',
    18: 'Sloecentrale 10',
    19: 'Elsta 1',
    32: 'NLROTTETH__1'
}

for row_idx, name in updates.items():
    df_Entsoe.at[df_Entsoe.index[row_idx], 'name_g'] = name
df_Entsoe = df_Entsoe.set_index('name_g')


stable_load = {}
for data in [df_simple_2019, df_simple_2022]:
    for plant in data.columns:
        # Skip plants not in the capacity DataFrame
        if plant not in df_Entsoe.index:
            continue

        # Get production series and capacity
        production = data[plant]
        capacity = df_Entsoe.loc[plant, 'capacity_g']
        threshold = 0.05 * capacity

        # Find stable loads
        stable_loads = production[
            (production.shift(1) >= production) &  # x_t <= x_{t-1}
            (production.shift(-1) >= production) &  # x_t <= x_{t+1}
            (production > 0)  # x_t > 0
        ]
    
        # Filter stable loads above the threshold
        stable_loads_above_threshold = stable_loads[stable_loads > threshold]
        if plant in stable_load.keys():
            stable_load[plant] = pd.concat([stable_load[plant],stable_loads_above_threshold])
        else: 
            stable_load[plant] =  stable_loads_above_threshold

for plant in stable_load.keys():
    capacity = df_Entsoe.loc[plant,  'capacity_g']
    df_Entsoe.loc[plant, 'min_load_entsoe']= stable_load[plant].quantile(0.01)/capacity  #so min load is as a fraction of the capacity 

df_Entsoe['min_load'] = df_Entsoe['min_load_entsoe'].fillna(df_Entsoe['min_load'])
del df_Entsoe['min_load_entsoe']

#change ramp up and ramp down, 


# Example capacity DataFrame (user should provide the actual capacity data)
# capacity_df = pd.DataFrame({'Capacity': [500, 400, 300]}, index=['Plant A', 'Plant B', 'Plant C'])


# Calculate max percentage changes
max_changes = pd.DataFrame(columns = ['Max Upward Change', 'Max Downward Change'])
for data in [df_simple_2019, df_simple_2022]:
    for plant in data.columns:
        # Skip plants not in the capacity DataFrame
        if plant not in df_Entsoe.index:
            continue

        # Get production series and capacity
        production = data[plant]
        capacity = df_Entsoe.loc[plant, 'capacity_g']

        # Find stable loads
        if capacity > 0:  # Avoid division by zero
               # Normalize production by capacity
               normalized_production = production/ capacity
               
               # Compute percentage changes between consecutive hours
               percentage_changes = normalized_production.diff()
               
               # Find the maximum upward and downward changes
               max_up = percentage_changes.max()
               max_down = percentage_changes.min()
               
               # Store the results
               if plant in max_changes.index:
                   max_changes.loc[plant] = {
                       'Max Upward Change': max_up/60,  # Convert to percentage
                       'Max Downward Change': max_down/60 # Convert to percentage
                   }
               else: 
                   max_changes.loc[plant] = {
      'Max Upward Change': max(max_changes.loc[plant, 'Max Upward Change'], max_up / 60) if plant in max_changes.index else max_up / 60,
      'Max Downward Change': min(max_changes.loc[plant, 'Max Downward Change'], max_down / 60) if plant in max_changes.index else max_down / 60
  }



df_Entsoe = df_Entsoe.merge(max_changes, how =  'left', left_index=True, right_index=True)

df_Entsoe['ramp_up'] = df_Entsoe['Max Upward Change'].fillna(df_Entsoe['ramp_up'])
del df_Entsoe['Max Upward Change']
df_Entsoe['ramp_down'] = df_Entsoe['Max Downward Change'].fillna(df_Entsoe['ramp_down'])
del df_Entsoe['Max Downward Change']


#Do differenciation in terms of efficiency 
df_Entsoe['eff min'] = df_Entsoe['eff']*df_Entsoe['min_load']
df_Entsoe['eff avg'] = (df_Entsoe['eff min'] + df_Entsoe['eff'])/2
df_Entsoe['Co2/Mwh (kg) max'] = df_Entsoe['Co2/Mwh (kg)']*(df_Entsoe['eff avg']/df_Entsoe['eff min'])
df_Entsoe['Co2/Mwh (kg) min'] = df_Entsoe['Co2/Mwh (kg)']*(df_Entsoe['eff avg']/df_Entsoe['eff'])


Entsoe_2019 = df_Entsoe.copy()
Entsoe_2019.drop(['Eems 3', 'Eems 4', 'Claus C'], axis = 0, inplace = True)


Entsoe_2022 = df_Entsoe.copy()
Entsoe_2022.drop(['Hemweg 8'], axis = 0, inplace = True) #Hemweg 8 was stopped before 2022


Entsoe_2019.to_csv("C:\\Users\\..\\Parameters_plants_2019.csv")
Entsoe_2022.to_csv("C:\\Users\\...\\Parameters_plants_2022.csv")

    


 
