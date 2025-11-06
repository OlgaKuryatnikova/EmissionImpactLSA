"""

This file take results from the optimisation compile and generate the results of the paper
To compile this file, the folder needs to be set like explain in readme 
"""
# remove all existing variables
from IPython import get_ipython
get_ipython().magic('reset -sf')


import os

#To complete:
userpath = "C:\\Users\\"

path_file_source= os.path.join(userpath,"Data\\source\\")
path_file_optimisation = os.path.join(userpath,"Data\\result_optimisation\\All\\") #this is a folder for new files, then needs to be copy in the right folder, I do that to not overwrite some files
path_file_result = os.path.join(userpath,"Data\\result\\")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PIL import Image
class Object(object):
    pass


'''load source files and some adjustemnts'''
par_temp_ori = {}
par_temp_ori[2019] = pd.read_csv(f"{path_file_source}Parameters_plants_2019.csv", sep=',', index_col = 0, usecols=[2,3,4,7, 8,9,12,14,15,16, 17, 18, 19,20])
par_temp_ori[2022] = pd.read_csv(f"{path_file_source}Parameters_plants_2022.csv", sep=',', index_col = 0, usecols=[2,3,4,7, 8,9,12,14,15,16,17,18, 19, 20])
Prices = pd.read_csv(f"{path_file_source}Prices.csv", sep=',', index_col=0)

ets={}
ets[2019] = Prices.loc[0:365, 'ETS'].mean()
ets[2022] = Prices.loc[1096:1461,'ETS'].mean()


#demand file
D_full = pd.read_csv(f"{path_file_source}Load_2019_2022.csv", skipinitialspace=True, usecols=[2])
D_full = D_full.values/1000  #put in GwH

#renewable file
Q_r_full = pd.read_csv(f"{path_file_source}Renewable_2019_2022.csv", skipinitialspace=True, usecols=[1, 2])
Q_r_full = Q_r_full/1000


#Replace nans with approximations in Demand and renewable files
# Remove nans in the data by taking the averages between the two values before and after nan, our data allows for this
# the rows where the data on renewables and demand is missing
ind_nan = np.where(np.isnan(D_full) == True)
ind_nan = ind_nan[0]
for i in ind_nan:
    Q_r_full["Wind MWH"][i] = (
        Q_r_full["Wind MWH"][i-1] + Q_r_full["Wind MWH"][i+1])/2
    Q_r_full["Solar MWH"][i] = (
        Q_r_full["Solar MWH"][i-1] + Q_r_full["Solar MWH"][i+1])/2
    D_full[i] = (D_full[i-1] + D_full[i+1])/2


D_2019 = D_full[0:8760]
Wind = np.reshape((Q_r_full['Wind MWH'][0:8760]).values, (8760, 1))
Solar = np.reshape(
    (Q_r_full['Solar MWH'][0:8760]).values, (8760, 1))
Dr_2019 = D_2019 - Wind - Solar - 0.447  # Residual Demand, 4447 is the generation from waste which is a baseload
Dr_2019[Dr_2019 > par_temp_ori[2019]['capacity_g'].cumsum().iloc[-1]/1000] = 15.372



D_2022 = D_full[1096*24:1461*24+24]
Wind = np.reshape((Q_r_full['Wind MWH'][1096*24:1461*24+24]).values, (8760, 1))
Solar = np.reshape(
    (Q_r_full['Solar MWH'][1096*24:1461*24+24]).values, (8760, 1))
Dr_2022 = D_2022 - Wind - Solar - 0.447  # Residual Demand
Dr_2022[Dr_2022 > par_temp_ori[2022]['capacity_g'].cumsum().iloc[-1]/1000]  = 16.771


'''Set up main prameters'''
LSA_l = ['Hydrogen', 'PHS', 'CAES', 'Battery', 'Perfect ESS']
par = Object() # This is to store all the elements that are stable through the days
par.T = 24  # number of periods per day
par.eta_b = np.sqrt(
    np.matrix([0.375, 0.76, 0.795, 0.9, 1])
)  
# share of the generated energy that reaches the customer when discharged
par.eta_s = par.eta_b #assumed to be the same
par.c_bs = np.matrix([0.5, 0.11, 1.55, 1.25, 0.01]) # LSA energy sell cost
par.c_bb = par.c_bs  # LSA energy buy cost assume to be the same as sell
par.rate = np.matrix([1, 0.12, 0.19, 0.57, 1])   #how much energy (as percent of the capacity) can be charged or discharge in an hour
par.pmin = 0 # price of the renewable
par.n_f = len(par_temp_ori)  # number of fossil fuel operators, do not include renewables
par.n_tot = par.n_f + 1 #number of generators, including renewables


'''Functions used in the file'''    

#function to open the files resulting from the optimization          
def open_file(lsa,  mult_lsa = 1,  year = 2019, tax ='ETS', subsidy = 0, mult_s = 1.0, mult_w = 1.0, ramp = False, min_load = False, start_cost = False, n_p = 1):
    if year == 2019:
        add_days = 0
    else:
        add_days = 365*2+366
    file = []
    text = f'{path_file_optimisation}sol_sp_cap_'+','.join(str(mult_lsa).split('.'))+'_'+'_'.join(LSA_l[lsa].split(' '))+'_'+str(year)+'_tax_'+','.join(str(tax).split('.'))+'_solar_'+','.join(str(mult_s).split('.'))+'_wind_'+','.join(str(mult_w).split('.'))+'_ramp_'+str(ramp)+'_min_load_'+str(min_load)+'_start_c_'+str(start_cost)+'_n_piece_'+str(n_p)+'_'+str(0 + add_days)+'_'+str(add_days +  365)
    with open(text, 'rb') as fr:
        try:
            while True:
                  file.append(pickle.load(fr))
        except EOFError:
            pass  
    return(file)
 
#function to get from the uploaded file an hourly summary of the storage operation and its emission impact. 
def agg_sol(sol , sol_wobda,T = 24, n_p =1):
    q_dif_agg = pd.DataFrame()
    if n_p ==1:
        l = 24
    else:
        l = 72
    emis = np.concatenate([[0 if np.round(i,4) == 0 else i for i in np.reshape(sol[kk].var.emis_hourly,(l,)).tolist()[0]] for kk in range(len(sol))])
    if sol_wobda != sol:
        emis_wobda = list(np.concatenate([[0 if np.round(i,4) == 0 else i for i in np.reshape(sol_wobda[kk].var.emis_hourly,(l,)).tolist()[0]] for kk in range(len(sol_wobda))]))
    else:
        #q_dif_agg['Price'] = np.concatenate([sol[kk].var.price.flatten() for kk in range(len(sol))]).ravel().tolist()[0]
        emis_wobda = list(np.concatenate([[0 if np.round(i,4) == 0 else i for i in np.reshape(sol[kk].var.emis_hourly_wobda,(l,)).tolist()[0]] for kk in range(len(sol))]))
    if 'price_dual' not in dir(sol[0].var):
        q_dif_agg['Price'] = list(np.concatenate([[0 if np.round(i,4) == 0 else i for i in np.reshape(sol[kk].var.price,(24,)).tolist()[0]] for kk in range(len(sol))]))
    else:
        q_dif_agg['Price'] = list(np.concatenate([[0 if np.round(i,4) == 0 else i for i in np.reshape(sol[kk].var.price_dual,(24,)).tolist()] for kk in range(len( sol))]))

    if n_p == 1:
        q_dif_agg['Emission'] = list(emis)
        emis_dif = emis - emis_wobda
        q_dif_agg['diff emis'] = list(emis_dif)
    else:
        q_dif_agg['Emission'] =  list(emis.reshape(8760,3).sum(axis = 1))
        emis_dif = emis - emis_wobda 
        q_dif_agg['diff emis'] = list(emis_dif.reshape(8760,3).sum(axis = 1))    
    q_dif_agg['Buy LSA']= list(np.concatenate([[0 if np.round(i,4) == 0 else i for i in np.reshape(sol[kk].var.w_b,(24,)).tolist()[0]] for kk in range(len(sol))]))
    q_dif_agg['Sell LSA']= list(np.concatenate([[0 if np.round(i,4) == 0 else i for i in np.reshape(sol[kk].var.q_b,(24,)).tolist()[0]] for kk in range(len(sol))]))
    mask_col1_greater = q_dif_agg['Buy LSA'] > q_dif_agg['Sell LSA']
    # Set col2 to 0 where col1 is greater, and col1 to 0 where col2 is greater or equal
    q_dif_agg['Sell LSA'] = q_dif_agg['Sell LSA'].where(~mask_col1_greater, 0)
    q_dif_agg['Buy LSA'] = q_dif_agg['Buy LSA'].where(mask_col1_greater, 0)
    q_dif_agg.loc[(q_dif_agg['Buy LSA'] == 0) & (q_dif_agg['Sell LSA'] == 0), 'diff emis'] = 0
    q_dif_agg['soc']= np.concatenate([sol[kk].var.s[1:].flatten() for kk in range(len(sol))]).ravel().tolist()[0]
    q_dif_agg['obj']= list(np.concatenate([[sol[kk].obj]*24  for kk in range(len(sol))]))
    q_dif_agg['Profit'] = (q_dif_agg['Sell LSA']-q_dif_agg['Buy LSA'])*q_dif_agg['Price']
    q_dif_agg['Days'] =  [i for i in range(len(sol)) for _ in range(24)]
    #q_dif_agg['Residual Demand'] = np.concatenate([sol_sp[kk].D_r.flatten() for kk in range(len(sol_sp))]).ravel().tolist()
    #q_dif_agg['Curtailment'] =-(q_dif_agg['Buy LSA']  +  q_dif_agg['Residual Demand']).where( q_dif_agg['Residual Demand'] + q_dif_agg['Buy LSA']   <0,-0)
    return(q_dif_agg)


#daily aggregation of the hourly summary 
def diff_agg_daily (name_input, sub = False):
    result_daily = pd.DataFrame()
    result_daily['Emission'] = name_input.groupby('Days')['Emission'].sum() 
    result_daily['diff emis'] = name_input.groupby('Days')['diff emis'].sum()
    result_daily['Buy LSA'] = name_input.groupby('Days')['Buy LSA'].sum()
    result_daily['Sell LSA'] = name_input.groupby('Days')['Sell LSA'].sum()
    result_daily['obj'] = name_input.groupby('Days')['obj'].mean()
    #result_daily['Residual Demand'] = name_input.groupby('Days')['Residual Demand'].sum() 
    #result_daily['Curtailment'] = name_input.groupby('Days')['Curtailment'].sum() 
    result_daily['Profit'] = name_input.groupby('Days')['Profit'].sum()  
    return (result_daily)


#function to couple transaction two by two and gives the emission impact of each transaction 
def coupling(sol, sol_wobda, LSA_nb, col_em):
    transaction_amount = []
    transaction_impact = []
    period_buy = []
    period_sell = []
    T_Buy = []
    T_Sell = []
    Days = []
    efficiency = par.eta_b[0,LSA_nb]**2
    day_to_run = []
    for day in range(0, 365):
        test = sol[day]
        if sol == sol_wobda:
            change_prod = test.var.q_f - test.var.q_f_wobda
        else:
            no_lsa = sol_wobda[day]
            change_prod = test.var.q_f - no_lsa.var.q_f
        change_prod =  np.where(np.round(change_prod, 3) == 0, 0, change_prod)
        if  col_em == "3_p":
            new_array = np.zeros((24, len(change_prod.T)*3))
            # Fill the new array
            for i in range(24):  # 24 blocks
                block = change_prod[i*3:(i+1)*3, :]  # shape (3, 39)
                new_array[i, :] = block.T.flatten() 
            change_prod = new_array
        change_prod = np.hstack((change_prod,  np.arange(24).reshape(-1, 1) ))
        #operations = np.where(np.round(test.var.q_b + test.var.w_b,3) != 0)[0]
        change_prod = np.hstack((change_prod, np.array(- test.var.q_b + test.var.w_b)))
        x = change_prod[np.round(change_prod[:,-1],3) != 0,:]
        #if sum([any(row > 0) and any(row < 0) for row in x[:,:-2]]) != 0:
        #    day_to_run.append(day)
        #    pass
           # x = np.hstack((x,  x[:,:-1].sum(axis = 1).reshape(-1,1)))
        if sum([any(row > 0) and any(row < 0) for row in x[:,:-2]]) != 0:
              day_to_run.append(day)
              pass
        else:
            nonzero_cols_per_row = [np.nonzero(row)[0].tolist() for row in x[:,:-2]]
                #you should check if it fits with the hour buying and selling
            Buy = np.where(x[:,-1] > 0)[0]
            Sell = np.where(x[:,-1] < 0)[0]
            Buy_p = [ int(i) for i in x[np.where(x[:,-1] > 0)[0],-2]]
            Sell_p = [int(i) for i in x[np.where(x[:,-1] < 0)[0],-2]]
            n = 0
            r = 0
            if col_em == "3_p":
                emission = test.par_temp[:,[6,9,10]].flatten().reshape(-1, 1)
                emission[0:3] = 0
                emission = np.vstack([np.zeros((3, 1)), emission])
                plant_type =  np.repeat(test.par_temp[:,1], repeats=3, axis=0).reshape(-1, 1)
                plant_type = np.vstack([['RES'], plant_type])
                plant_type = np.vstack([['RES'], plant_type])
                plant_type = np.vstack([['RES'], plant_type])
                emission_change = x[:,:-2]*emission.T
            else:
                plant_type = np.insert(test.par_temp[:,1],0, 'RES')
                emission_change = x[:,:-2]*np.insert(test.par_temp[:,col_em],0,0)
            for q, i in enumerate(Sell):
                for j in nonzero_cols_per_row[i]:
                    while   x[i, j] !=0 and np.count_nonzero(np.round(x[:,-1], 2)) != 0 : 
                        if -x[i, j] > x[Buy[n], nonzero_cols_per_row[Buy[n]][r]]*efficiency:#sell columns is higher
                            transaction_amount.append(x[Buy[n], nonzero_cols_per_row[Buy[n]][r]]*efficiency)
                            transaction_impact.append(emission_change[i, j]*(-transaction_amount[-1]/x[i, j])+   emission_change[Buy[n], nonzero_cols_per_row[Buy[n]][r]])
                            emission_change[i,j] += emission_change[i, j]*(transaction_amount[-1]/x[i, j])
                            x[Buy[n], nonzero_cols_per_row[Buy[n]][r]] = 0
                            x[i, j] = x[i,j] + transaction_amount[-1]
                            x[Buy[n], -1] = x[Buy[n], -1] - transaction_amount[-1]/efficiency
                            x[i, -1] =  x[i, -1] +  transaction_amount[-1]
                            period_buy.append(Buy_p[n])
                            period_sell.append(Sell_p[q])
                            T_Sell.append(plant_type[j])
                            Days.append(day)
                            T_Buy.append(plant_type[nonzero_cols_per_row[Buy[n]][r]])
                            if r == len(nonzero_cols_per_row[Buy[n]])-1:
                                n = n + 1 
                                r = 0
                            else:
                                r = r+1
                                        
                        else:
                            transaction_amount.append(-x[i, j])
                            transaction_impact.append(emission_change[i, j] +   emission_change[Buy[n], nonzero_cols_per_row[Buy[n]][r]] * (transaction_amount[-1]/efficiency/x[Buy[n], nonzero_cols_per_row[Buy[n]][r]]))
                            emission_change[Buy[n], nonzero_cols_per_row[Buy[n]][r]] -=   emission_change[Buy[n], nonzero_cols_per_row[Buy[n]][r]] * (transaction_amount[-1]/efficiency/x[Buy[n], nonzero_cols_per_row[Buy[n]][r]])
                            x[Buy[n], nonzero_cols_per_row[Buy[n]][r]] = x[Buy[n], nonzero_cols_per_row[Buy[n]][r]] - transaction_amount[-1]/efficiency
                            x[i, j] = 0
                            x[Buy[n], -1] = x[Buy[n], -1] - transaction_amount[-1]/efficiency
                            x[i, -1] =  x[i, -1] + transaction_amount[-1]
                            period_buy.append(Buy_p[n])
                            T_Buy.append(plant_type[nonzero_cols_per_row[Buy[n]][r]])
                            period_sell.append(Sell_p[q])
                            T_Sell.append(plant_type[j])
                            Days.append(day)
                           
    pollution_rate = [a/b for a,b in zip(transaction_impact, transaction_amount)]
    combined_techno = [(x, y) for x, y in zip(T_Buy, T_Sell)]
    df = pd.DataFrame({'Transaction Amount': transaction_amount, 'Transaction Impact': transaction_impact, 'Period Buy' : period_buy, 'Period Sell' : period_sell, 'Pollution rate' : pollution_rate, 'transaction_type': combined_techno, 'Day': Days})
    return({'df' : df, 'to_run':day_to_run})



#function that couple transaction but do not give emission impact, used for opti with constraint to recompute after the change in emission. 
def coupling_compute(sol,LSA_nb):
    trans={}
    for day in range(0, 365):
        trans[day] = []
        D_b = sol[day].var.w_b + 0
        D_s = sol[day].var.q_b + 0
        for t in range(24):
            # if t is a selling period
            if sol[day].var.q_b[t]  > 1e-4:
                # hour_sell.add(t)
                while D_s[t] > 1e-4:
                    # find closest buying period
                    k = min(np.where(D_b > 1e-4)[0])
                    # hour_buy.add(k)
                    if D_s[t] > D_b[k]*par.eta_s[0,LSA_nb]**2:
                        trans[day].append([(D_b[k]*par.eta_s[0,LSA_nb]**2 + 0)[0,0],k,t])
                        D_s[t] = D_s[t] - D_b[k]*par.eta_s[0,LSA_nb]**2
                        D_b[k] = 0 
                    else:
                        trans[day].append([(D_s[t] + 0)[0,0],k,t])
                        D_b[k] = D_b[k] - D_s[t]/par.eta_s[0,LSA_nb]**2
                        D_s[t] = 0
            elif sol[day].var.w_b[t] > 1e-4:
                # if t is a buying period
                # hour_buy.add(t)
                while D_b[t] > 1e-4:
                    # find closest selling period
                    u = min(np.where(D_s > 1e-4)[0])
                    # hour_sell.add(u)
                    if D_s[u] > D_b[t]*par.eta_s[0,LSA_nb]**2:
                        trans[day].append([(D_b[t]*par.eta_s[0,LSA_nb]**2 + 0)[0,0],t,u])
                        D_s[u] = D_s[u] - D_b[t]*par.eta_s[0,LSA_nb]**2
                        D_b[t] = 0
                    else:
                        trans[day].append([(D_s[u] + 0)[0,0],t,u])
                        D_b[t] = D_b[t] - D_s[u]/par.eta_s[0,LSA_nb]**2
                        D_s[u] = 0
        #trans[day] = np.array(trans[day])        
    return(trans)


#to find theoretical intervals of pollution rate of a storage
def P_theoric_f(alpha, cl, par_temp, n_p = 1):
  if n_p == 3:
      plants_df = np.concatenate(( par_temp[:,[6,9,10]].flatten().reshape(-1, 1), par_temp[:,-3:].flatten().reshape(-1, 1)), axis = 1)
      plants_df = np.insert(plants_df,0,[0,0], axis = 0)
      plants_df = np.insert(plants_df,0,[0,0], axis = 0)
      plants_df = np.insert(plants_df,0,[0,0], axis = 0)
      plants_df[3:6,0] = 0
      plants_df = plants_df[plants_df[:, -1].argsort()]    
      indice_em = 0
  else:
      plants_df = np.insert(par_temp,0,[0,'RES'] + [0]*(len(par_temp.T)-2), axis = 0)
      indice_em = 6
  P_possibilities_c = np.zeros((len(plants_df),len(plants_df)), dtype=float)
  P_possibilities_c[:] = np.nan
  for i in range(0,len(plants_df))      :  
        for j in range(i+1, len(plants_df))     :       
            if plants_df[i,-1]+cl<(plants_df[j,-1]-cl)*alpha: # cm+cl>(cn-cl) alpha
               P_possibilities_c[i,j] = (plants_df[i,indice_em]/alpha - plants_df[j,indice_em])
  return(np.nanmin(P_possibilities_c), np.nanmax(P_possibilities_c))


#to find change in emission between two different optimisation, gives the change in terms of type of generators production and emission
def change_ss_c(sol, sol_comparison, n_p = 1):
    change = pd.DataFrame(columns = ['Coal', 'Gas peaker', 'Gas_cc', 'Nuclear', 'RES'])
    for obs in range(365):
        #if round(sum(sol[obs].var.q_b).item(),3) == 0:
          #  rows_to_add = pd.DataFrame(np.zeros((24, change.shape[1])), columns=change.columns)
          #  change = pd.concat([change, rows_to_add], axis = 0)
       # else:
        q_f = sol[obs].var.q_f
        if n_p == 3:
            q_f = np.array(q_f).reshape(24, 3, len(q_f.T)).sum(axis = 1)
        x = q_f - sol_comparison[obs].var.q_f
        x = np.where(np.round(x, 3) == 0, 0, x)
        df = pd.DataFrame(x)
        df.columns = np.insert(sol[obs].par_temp[:,1],0, 'RES', axis=0)
        grouped_df = df.groupby(df.columns, axis=1).sum()
        grouped_df.index = range(len(change), len(change)+24)
        change = pd.concat([change, grouped_df], axis = 0)
    change.columns = [x if x!= 'Gas_cc' else 'CCGT' for x in change.columns]
    if n_p == 3:
        emis = np.concatenate([[0 if np.round(i,4) == 0 else i for i in np.reshape(sol[kk].var.emis_hourly,(72,)).tolist()[0]] for kk in range(len(sol))])
        change['emis_total'] = emis.reshape(8760,3).sum(axis = 1)
    else:
        change['emis_total'] = np.concatenate([[0 if np.round(i,4) == 0 else i for i in np.reshape(sol[kk].var.emis_hourly,(24,)).tolist()[0]] for kk in range(len(sol))])
    change['emis comparison'] = list(np.concatenate([[0 if np.round(i,4) == 0 else i for i in np.reshape(sol_comparison[kk].var.emis_hourly,(24,)).tolist()[0]] for kk in range(len(sol_comparison))]))
    change['Buy'] =  list(np.concatenate([[0 if np.round(i,4) == 0 else i for i in np.reshape(sol[kk].var.w_b,(24,)).tolist()[0]] for kk in range(len(sol))]))
    change['Sell'] =  list(np.concatenate([[0 if np.round(i,4) == 0 else i for i in np.reshape(sol[kk].var.q_b,(24,)).tolist()[0]] for kk in range(len(sol))])) 
    #change['obj after'] = list(np.concatenate([[sol[kk].obj]*24  for kk in range(len(sol))]))
    #change['obj before'] = list(np.concatenate([[sol_comparison[kk].obj]*24  for kk in range(len(sol_comparison))]))
    return change.sum(axis = 0)   



#parameters used for the style of the graphs
linestyles = ['dashdot', 'dotted', 'dashed', (0, (3, 1, 1, 1, 1, 1)), 'solid']
colors = ['darkgray', 'dimgray', 'gray', 'black', 'silver']

linestyles_dict = {'PHS': 'dashdot', 'CAES': 'dotted', 'Battery':'dashed', 'Perfect ESS': (0, (3, 1, 1, 1, 1, 1)), 'Hydrogen': 'solid'}
colors_dict = {'PHS': 'darkgray', 'CAES': 'dimgray', 'Battery':'gray', 'Perfect ESS': 'black', 'Hydrogen': 'silver', 'No Lsa': 'gainsboro'}

# ==========================================================
# Graphs
# =============================================================================
    

# ==========================================================
# Figure 2 Merit order curve
# =============================================================================
sol_sp_basic  = {}
sol_sp_basic[2019] = open_file(lsa = 3, year = 2019, mult_lsa = 0)
sol_sp_basic[2022] = open_file(lsa = 3, year = 2022, mult_lsa = 0)

#compute the marginal cost for each day and for each fuel type' 
df_marginal_emission = {}
df_capacity_emission ={}
df_marginal_cost = {}
df_plants_count = {}
marginal_cost_gas = {}
marginal_cost_coal = {}
marginal_cost_gas_cc = {}
marginal_cost_nuclear = {}
for year in [2019,2022]:
    par_temp_0 = sol_sp_basic[year][0].par_temp
    marginal_emission = np.empty(shape=(365,len(par_temp_0)*2))
    capacity_emission = np.empty(shape=(365,len(par_temp_0)*2))
    marginal_cost = np.empty(shape=(365,len(par_temp_0)*2))
    df_plants = pd.DataFrame(columns = range(len(par_temp_0)*2))
    df_plants_count[year] = pd.DataFrame(columns = ['Nuclear', 'Coal', 'Gas Peaker', 'Gas_cc'])
    for obs in range(0,365):
        par_temp = sol_sp_basic[year][obs].par_temp
        marginal_emission[obs] = np.repeat(par_temp[:,6], 2)
        capacity =  np.repeat(par_temp[:,0].cumsum(), 2)   
        capacity = np.insert(capacity[0:-1], 0, 0)     
        capacity_emission[obs] = capacity
        marginal_cost[obs] = np.repeat(par_temp[:,-1], 2)
        df_plants.loc[obs] = np.repeat(par_temp[:,1], 2)
    df_marginal_emission[year] = marginal_emission.mean(axis = 0) #yearly average
    df_capacity_emission[year] = capacity_emission.mean(axis = 0)
    df_marginal_cost[year] = marginal_cost.mean(axis = 0)
  #You need to get the cost such that it represents the share of the technology at that time
# Iterate through each column and calculate value counts
    for col in df_plants.columns:
        df_plants_count[year].loc[col] = df_plants[col].value_counts()/365
    df_plants_count[year] = df_plants_count[year].fillna(0)
    marginal_cost_nuclear[year] = df_marginal_cost[year]*df_plants_count[year]['Nuclear']
    marginal_cost_gas[year] = df_marginal_cost[year]*df_plants_count[year]['Gas Peaker']
    marginal_cost_coal[year] = df_marginal_cost[year]*df_plants_count[year]['Coal']
    marginal_cost_gas_cc[year] = df_marginal_cost[year]*df_plants_count[year]['Gas_cc']
    df_capacity_emission[year] = df_capacity_emission[year]/1000


#graph 2019
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0, height_ratios = [0.20, 0.8])
axs = gs.subplots(sharex=True)
axs[0].boxplot(Dr_2019, notch = False, sym ='', vert = False, labels = [''])
axs[0].set_axis_off()   
axs[1].scatter(df_capacity_emission[2019],df_marginal_cost[2019], alpha = 0)
axs[1].fill_between(list(df_capacity_emission[2019]), list(marginal_cost_nuclear[2019]),  facecolor='silver', label = 'Nuclear', step = 'pre')
axs[1].fill_between(list(df_capacity_emission[2019]),  list(marginal_cost_coal[2019]), list(marginal_cost_coal[2019] + marginal_cost_gas_cc[2019]), facecolor='darkgrey', label = 'CCGT')
axs[1].fill_between(list(df_capacity_emission[2019]), list(marginal_cost_coal[2019] + marginal_cost_gas_cc[2019] + marginal_cost_nuclear[2019]),  list(df_marginal_cost[2019]), facecolor='dimgrey', label = 'Gas Peaker')
axs[1].fill_between(list(df_capacity_emission[2019]), list(marginal_cost_coal[2019]),  facecolor='black', label = 'Coal', step = 'pre')
ax2 = axs[1].twinx() 
ax2.set_ylabel('CO2 emission (ton/Gwh)')
ax2.plot(list(df_capacity_emission[2019]), list(df_marginal_emission[2019]), color = 'black')  
axs[1].legend(loc = 'upper left')
ax2.legend()
axs[1].set_frame_on(False)
axs[1].set_xlabel('Capacity (Gwh)')
axs[1].set_ylabel('Price (1000€/GWh)')
ax2.set_frame_on(False)

#graph 2022
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0, height_ratios = [0.20, 0.8])
axs = gs.subplots(sharex=True)
axs[0].boxplot(Dr_2022, notch = False, sym ='', vert = False, labels = [''])
axs[0].set_axis_off()   
axs[1].scatter(df_capacity_emission[2022],df_marginal_cost[2022], alpha = 0)
axs[1].fill_between(list(df_capacity_emission[2022]), list(marginal_cost_nuclear[2022]),  facecolor='silver', label = 'Nuclear', step = 'pre')
axs[1].fill_between(list(df_capacity_emission[2022]), list(marginal_cost_coal[2022]),  facecolor='black', label = 'Coal', step = 'pre')
axs[1].fill_between(list(df_capacity_emission[2022]),  list(marginal_cost_coal[2022]), list(marginal_cost_coal[2022] + marginal_cost_gas_cc[2022]), facecolor='darkgrey', label = 'CCGT')
axs[1].fill_between(list(df_capacity_emission[2022]), list(marginal_cost_coal[2022] + marginal_cost_gas_cc[2022] + marginal_cost_nuclear[2022]),  list(df_marginal_cost[2022]), facecolor='dimgrey', label = 'Gas Peaker')

ax2 = axs[1].twinx() 
ax2.set_ylabel('CO2 emission (ton/Gwh)')
ax2.plot(list(df_capacity_emission[2022]), list(df_marginal_emission[2022]), color = 'black', label = 'Average marginal emission')   
ax2.legend(loc = 'upper right')
axs[1].set_frame_on(False)
axs[1].set_xlabel('Capacity (Gwh)')
axs[1].set_ylabel('Price (1000€/GWh)')
ax2.set_frame_on(False)


# ==========================================================
#Figure 3 :  impact LSA with sensitivity to capacity
# =============================================================================


# Loading the results of the optimisation for different capacity
#load the result files
sol_sp_cap = {}
for year in [2019, 2022]:
    sol_sp_cap[year] = {}
    for j in range(1,11):
        print(j)
        sol_sp_cap[year][j]={}
        for i in [0,1,2,3,4]:
            sol_sp_cap[year][j][LSA_l[i]]=open_file(lsa = i,  mult_lsa = j, year = year, tax = 'ETS')
    sol_sp_cap[year]['No Lsa'] = open_file(3, mult_lsa = 0, year = year)



#Compute result in a dataframe
result_sp_cap={}
for year in [2019, 2022]:
    result_sp_cap[year]={}
    for j in range(1,11):
        #print(j)
        result_sp_cap[year][j]={}
        for i in  [0,1,2,3,4]:
            result_sp_cap[year][j][LSA_l[i]] = agg_sol(sol_sp_cap[year][j][LSA_l[i]], sol_sp_cap[year]['No Lsa'])

        
#Compute the daily result     
result_sp_cap_daily={}
for year in [2019, 2022]:
    result_sp_cap_daily[year]={}
    for j in range(1,11):
        #print(j)
        result_sp_cap_daily[year][j]={}
        for i in [0,1,2,3,4]:
            result_sp_cap_daily[year][j][LSA_l[i]] = diff_agg_daily(result_sp_cap[year][j][LSA_l[i]])
         

#compute the yearly result           
result_sp_cap_period_all={}
for year in [2019, 2022]:
    result_sp_cap_period_all[year]={}
    for j in range(1,11):
       # print(j)
        result_sp_cap_period_all[year][j]=pd.DataFrame()
        for i in [0,1,2,3,4]:
            result_sp_cap_period_all[year][j][LSA_l[i]] = round(result_sp_cap_daily[year][j][LSA_l[i]].sum(),5)
   

#save the overall result
f = open(f'{path_file_result}result_sp_cap_period_all_AP.pkl',"wb")
pickle.dump(result_sp_cap_period_all,f)

#and load it
with open(f'{path_file_result}result_sp_cap_period_all_AP.pkl', 'rb') as file:
        result_sp_cap_period_all = pickle.load(file)

#summary of emission impact for different storage capacity
Emission_capacity_2019=pd.DataFrame()   
for j in range(1,11): 
    for i in [0,1,2,3,4]:
        Emission_capacity_2019.loc[j,LSA_l[i]]=result_sp_cap_period_all[2019][j][LSA_l[i]]['diff emis']/result_sp_cap_period_all[2019][j][LSA_l[i]]['Emission']*100
Emission_capacity_2019['index']=([10,20,30,40,50,60,70,80,90,100])
Emission_capacity_2019.set_index('index', inplace = True)
for i, lsa in enumerate(Emission_capacity_2019.columns):
    Emission_capacity_2019[lsa].plot(color=colors_dict[lsa], linestyle=linestyles_dict[lsa])
plt.legend()
plt.ylabel('Storage marginal CO2 emission (%)')
plt.xlabel('Share of daily average demand (%)')
plt.ylim(-0.85, 0.5)


Emission_capacity_2022=pd.DataFrame()   
for j in range(1,11): 
    for i in [0,1,2,3,4]:
        print(i)
        Emission_capacity_2022.loc[j,LSA_l[i]]=result_sp_cap_period_all[2022][j][LSA_l[i]]['diff emis']/result_sp_cap_period_all[2022][j][LSA_l[i]]['Emission']*100
Emission_capacity_2022['index']=([10, 20,30,40,50,60,70,80,90,100])
Emission_capacity_2022.set_index('index', inplace = True)
for i, lsa in enumerate(Emission_capacity_2022.columns):
    Emission_capacity_2022[lsa].plot(color=colors_dict[lsa], linestyle=linestyles_dict[lsa])
plt.legend()
plt.ylabel('Storage marginal CO2 emission (%)')
plt.xlabel('Share of daily average demand (%)')
plt.ylim(-0.85, 0.5)


# ==========================================================
#Figure 4 :Transaction-wise rate of added emission (dependence: 'result_sp_cap' from figure 3)
# =============================================================================

           
couple_sp = {}
for year in [2019, 2022]:
    couple_sp[year] = {}
    for i in [0,1,2,3,4]:
        print(i)
        couple_sp[year][LSA_l[i]] = {}
        for cap in range(1,2):
            couple_sp[year][LSA_l[i]][cap]  = coupling(sol_sp_cap[year][cap][LSA_l[i]],sol_sp_cap[year][cap][LSA_l[i]],i,6 )
 
f = open(f"{path_file_result}couple_sp.pkl","wb")
pickle.dump(couple_sp,f)


with open(f'{path_file_result}couple_sp.pkl', 'rb') as file:
        couple_sp = pickle.load(file)

#It gives you the max pollution rate per day 
P_theoric = {}
P_theoric_min = {}
for year in [2019, 2022]:
    P_theoric[year] = pd.DataFrame(columns = [LSA_l[0], LSA_l[1], LSA_l[2], LSA_l[3], LSA_l[4]])
    P_theoric_min[year] = pd.DataFrame(columns = [LSA_l[0], LSA_l[1], LSA_l[2], LSA_l[3], LSA_l[4]])
    for obs in range(0,365):
        par_temp = sol_sp_basic[year][obs].par_temp
        P_theo = [P_theoric_f(par.eta_b[0,lsa]**2, par.c_bb[0,lsa], par_temp) for lsa in [0,1,2,3,4]]
        P_theoric[year].loc[obs] = [P_theo[lsa][1] for lsa in range(5)]
        P_theoric_min[year].loc[obs] = [P_theo[lsa][0] for lsa in range(5)]

      

P_max_theoric = pd.DataFrame(data = {2019: P_theoric[2019].max(), 2022: P_theoric[2022].max()})


#graph
#2019
theoric_list = [P_max_theoric[2019][LSA_l[lsa]]for lsa in range(5)]
theoric_min = [P_theoric_min[2019][LSA_l[lsa]].loc[0] for lsa in range(5)]
list_lsa = [LSA_l[0], LSA_l[1], LSA_l[2], LSA_l[3], LSA_l[4]]
fig, ax = plt.subplots()
values = [couple_sp[2019][LSA_l[lsa]][1]['Pollution rate'] for lsa in range(5)]
ax.boxplot(values, labels=list_lsa)
ax.scatter(range(1, 6), theoric_list,marker='^', label='Upper theoretical bound', color='grey', zorder=4,s=50)
ax.scatter(range(1, 6), theoric_min, marker='v', label='Lower theoretical bound', color='grey', zorder=4, s = 50)
plt.ylabel('CO2 emission rate (ton/Gwh)')
plt.ylim(-900, 2600)
plt.legend()

#2022
fig, ax = plt.subplots()
theoric_list_x = [P_max_theoric[2022][LSA_l[lsa]]for lsa in range(5)]
theoric_min = [P_theoric_min[2022][LSA_l[lsa]].loc[0] for lsa in range(5)]
values = [couple_sp[2022][LSA_l[lsa]][1]['Pollution rate'] for lsa in range(5)]
ax.scatter(range(1, 6), theoric_list_x, marker='^', label='Upper theoretical bound', color='grey', zorder=4,s=50)
ax.scatter(range(1, 6), theoric_min, marker='v', label='Lower theoretical bound', color='grey', zorder=4, s = 50)
ax.boxplot(values, labels=list_lsa)#,  showfliers=False)
plt.ylabel('CO2 emission rate (ton/Gwh)')
plt.ylim(-900, 2600)
plt.legend()


            
 # ==========================================================
# Figure 5 : Impact RES generation capacity multiplication
# =============================================================================

RES_value = {}
RES_value[2019] = np.concatenate([np.arange(1, 5.1, 0.5),np.arange(6,10.1,1)])
RES_value[2022] = np.arange(1,5.1,0.5)

sol_sp_res = {}
sol_wind = {}
sol_solar = {}
for year in [2019, 2022]:
    sol_sp_res[year] = {}
    sol_wind[year] = {}
    sol_solar[year] = {}
    sol_sp_res[year]['No Lsa'] = {}
    sol_wind[year]['No Lsa'] = {}
    sol_solar[year]['No Lsa'] = {}
    for res in RES_value[year]:
        print(res)
        sol_sp_res[year][res]= open_file(lsa = 3, year = year, tax = 'ETS', mult_w = res, mult_s = res)
        sol_wind[year][res]= open_file(lsa = 3, year = year, tax = 'ETS', mult_w = res)
        sol_solar[year][res]= open_file(lsa = 3, year = year, tax = 'ETS', mult_s = res)

                

#Compute result in a dataframe
result_sp_res={}
result_wind={}
result_solar={}
for year in [2019, 2022]:
    result_sp_res[year] = {}
    result_wind[year] = {}
    result_solar[year] = {}
    for res in RES_value[year]:
       result_sp_res[year][res]= agg_sol(sol_sp_res[year][res], sol_sp_res[year][res])
       result_wind[year][res]= agg_sol(sol_wind[year][res], sol_wind[year][res])
       result_solar[year][res]= agg_sol(sol_solar[year][res], sol_solar[year][res])


def final_result(df):
    df_final = {}
    for year in [2019, 2022]:
        result_daily = {}
        df_final[year] = pd.DataFrame()
        for res in RES_value[year]:
            result_daily = diff_agg_daily(df[year][res])
            df_final[year][res] = round(result_daily.sum(),5)
        df_final[year].loc['diff emis %'] = df_final[year].loc['diff emis']/df_final[year].loc['Emission']*100
    return df_final
        
result_res_all = final_result( result_sp_res)
result_solar_all = final_result(result_solar)
result_wind_all = final_result( result_wind)


#f = open(f"{path_file_result}result_res_all.pkl","wb")
#pickle.dump(result_res_all,f)
f = open(f"{path_file_result}result_solar_all.pkl","wb")
pickle.dump(result_solar_all,f)
        
        
with open(f'{path_file_result}result_solar_all.pkl', 'rb') as file:
        result_solar_all = pickle.load(file)
f = open(f"{path_file_result}result_wind_all.pkl","wb")
pickle.dump(result_wind_all,f)

with open(f'{path_file_result}result_wind_all.pkl', 'rb') as file:
        result_wind_all = pickle.load(file)

#with open(f'{path_file_result}result_res_all.pkl', 'rb') as file:
#        result_res_all = pickle.load(file)

        
fig, ax = plt.subplots()


# Plot all RES (plain lines)
ax.plot(result_res_all[2019].T.index, result_res_all[2019].T['diff emis'], color='dimgrey', label='All RES 2019')
ax.plot(result_res_all[2022].T.index, result_res_all[2022].T['diff emis'], color='silver', label='All RES 2022')

# Plot wind (dashed lines)
ax.plot(result_wind_all[2019].T.index, result_wind_all[2019].T['diff emis'], color='dimgrey', linestyle='dashed', label='Wind 2019')
ax.plot(result_wind_all[2022].T.index, result_wind_all[2022].T['diff emis'], color='silver', linestyle='dashed', label='Wind 2022')


# Plot solar (dotted lines)
ax.plot(result_solar_all[2019].T.index, result_solar_all[2019].T['diff emis'], color='dimgrey', linestyle='dotted', label='Solar 2019')
ax.plot(result_solar_all[2022].T.index, result_solar_all[2022].T['diff emis'], color='silver', linestyle='dotted' , label='Solar 2022')


ax.legend()
plt.xlabel("Multiplication RES")
plt.ylabel('Storage marginal CO2 emission (ton)')
#plt.show()


fig, ax = plt.subplots()

# Plot all RES (plain lines)
ax.plot(result_res_all[2019].T.index, result_res_all[2019].T['Sell LSA'], color='dimgrey', label='All RES 2019')
ax.plot(result_res_all[2022].T.index, result_res_all[2022].T['Sell LSA'], color='silver', label='All RES 2022')

# Plot wind (dashed lines)
ax.plot(result_wind_all[2019].T.index, result_wind_all[2019].T['Sell LSA'], color='dimgrey', linestyle='dashed', label='Wind 2019')
ax.plot(result_wind_all[2022].T.index, result_wind_all[2022].T['Sell LSA'], color='silver', linestyle='dashed', label = 'Wind 2022')

# Plot solar (dotted lines)
ax.plot(result_solar_all[2019].T.index, result_solar_all[2019].T['Sell LSA'], color='dimgrey', linestyle='dotted', label='Solar 2019')
ax.plot(result_solar_all[2022].T.index, result_solar_all[2022].T['Sell LSA'], color='silver', linestyle='dotted', label = 'Solar 2022')

# Custom Legend (Ensuring no duplicates)
handles = [
    plt.Line2D([0], [0], color='dimgrey', lw=2, label='2019'),
    plt.Line2D([0], [0], color='silver', lw=2, label='2022'),
    plt.Line2D([0], [0], color='black', lw=2, linestyle='solid', label='All RES'),
    plt.Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='Wind'),
    plt.Line2D([0], [0], color='black', lw=2, linestyle='dotted', label='Solar'),
]

#ax.legend(handles=handles)
# Labels & Title
#ax.legend()
plt.xlabel("Multiplicator RES")
plt.ylabel("Storage annual volume of sell (Gwh)")
# Show the plot
plt.show()

#what about the data you mentionned there, how did you get there

 # ==========================================================
# Figure 6 : Impact RES generation capacityon the volume of different transaction types
# =============================================================================
couple_wind={}
couple_solar={}
for year in [2019, 2022]:
    couple_wind[year] = {}
    couple_solar[year] = {}
    for res in RES_value[year]:
        print(res)
        couple_wind[year][res]= coupling(sol_wind[year][res],sol_wind[year]['No Lsa'][res], 3, 6)
        couple_solar[year][res]= coupling(sol_solar[year][res],sol_solar[year]['No Lsa'][res], 3, 6) 


        
f = open(f"{path_file_result}couple_wind.pkl","wb")
pickle.dump(couple_wind,f)

with open(f'{path_file_result}couple_wind.pkl', 'rb') as file:
        couple_wind = pickle.load(file)
        
f = open(f"{path_file_result}couple_solar.pkl","wb")
pickle.dump(couple_solar,f)

with open(f'{path_file_result}couple_solar.pkl', 'rb') as file:
        couple_solar = pickle.load(file)
                
def merge_series_to_df(dfs, df_names):
    # Step 1: Get the category counts for each DataFrame
    count_dict = {name: df.groupby('transaction_type')['Transaction Amount'].sum() for name, df in zip(df_names, dfs)}

    # Step 2: Find all unique indices across all the Series
    all_indices = set()
    for series in count_dict.values():
        all_indices.update(series.index)
    
    # Convert the set of indices to a sorted list (optional for ordering)
    all_indices = sorted(list(all_indices))

    # Step 3: Reindex each Series and fill missing indices with 0
    aligned_dict = {
        key: series.reindex(all_indices, fill_value=0) for key, series in count_dict.items()
    }

    # Step 4: Convert the aligned dictionary into a DataFrame
    final_df = pd.DataFrame.from_dict(aligned_dict)
    final_df = final_df[(final_df >= 10).any(axis=1)] #remove the type of transactions that are too small
    
    return final_df


df_names = {}
df_names[2019] = np.arange(1,10.1,1)
df_names[2022] = np.arange(1,5.1,1)
    
count_wind = {}
for year in [2019, 2022]:
    count_wind[year] = merge_series_to_df([couple_wind[year][res]['df'] for res in df_names[year]], df_names[year])
    
count_solar = {}
for year in [2019, 2022]:
    count_solar[year] = merge_series_to_df([couple_solar[year][res]['df'] for res in df_names[year]], df_names[year])
    
#this is to uniformize the terms used for plant type 
def change_index(df):
    df.index = pd.MultiIndex.from_tuples(
        [( 
            'CCGT' if x[0] == 'Gas_cc' else  # Replace 'Nuclear' with 'RES'
            ('Gas peaker' if x[0] == 'Fossil Gas' else 
             'Coal' if x[0] == 'Fossil Hard coal' else x[0]),  # Replace 'Fossil Gas' and 'Fossil Hard coal'
            
            'CCGT' if x[1] == 'Gas_cc' else
            ('Gas Peaker'  if x[1] == 'Fossil Gas' else 
             ('Coal' if x[1] == 'Fossil Hard coal' else x[1])  # Replace 'Fossil Hard coal' with 'Coal' in second part
        )) for x in df.index],
        names=['Energy Charging', 'Energy Discharging']
    )
    df = df.groupby(['Energy Charging', 'Energy Discharging']).sum()
    df = df.loc[~((df.index.get_level_values('Energy Charging') == 'RES') & (df.index.get_level_values('Energy Discharging') == 'RES'))]
    df = df[df.sum(axis=1) > 1]
    return df



for year in [2019,2022]:
    count_wind[year] = change_index(count_wind[year])
    count_solar[year] = change_index(count_solar[year])



def graph_transaction(df):
    colormap = plt.cm.gray
    num_columns = len(df.columns)
    x = np.arange(len(df))  # the label locations
    width = 0.8 / num_columns  # Width of bars to fit all in one group
    fig, ax = plt.subplots(figsize=(12, 6))
    start = 0.85  # Adjust this value to make the lightest color more visible
    end = 0.0  # Keep the darkest color as black
    # Generate a list of colors by sampling evenly from the colormap
    colors = [colormap(start - i * (start - end) / (10 - 1)) for i in range(10)]
    for i, col in enumerate(df.columns):
       ax.bar(x + i * width, df[col], width, label=str(col), color = colors[i])
    
    ax.set_xticks(x + width * (num_columns / 2 - 0.5))  # Center x-axis labels
    ax.set_xticklabels( [', '.join(map(str, idx)) for idx in df.index], rotation=25, fontsize = 20)
    ax.set_ylabel("Volume transaction (Gwh)", fontsize = 20)
    ax.set_ylim(0, 160)  # Set y-axis limit
    ax.legend()
    plt.show()    


graph_transaction(count_wind[2019])
graph_transaction(count_wind[2022])
graph_transaction(count_solar[2019])
graph_transaction(count_solar[2022])


# ==========================================================
#Figure 7 : Transaction-wise and yearly rates of added emission
# =============================================================================
tax_boxplot = {}
tax_boxplot[2019] = [0, 20,40,60, 100,600]
tax_boxplot[2022] = [0, 100, 400,800, 1000,3000]

tax_graph = {}
tax_graph[2019] =  np.concatenate([np.linspace(tax_boxplot[2019][i], tax_boxplot[2019][i+1], num=10, endpoint=False) for i in range(len(tax_boxplot[2019])-1)])
tax_graph[2019]  = np.append(tax_graph[2019] , tax_boxplot[2019][-1])
tax_graph[2019] = np.round(tax_graph[2019])
tax_graph[2022] =  np.concatenate([np.linspace(tax_boxplot[2022][i], tax_boxplot[2022][i+1], num=10, endpoint=False) for i in range(len(tax_boxplot[2022])-1)])
tax_graph[2022]  = np.append(tax_graph[2022] , tax_boxplot[2022][-1])


tax = {}
tax[2019] = {}
tax[2022] ={}
for lsa in [0,1,2,3,4]:
    if lsa == 0:
        tax[2022][LSA_l[lsa]] = sorted(list(tax_graph[2022]) + [385])
        tax[2019][LSA_l[lsa]] = tax_graph[2019]
    elif lsa == 1:
        tax [2022][LSA_l[lsa]]  =   sorted(list(tax_graph[2022]) + [2803])
        tax[2019][LSA_l[lsa]] = sorted(list(tax_graph[2019]) + [45, 55])
    elif lsa == 2:
        tax [2022][LSA_l[lsa]]  =  sorted(list(tax_graph[2022]) + [3200,3400,3600,3800,4000, 6000, 8000, 8502]) 
        tax[2019][LSA_l[lsa]] = sorted(list(tax_graph[2019]) + [45, 55])
    elif lsa == 3:
        tax [2022][LSA_l[lsa]]  =  sorted(list(tax_graph[2022]) +  [3200,3400,3600,3800,3850])
        tax[2019][LSA_l[lsa]] = sorted(list(tax_graph[2019]) + [45,55, 65, 70, 75, 85, 90, 101])
    else:
        tax [2022][LSA_l[lsa]]  =    sorted(list(tax_graph[2022]) +  [3200,3400,3600,3800,4000, 6000, 8000, 10000, 10050])
        tax[2019][LSA_l[lsa]] = sorted(list(tax_graph[2019]) + [45,55, 65, 70, 75, 85, 90, 110,130,170,190,220,240,260,280])
tax[2019]['No Lsa'] = tax[2019]['Perfect ESS'] + [101]
tax[2022]['No Lsa'] = tax[2022]['Perfect ESS'] + [385, 2803, 8502, 3850]


sol_sp_tax ={}
for year in [2019, 2022]:
    sol_sp_tax[year] = {}
    for lsa in [0,1,2,3,4]:
            sol_sp_tax[year][LSA_l[lsa]]={} 
            for t in tax[year][LSA_l[lsa]]:
                try:
                    sol_sp_tax[year][LSA_l[lsa]][t] = open_file(lsa = lsa, tax = t/1000, year = year)
                except Exception:
                    print(f"Error encountered for year={year}, lsa={lsa}, tax={t}")
                    continue  # Skip to the next iteration

for year in [2019, 2022]:
    sol_sp_tax[year]['No Lsa'] = {}
    for t in tax[year]['No Lsa']:
            try:
                sol_sp_tax[year]['No Lsa'][t] = open_file(lsa = 3, tax = t/1000, year = year, mult_lsa = 0)
            except Exception:
                print(f"Error encountered for year={year}, no_lsa, tax={t}")
                continue  # Skip to the next iteration


#Overal impact
result_sp_tax={} 
result_sp_tax_daily={}
result_sp_tax_period_all={}
for year in [ 2019,2022]:
    if year == 2022: 
       add_days = 365*2+366
    else:  
        add_days = 0
    result_sp_tax[year]={}
    result_sp_tax_daily[year]={}
    result_sp_tax_period_all[year]={}  
    for i in [0,1,2,3,4]:
            result_sp_tax[year][LSA_l[i]]={}
            result_sp_tax_daily[year][LSA_l[i]]={}
            result_sp_tax_period_all[year][LSA_l[i]]=pd.DataFrame()
            for ets in  tax[year][LSA_l[i]]:
                result_sp_tax[year][LSA_l[i]][ets] = agg_sol(sol_sp_tax[year][LSA_l[i]][ets], sol_sp_tax[year]['No Lsa'][ets])
                result_sp_tax_daily[year][LSA_l[i]][ets] = diff_agg_daily(result_sp_tax[year][LSA_l[i]][ets])
                result_sp_tax_period_all[year][LSA_l[i]] = pd.concat([result_sp_tax_period_all[year][LSA_l[i]],pd.DataFrame({ets: round(result_sp_tax_daily[year][LSA_l[i]][ets].sum(),5)})], axis = 1)

                    
f = open(f"{path_file_result}result_sp_tax_period_all.pkl","wb")
pickle.dump(result_sp_tax_period_all,f)

with open(f'{path_file_result}result_sp_tax_period_all.pkl', 'rb') as file:
        result_sp_tax_period_all = pickle.load(file)


        
Emission = {}
for year  in [2019,2022]: 
    Emission[year] = pd.DataFrame()
    for i in [0, 1,2,3,4]:
            Emission[year]=pd.merge(Emission[year], pd.DataFrame({LSA_l[i]: result_sp_tax_period_all[year][LSA_l[i]].loc['Emission'].values}, index = result_sp_tax_period_all[year][LSA_l[i]].columns), left_index=True, right_index=True, how='outer')
        
 

Amount = {}
for year  in [2019,2022]: 
    Amount[year] = {}
    for cap in [1]:
        Amount[year][cap]= pd.DataFrame()
        for i in [0, 1,2,3,4]:
            Amount[year][cap]=pd.merge(Amount[year][cap], pd.DataFrame({LSA_l[i]: result_sp_tax_period_all[year][LSA_l[i]].loc['Sell LSA'].values}, index = result_sp_tax_period_all[year][LSA_l[i]].columns), left_index=True, right_index=True, how='outer')


Diff_emis = {}
for year  in [2019,2022]: 
    Diff_emis[year] = {}
    for cap in [1]:
        Diff_emis[year][cap]= pd.DataFrame()
        for i in [0, 1,2,3,4]:
           Diff_emis[year][cap]=pd.merge(Diff_emis[year][cap], pd.DataFrame({LSA_l[i]: result_sp_tax_period_all[year][LSA_l[i]].loc['diff emis'].values}, index = result_sp_tax_period_all[year][LSA_l[i]].columns), left_index=True, right_index=True, how='outer')
       
Overal_rate = {}
for year  in [2019,2022]: 
    Overal_rate[year] = {}
    for cap in [1]:
        Overal_rate[year][cap]= pd.DataFrame()
        for i in [0, 1,2,3,4]:
            Overal_rate[year][cap] = Diff_emis[year][cap].div(Amount[year][cap], axis = 0)

Overal_rate[2019][1].drop_duplicates(inplace = True)
Overal_rate[2022][1].drop_duplicates(inplace = True)

## Impact per transaction ##
couple_sp_tax = {}
for year in [2019,2022]:
    print(year)
    couple_sp_tax[year] = {}
    for i in [0,1,2,3,4]:
        print(i)
        couple_sp_tax[year][LSA_l[i]] = {}
        for t in tax[year][LSA_l[i]]: 
                couple_sp_tax[year][LSA_l[i]][t]  = coupling(sol_sp_tax[year][LSA_l[i]][t], sol_sp_tax[year]['No Lsa'][t], i, 6)



f = open(f"{path_file_result}couple_sp_tax_all_plants.pkl","wb")
pickle.dump(couple_sp_tax,f)

with open(f'{path_file_result}couple_sp_tax.pkl', 'rb') as file:
        couple_sp_tax = pickle.load(file)
        


P_theoric_tax={} 
P_theoric_min = {}
for year in [2019, 2022]:
    P_theoric_tax[year]={}
    P_theoric_min[year]={}
    for t in  tax_boxplot[year]:
        P_theoric_tax[year][t]=pd.DataFrame(columns = range(5))
        P_theoric_min[year][t]=pd.DataFrame(columns = range(5))
        for obs in range(0,365):
            par_temp = sol_sp_tax[year]['Hydrogen'][t][obs].par_temp
            P_theo = [P_theoric_f(par.eta_b[0,lsa]**2, par.c_bb[0,lsa], par_temp) for lsa in [0,1,2,3,4]]
            P_theoric_tax[year][t].loc[obs] = [P_theo[lsa][1] for lsa in range(5)]
            P_theoric_min[year][t].loc[obs] = [P_theo[lsa][0] for lsa in range(5)]


P_max_theoric_tax ={}
P_max_theoric_tax[2019] = pd.DataFrame(data = [P_theoric_tax[2019][tax].max() for tax in tax_boxplot[2019]])
P_max_theoric_tax[2022] = pd.DataFrame(data = [P_theoric_tax[2022][tax].max() for tax in tax_boxplot[2022]])
P_max_theoric_tax[2019].columns = LSA_l
P_max_theoric_tax[2022].columns = LSA_l
P_min_theoric_tax ={}
P_min_theoric_tax[2019] = pd.DataFrame(data = [P_theoric_min[2019][tax].min() for tax in tax_boxplot[2019]])
P_min_theoric_tax[2022] = pd.DataFrame(data = [P_theoric_min[2022][tax].min() for tax in tax_boxplot[2022]])
P_min_theoric_tax[2019].columns = LSA_l
P_min_theoric_tax[2022].columns = LSA_l



position_ets = {}
position_ets[2019] = 13
position_ets[2022] = 8
Overal_rate_l = {}
values = {}
for year in [2019, 2022]:
    Overal_rate_l[year]={}
    values[year] ={}
    for lsa in range(5):
        Overal_rate_l[year][LSA_l[lsa]] = [Overal_rate[year][1].loc[i,LSA_l[lsa]] for i in tax_graph[year]]
        values[year][LSA_l[lsa]] = [couple_sp_tax[year][LSA_l[lsa]][i]['Pollution rate'] for i in tax_boxplot[year]]


#Remove potential nas transaction
for year in [2019,2022]:
    for lsa in range(5):
        for i in range(6):
            values[year][LSA_l[lsa]][i] = pd.Series(values[year][LSA_l[lsa]][i]).dropna().tolist()


for lsa in range(5):
    for year in [2019, 2022]:  
        fig, ax = plt.subplots()
        ax.scatter([0,10,20,30,40,50], P_max_theoric_tax[year][LSA_l[lsa]], marker='^', label='Upper theoretical bound', color='grey', zorder=4,s=50)
        ax.scatter([0,10,20,30,40,50], P_min_theoric_tax[year][LSA_l[lsa]], marker='v', label='Lower theoretical bound', color='grey', zorder=4, s = 50)
        ax.plot(range(51), Overal_rate_l[year][LSA_l[lsa]], label = 'Yearly marginal emission rate', color = 'black')
        ax.boxplot(values[year][LSA_l[lsa]], labels=tax_boxplot[year],  showfliers=True, positions = [0,10,20,30,40,50], widths = 3)
        ax.axvline(x=position_ets[year], color='black', linestyle='dotted', label = 'ETS Mean')
        plt.ylabel('Marginal Co2 emission rate (ton/Gwh)')
        plt.xlabel('Carbon levy level (€/ton)')
        #plt.ylim(-700, 2650)
        #plt.xlim(-10, 320)
        if year == 2019:
            plt.legend()


# ==========================================================
#Figure 8 : Minimal carbon levy for different maximal allowed rates of added emission
# =============================================================================

#compute the theoretical tax
rows = []
Co2_Tax = {}
Ax_tax_indices = {}
for lsa in [0,1,2,3,4]:
    #Co2_Tax[LSA_l[lsa]] = pd.DataFrame()
    #Ax_tax_indices[LSA_l[lsa]] = pd.DataFrame()
    Ax_tax_indices[LSA_l[lsa]] = Ax_tax_indices[LSA_l[lsa]].astype(object)  
#find couple to look at that would need an higher tax

for year in [2019]:
    add_days = 365 * 2 + 366 if year == 2022 else 0
    for obs in range(365):
        par_temp = sol_sp_tax[year][1]['Hydrogen'][0][obs].par_temp
        par_temp_6 = par_temp[:, 6]
        par_temp_last = par_temp[:, -1]
        for lsa in [0, 1, 2, 3, 4]:
            alpha = par.eta_b[0, lsa] ** 2
            for P in range(600,800):
                indices = np.argwhere((par_temp_6[:, None] / alpha - par_temp_6) > P)
                if indices.size > 0:
                    i_vals, j_vals = indices[:, 0], indices[:, 1]
                    tax_values = (alpha * par_temp_last[j_vals] - par_temp_last[i_vals] - par.c_bb[0, lsa] * (1 + alpha)) / (par_temp_6[i_vals] - alpha * par_temp_6[j_vals])
                    max_tax = max(0, np.max(tax_values))
                    max_index = indices[np.argmax(tax_values)]
                    Co2_Tax[LSA_l[lsa]].loc[P, obs + add_days] = max_tax
                    #Ax_tax_indices[LSA_l[lsa]].at[P, obs + add_days] = max_index.tolist()
                else:
                    Co2_Tax[LSA_l[lsa]].loc[P, obs + add_days] =  0

              
                

f = open(f"{path_file_result}Ax_tax_P_LSA_AP.pkl","wb")
pickle.dump(Co2_Tax,f) 

with open(f'{path_file_result}Ax_tax_P_LSA_AP.pkl', 'rb') as file:
    Co2_Tax =  pickle.load(file)
    


#Remove potential nas transaction
for year in [2019,2022]:
    add_days = 365 * 2 + 366 if year == 2022 else 0
    for lsa in range(5):
        for i in range(6):
            Co2_Tax[LSA_l[lsa]].loc[:, add_days : 365 + add_days] = Co2_Tax[LSA_l[lsa]].loc[:, add_days : 365 + add_days].dropna()


#compute average
Ax_mean = {}
for year in [2019, 2022]:
    if year == 2022: 
       add_days = 365*2+366
    else:  
        add_days = 0
    Ax_mean[year] = pd.DataFrame(columns = range(0,900))
    for key,df in Co2_Tax.items():
        Ax_mean[year].loc[key] = Co2_Tax[key].loc[:,0+add_days:364+add_days].mean(axis = 1)*1000
 
    

ets={}
ets[2019] = Prices.loc[0:365, 'ETS'].mean()
ets[2022] = Prices.loc[1096:1461,'ETS'].mean()    
        
#graph
for year in [2019, 2022]:
    plt.figure()
    for i, column in enumerate(Ax_mean[year].T.columns):
        Ax_mean[year].T[column].plot(color=colors_dict[column], linestyle=linestyles_dict[column])
    plt.axvline(x=ets[year], color='black', linestyle='solid', label = 'ETS Mean')
    #plt.plot(range(9000), [ets[year]]*9000, linestyle='solid', label='ETS mean' , color='black',   marker='o'),  markeredgewidth=2)
    plt.xlabel('Maximum allowed marginal emission rate (ton/Gwh)')
    plt.ylabel('Average carbon levy (€/ton)')
    plt.legend()
    if year == 2019:
        plt.ylim(0,400)
        plt.xlim(0,800)
    else:
        plt.ylim(0,4000)
        plt.xlim(0,800)
    plt.show() 



# ==========================================================
#Table 4 and 5 : Maximum ration phi and 90th quantile of ratio phi, dependence Co2_tax from figure 8 and couple_sp_tax from figure 7
# =============================================================================

for t in couple_sp_tax[2022]['Perfect ESS'].keys():
    couple_sp_tax[2022]['Perfect ESS'][t]['df'] = couple_sp_tax[2022]['Perfect ESS'][t]['df'] [round(couple_sp_tax[2022]['Perfect ESS'][t]['df'] ['Transaction Amount'], 3) != 0]


Max_Pollution_rate = {}
for year in [2019, 2022]:
    Max_Pollution_rate[year]  = {}
    for i in [0,1,2,3,4]:
        Max_Pollution_rate[year][LSA_l[i]] = pd.DataFrame(index = range(365))
        for t in tax[year][LSA_l[i]]:
            Max_Pollution_rate[year][LSA_l[i]] = pd.concat([Max_Pollution_rate[year][LSA_l[i]], pd.DataFrame({t: couple_sp_tax[year][LSA_l[i]][t].groupby('Day')['Pollution rate'].max()})], axis = 1)    

Max_Pollution_rate[2022]['Perfect ESS'][10050] = [round(x, 1) if round(x,1) == 0 else x for x in Max_Pollution_rate[2022]['Perfect ESS'][10050]]

for year in [2019, 2022]:
    for i in [0,1,2,3,4]:
        Max_Pollution_rate[year][LSA_l[i]].fillna(0, inplace=True)
        Max_Pollution_rate[year][LSA_l[i]] = Max_Pollution_rate[year][LSA_l[i]].applymap(lambda x: 0 if round(x, 2) == 0 else x)




def find_index(series ,P, d):
    # Find the last occurrence of a positive value
    last_positive_index = -1
    for i in range(len(series)):
        if series.iloc[i] > P:
            last_positive_index = i
    # Check if all values after last_positive_index are 0
    if last_positive_index != -1 and all(value <= P for value in series.iloc[last_positive_index+1:]):
            # Find the first 0 after last_positive_index
            first_zero_index = Max_Pollution_rate[year][LSA_l[lsa]].loc[d].index[last_positive_index+1]
            return first_zero_index
    else:
        return 0


for lsa in range(5):
    Co2_Tax[LSA_l[lsa]] = Co2_Tax[LSA_l[lsa]].fillna(0)

Th_su = {}
for year in [2019, 2022]:
    if year == 2022: 
       add_days = 365*2+366
    else:  
        add_days = 0
    Th_su[year] = {}
    for lsa in [0,1,2,3,4]:
        Th_su[year][LSA_l[lsa]] = {}
        for P in [0,100,500]:
            Th_su[year][LSA_l[lsa]][P] = pd.DataFrame(index = range(0+add_days, 365+add_days))
            Th_su[year][LSA_l[lsa]][P]['Avg_price'] = Co2_Tax[LSA_l[lsa]].T.loc[0+add_days:365+add_days][P]
            Th_su[year][LSA_l[lsa]][P]['tax achieving this target']  = [find_index(Max_Pollution_rate[year][LSA_l[lsa]].loc[d], P, d)/1000 for d in range(0,365)]
            Th_su[year][LSA_l[lsa]][P]['ratio'] = Th_su[year][LSA_l[lsa]][P]['tax achieving this target']/Th_su[year][LSA_l[lsa]][P]['Avg_price'] 
            Th_su[year][LSA_l[lsa]][P]['ratio'] = Th_su[year][LSA_l[lsa]][P]['ratio'].clip(upper=1)
            Th_su[year][LSA_l[lsa]][P] = Th_su[year][LSA_l[lsa]][P].dropna()
  
f = open(f"{path_file_result}Th_su.pkl","wb")
pickle.dump(Th_su,f) 

with open(f'{path_file_result}Th_su.pkl', 'rb') as file:
    Th_su=  pickle.load(file)            

#the two different tables
Th_su_qt = {}
for q in [90,100]:
    Th_su_qt[q]= {}
    for P in [0,100,500]:
        Th_su_qt[q][P]= pd.DataFrame(index = [2019,2022], columns = LSA_l)
        for lsa in [0,1,2,3]:
            if lsa != 0:
                Th_su_qt[q][P][LSA_l[lsa]] = [np.percentile(Th_su[year][LSA_l[lsa]][P]['ratio'], q) for year in [2019, 2022]]
            else:
                Th_su_qt[q][P][LSA_l[lsa]][2022] = [np.percentile(Th_su[year][LSA_l[lsa]][P]['ratio'], q) for year in [2022]]

       
            
            
 # ==========================================================
# Figure 9 : LSA changes of profit compared to the case without any carbon levy, dependence on 'result_sp_tax_period' from figure 7
# =============================================================================

change = {}
change[2019] = pd.DataFrame()
for i in [0,1,2,3,4]:
    change[2019][LSA_l[i]] = (result_sp_tax_period_all[2019][LSA_l[i]].loc['Profit']/result_sp_tax_period_all[2019][LSA_l[i]].loc['Profit', 0] -1)*100
#change[2019].drop('ETS',  inplace = True)
change[2019] = change[2019].sort_index()
change[2019].index = np.float64(change[2019].index)

change[2022] = pd.DataFrame()
for i in [0,1,2,3,4]:
    change[2022][LSA_l[i]] = (result_sp_tax_period_all[2022][LSA_l[i]].loc['Profit']/result_sp_tax_period_all[2022][LSA_l[i]].loc['Profit', 0] -1)*100
#change[2022].drop('ETS',  inplace = True)
change[2022] = change[2022].sort_index()
#change[2022].index = np.float64(change[2022].index)*1000



change[2022] = change[2022].drop(385)

for i, column in enumerate(change[2022].columns):
    change[2022][column].plot(color=colors[i], linestyle=linestyles[i], legend = True)
plt.xlabel ('Carbon levy level (€/ton)')
plt.xlim(0,800)
plt.ylim(-20, 80)
plt.ylabel( 'Profit change (%)')
plt.axvline(x=ets[2022], color="black", linestyle='-', label='ETS mean')
plt.legend()
plt.show()



for i, column in enumerate(change[2019].columns):
    change[2019][column].plot(color=colors[i], linestyle=linestyles[i], legend = True)
plt.xlabel ('CO2 Tax (€/ton)')
plt.xlim(0,70)
plt.ylim(0, 400)
plt.ylabel( 'Profit change (%)')
plt.axvline(x=ets[2019], color='black', linestyle='-', label='ETS mean')
plt.legend()
plt.show()

# ==========================================================
#Figure 10: annual CO2 emission impact with ramping
# =============================================================================
path_file_optimisation = path_file_optimisation = os.path.join(userpath,"Data\\result_optimisation\\constraint\\Ramp\\")


sol_ramping= {}
sol_ss_c_ramp = {}
for year in [2019,2022]:
   sol_ramping[year] = {}
   sol_ss_c_ramp[year] = {}
   for lsa in [0,1,2,3,4]:
        sol_ss_c_ramp[year][LSA_l[lsa]] = open_file(lsa, ramp = False, year = year)
   for cap in range(1,11):
        sol_ramping[year][cap] = {}
        for lsa in [0,1,2,3,4]:
            sol_ramping[year][cap][LSA_l[lsa]] = open_file(lsa, ramp = True, year = year, mult_lsa = cap)
   sol_ramping[year]['No Lsa'] = open_file(3, mult_lsa = 0, ramp = True, year = year)
   sol_ss_c_ramp[year]['No lsa'] = open_file(3, mult_lsa = 0, year = year)


result_ramping = {}
result_ss_c_ramp = {}
for year in [2019, 2022]:
   result_ramping[year] = {}
   result_ss_c_ramp[year] = {}
   for lsa in range(5):
       result_ss_c_ramp[year][LSA_l[lsa]] = agg_sol(sol_ss_c_ramp[year][LSA_l[lsa]], sol_ss_c_ramp[year][LSA_l[lsa]])
   for cap in range(1,11):
        result_ramping[year][cap] = {}
        for lsa in range(5):
            try:
                result_ramping[year][cap][LSA_l[lsa]] = agg_sol(sol_ramping[year][cap][LSA_l[lsa]], sol_ramping[year]['No Lsa'])
            except Exception:
                     print(f"Error encountered for year={year}, lsa={lsa}, cap = {cap}")

result_all_ramping = {} 
result_all_ss_c_ramp = {}

for year in [2022]:
    result_all_ramping[year]={}
    result_all_ss_c_ramp[year]=pd.DataFrame()
    for lsa in range(5):
        result_all_ss_c_ramp[year][LSA_l[lsa]] = round(result_ss_c_ramp[year][LSA_l[lsa]].sum(),5)
    result_all_ss_c_ramp[year].loc['contribution emission'] = result_all_ss_c_ramp[year].loc['diff emis']/result_all_ss_c_ramp[year].loc['Emission'] *100
    for cap in range(1,11):
        result_all_ramping[year][cap] = pd.DataFrame()
        for lsa in range(5):
            result_all_ramping[year][cap][LSA_l[lsa]] = round(result_ramping[year][cap][LSA_l[lsa]].sum(), 5)
        result_all_ramping[year][cap].loc['contribution emission'] = result_all_ramping[year][cap].loc['diff emis']/result_all_ramping[year][cap].loc['Emission'] *100    

ramp_capacity  = {}
for year in [2019, 2022]:
    ramp_capacity[year] = pd.DataFrame()
    for cap in range(1,11): 
        for lsa in [0,1,2,3,4]:
            ramp_capacity[year].loc[cap,LSA_l[lsa]]=result_all_ramping[year][cap][LSA_l[lsa]]['contribution emission']
    ramp_capacity[year]['index']=([10,20,30,40,50,60,70,80,90,100])
    ramp_capacity[year].set_index('index', inplace = True)

f = open(f'{path_file_result}ramp_capacity.pkl',"wb")
pickle.dump(ramp_capacity,f)

#and load it
with open(f'{path_file_result}ramp_capacity.pkl', 'rb') as file:
        ramp_capacity = pickle.load(file)



for year in [2019,2022]:
    for i, lsa in enumerate(ramp_capacity[year].columns):
        ramp_capacity[year][lsa].plot(color=colors_dict[lsa], linestyle=linestyles_dict[lsa])
    plt.legend()
    plt.ylabel('Storage marginal CO2 emission (%)')
    plt.xlabel('Share of daily average demand (%)')
    #plt.ylim(-0.85, 0.5)    
    plt.plot()
    plt.show()


   
     
   
# ==========================================================
#Figure 11: change generation with ramping constraints
# =============================================================================

#Numbers impact without LSA can be also found in this table:
Impact_ramp = {}
for year in [2019, 2022]:
    Impact_ramp[year] = pd.DataFrame()
    for lsa in range(5):
        Impact_ramp[year][LSA_l[lsa]] = change_ss_c(sol_ramping[year][1][LSA_l[lsa]], sol_ss_c_ramp[year]['No lsa'])
    Impact_ramp[year]['No Lsa'] = change_ss_c(sol_ramping[year]['No Lsa'], sol_ss_c_ramp[year]['No lsa'])
    Impact_ramp[year].loc['change_emis %'] =     (Impact_ramp[year].loc['emis_total']/  Impact_ramp[year].loc['emis comparison'] - 1)*100
    #Impact_ramp[year].loc['change_obj %'] =     (Impact_ramp[year].loc['obj after']/  Impact_ramp[year].loc['obj before'] - 1)*100
 

Impact_ramp[2019].iloc[0:5].plot(kind = 'bar', color = colors_dict, ylabel = 'Generation change (Gwh)', legend = False)



Impact_ramp[2022].iloc[0:5].plot(kind = 'bar', color = colors_dict, ylabel = 'Generation change (Gwh)')


# ==========================================================
#Table 6: above bound estimation
# =============================================================================
#for this, the optimisation function from the optimisation file needs to be loaded

couple_ramping = {}
for year in [2022]:
    couple_ramping[year] = {}
    for lsa in [2,3,4]:
        couple_ramping[year][LSA_l[lsa]] = coupling_compute(sol_ramping[year][1][LSA_l[lsa]], lsa)


couple_ramping_final = {}
for year in [2019,2022]:
    couple_ramping_final[year] = {}
    for lsa in [2,3,4]:
        print(lsa)
        couple_ramping_final[year][LSA_l[lsa] ]= pd.DataFrame(columns=[ 'Day', 'Transaction Amount', 'Transaction Impact', 'Pollution rate','Period Buy', 'Period Sell'])
        for day in range(365):
            D_b = np.zeros((par.T,1))
            D_s = np.zeros((par.T,1))
            sol_lsa_upd = sol_ramping[year]['No Lsa'][day]
            emis_total = [sum(sol_lsa_upd.var.emis_hourly)[0,0]]
            for tr in couple_ramping[year][LSA_l[lsa]][day]:
                # tr[1] is time of bying, tr[2] is time of selling, tr[0] is how much sold
                  D_b[tr[1]] = D_b[tr[1]]+tr[0]/par.eta_s[0,lsa]**2
                  D_s[tr[2]] = D_s[tr[2]] + tr[0]
                  sol_lsa_upd = optimisation_sp(lsa=lsa, year=year, mult_lsa=1, num_days=1,
                                          start_day=day, mult_d = 1, mult_s = 1, mult_w = 1, 
                                          D_b = D_b, D_s = D_s, fix_lsa = True, save = False, ramp = True)
                  emis_total.append(sum(sol_lsa_upd.var.emis_hourly))
                  pollution_rate = (emis_total[-1]-emis_total[-2])[0,0]
                  new_row = {
                        'Day': day,
                        'Transaction Amount': tr[0],
                        'Transaction Impact': pollution_rate,
                        'Pollution rate': pollution_rate/tr[0],
                        'Period Buy': tr[1],
                        'Period Sell': tr[2]
                    }
                  couple_ramping_final[year][LSA_l[lsa] ].loc[len(couple_ramping_final[year][LSA_l[lsa]])] = new_row
    
f = open(f"{path_file_result}couple_ramping.pkl","wb")
pickle.dump(couple_ramping_final,f)
        




with open(f'{path_file_result}couple_ramping.pkl', 'rb') as file:
    couple_ramping_new_2=  pickle.load(file)

P_theoric = {}
for year in [2019, 2022]:
    P_theoric[year] = pd.DataFrame(columns = [LSA_l[0], LSA_l[1], LSA_l[2], LSA_l[3], LSA_l[4]])
    for obs in range(0,365):
        par_temp = sol_ss_c_ramp[year][LSA_l[0]][obs].par_temp
        P_theoric[year].loc[obs] = [P_theoric_f(par.eta_b[0,lsa]**2, par.c_bb[0,lsa], par_temp) for lsa in [0,1,2,3,4]]
     
#compute the maximum theoric pollution rate in the year per LSA 
P_max_theoric = pd.DataFrame(data = {2019: P_theoric[2019].max(), 2022: P_theoric[2022].max()})


P_theoric_min = {}
for year in [2019, 2022]:
    P_theoric_min[year] = pd.DataFrame(columns = [LSA_l[0], LSA_l[1], LSA_l[2], LSA_l[3], LSA_l[4]])
    for obs in range(0,365):
        par_temp =  sol_ss_c_ramp[year][LSA_l[0]][obs].par_temp
        P_theo = [P_theoric_f(par.eta_b[0,lsa]**2, par.c_bb[0,lsa], par_temp) for lsa in [0,1,2,3,4]]
        P_theoric[year].loc[obs] = [P_theo[lsa][1] for lsa in range(5)]
        P_theoric_min[year].loc[obs] = [P_theo[lsa][0] for lsa in range(5)]

#graph
#2019
theoric_list = [P_max_theoric[2019][LSA_l[lsa]]for lsa in range(5)]
theoric_min = [P_theoric_min[2019][LSA_l[lsa]].loc[0] for lsa in range(5)]
list_lsa = [LSA_l[0], LSA_l[1], LSA_l[2], LSA_l[3], LSA_l[4]]
fig, ax = plt.subplots()
values = [couple_ramping_final[2019][LSA_l[lsa]]['Pollution rate'] for lsa in range(5)]
ax.boxplot(values, labels=list_lsa)
ax.scatter(range(1, 6), theoric_list, marker='X', label='Upper theoretical bound', color='red', zorder=4)
ax.scatter(range(1, 6), theoric_min, marker='X', label='Lower theoretical bound', color='blue', zorder=4)
plt.ylabel('CO2 emission rate (ton/Gwh)')
#plt.ylim(-1500, 3000)
plt.legend()

#2022
fig, ax = plt.subplots()
theoric_list_x = [P_max_theoric[2022][LSA_l[lsa]]for lsa in range(5)]
theoric_min = [P_theoric_min[2022][LSA_l[lsa]].loc[0] for lsa in range(5)]
values = [couple_ramping_final[2022][LSA_l[lsa]]['Pollution rate'] for lsa in range(5)]
ax.scatter(range(1, 6), theoric_list_x, marker='X', label='Upper theoretical bound', color='red', zorder=4)
ax.scatter(range(1, 6), theoric_min, marker='X', label='Lower theoretical bound', color='blue', zorder=4)
ax.boxplot(values, labels=list_lsa)#,  showfliers=False)
plt.ylabel('CO2 emission rate (ton/Gwh)')
plt.ylim(-3000, 10000)
plt.legend()

above_daily_threshold =  pd.DataFrame(index = [2019, 2022], columns = LSA_l)
for year in [2019, 2022]:
    for lsa in [0,3]:
        couple_ramping_final[year][LSA_l[lsa]]['target'] = couple_ramping_final[year][LSA_l[lsa]]['Day'].map(P_theoric[year][LSA_l[lsa]])
        above_daily_threshold.loc[year, LSA_l[lsa]] = sum(couple_ramping_final[year][LSA_l[lsa]].loc[couple_ramping_final[year][LSA_l[lsa]]['Pollution rate'] > couple_ramping_final[year][LSA_l[lsa]]['target'] , 'Transaction Amount'])/sum(couple_ramping_final[year][LSA_l[lsa]]['Transaction Amount']) *100

bins = [0, 1, 1000]  # Custom bin edges
labels = ["]0,1]", "[1,["]  # Bin names


for year in [2019, 2022]:
    for lsa in [0,1,2,3,4]:
        couple_ramping_final[year][LSA_l[lsa]]['target'] = couple_ramping_final[year][LSA_l[lsa]]['Day'].map(P_theoric[year][LSA_l[lsa]])
        couple_ramping_final[year][LSA_l[lsa]]['difference target'] = (couple_ramping_final[year][LSA_l[lsa]]['Pollution rate'] - couple_ramping_final[year][LSA_l[lsa]]['target']).clip(lower = 0)
        couple_ramping_final[year][LSA_l[lsa]]['% above target'] = couple_ramping_final[year][LSA_l[lsa]]['difference target']/couple_ramping_final[year][LSA_l[lsa]]['target']
        couple_ramping_final[year][LSA_l[lsa]].loc[couple_ramping_final[year][LSA_l[lsa]]['difference target'] == 0, '% above target'] = 0
        couple_ramping_final[year][LSA_l[lsa]]['Bin'] = pd.cut( couple_ramping_final[year][LSA_l[lsa]]['% above target'] , bins = bins, labels = labels, right = False, include_lowest = True)
        couple_ramping_final[year][LSA_l[lsa]]['Bin'] =  couple_ramping_final[year][LSA_l[lsa]]['Bin'].astype(str)
        couple_ramping_final[year][LSA_l[lsa]].loc[couple_ramping_final[year][LSA_l[lsa]]['% above target'] == 0, 'Bin'] = "[0,0]"
        couple_ramping_final[year][LSA_l[lsa]].loc[(couple_ramping_final[year][LSA_l[lsa]]['target'] == 0) & (couple_ramping_final[year][LSA_l[lsa]]['difference target'] != 0), 'Bin'] = '[1,['
        couple_ramping_final[year][LSA_l[lsa]]['target low'] = couple_ramping_final[year][LSA_l[lsa]]['Day'].map(P_theoric_min[year][LSA_l[lsa]])
        couple_ramping_final[year][LSA_l[lsa]]['difference target low'] = (couple_ramping_final[year][LSA_l[lsa]]['target low'] - couple_ramping_final[year][LSA_l[lsa]]['Pollution rate']).apply(lambda x: max(x, 0))
        couple_ramping_final[year][LSA_l[lsa]]['% below target'] = abs(couple_ramping_final[year][LSA_l[lsa]]['difference target low']/couple_ramping_final[year][LSA_l[lsa]]['target low'])
        couple_ramping_final[year][LSA_l[lsa]].loc[couple_ramping_final[year][LSA_l[lsa]]['difference target low'] == 0, '% below target'] = 0
        couple_ramping_final[year][LSA_l[lsa]]['Bin low'] = pd.cut( couple_ramping_final[year][LSA_l[lsa]]['% below target'] , bins = bins, labels = labels, right = False, include_lowest = True)
        couple_ramping_final[year][LSA_l[lsa]]['Bin low'] =  couple_ramping_final[year][LSA_l[lsa]]['Bin low'].astype(str)
        couple_ramping_final[year][LSA_l[lsa]].loc[couple_ramping_final[year][LSA_l[lsa]]['% below target'] == 0, 'Bin low'] = "[0,0]"
        couple_ramping_final[year][LSA_l[lsa]].loc[(couple_ramping_final[year][LSA_l[lsa]]['target low'] == 0) & (couple_ramping_final[year][LSA_l[lsa]]['difference target low'] != 0), 'Bin low'] = '[1,['
      

# Sum 'Amount' within each bin
x = couple_ramping_final[year][LSA_l[lsa]].groupby('Bin')['Transaction Amount'].sum().reset_index()['Bin']

above_threshold = {}
for year in [2019, 2022]:
    above_threshold[year] = pd.DataFrame(index = x)
    for lsa in [0,1,2,3,4]:
        above_threshold[year][LSA_l[lsa]] =  couple_ramping_final[year][LSA_l[lsa]].groupby('Bin')['Transaction Amount'].sum()
    above_threshold[year] = above_threshold[year].div(above_threshold[year].sum())*100
    above_threshold[year] = above_threshold[year].fillna(0)

below_threshold = {}
for year in [2019, 2022]:
    below_threshold[year] = pd.DataFrame(index = x)
    for lsa in [0,1,2,3,4]:
        below_threshold[year][LSA_l[lsa]] =  couple_ramping_final[year][LSA_l[lsa]].groupby('Bin low')['Transaction Amount'].sum()
    below_threshold[year] = below_threshold[year].div(below_threshold[year].sum())*100
    below_threshold[year] = below_threshold[year].fillna(0)


# ==========================================================
#All the same with min load but no graph in the manuscript, same for start - up cost. 
# =============================================================================
path_file_optimisation = os.path.join(userpath,"Data\\result_optimisation\\constraint\\")

sol_sp_min_load = {}
sol_sp_ss_c = {}
for year in [2019, 2022]:
    sol_sp_min_load[year] = {}
    sol_sp_ss_c[year] = {}
    for cap in range(1,2):
        sol_sp_min_load[year][cap] = {}
        for lsa in range(5):
            sol_sp_min_load[year][cap][LSA_l[lsa]] = open_file(lsa, min_load = True, year = year, mult_lsa = cap)
            sol_sp_ss_c[year][LSA_l[lsa]] = open_file(lsa, year = year)
    sol_sp_min_load[year]['No Lsa'] = open_file(3, mult_lsa = 0, min_load = True, year = year)
    sol_sp_ss_c[year]['No Lsa'] = open_file(3, mult_lsa = 0, year = year)

year = 2022
x =  sol_sp_min_load[year]['Battery'] = open_file(3, mult_lsa = 1, min_load = True, year = year)

Impact_min_load = {}
for year in [2019, 2022]:
    Impact_min_load[year] = pd.DataFrame()
    for lsa in range(5):
        Impact_min_load[year][LSA_l[lsa]] = change_ss_c(sol_sp_min_load[year][LSA_l[lsa]], sol_sp_ss_c[year]['No Lsa'])
    Impact_min_load[year]['No Lsa'] = change_ss_c(sol_sp_min_load[year]['No Lsa'], sol_sp_ss_c[year]['No Lsa'])
    Impact_min_load[year].loc['change_emis %'] =     (Impact_min_load[year].loc['emis_total']/  Impact_min_load[year].loc['emis comparison'] - 1)*100


Impact_min_load[2019].iloc[0:5].plot(kind = 'bar', color = colors_dict, ylabel = 'Generation change (Gwh)', legend = False)

Impact_min_load[2022].iloc[0:5].plot(kind = 'bar', color = colors_dict, ylabel = 'Generation change (Gwh)')

result_min_load = {}
result_ss_c = {}
for year in [2019, 2022]:
    result_min_load[year] = {}
    result_ss_c[year] = {}
    for lsa in range(5):
        result_min_load[year][LSA_l[lsa]] = agg_sol(sol_sp_min_load[year][LSA_l[lsa]], sol_sp_min_load[year]['No Lsa'])
        result_ss_c[year][LSA_l[lsa]] = agg_sol(sol_sp_ss_c[year][LSA_l[lsa]], sol_sp_ss_c[year][LSA_l[lsa]])

#for couple min load, the issue is that for some time when there is a transaction, there is both up and down of some generators,
#so for the day when it happens I use the reoptimisation method to get the emission change due to a transaction. 
#for the other day, I use the classical method, 
#thus, there are three steps, first run coupling_new that gives couple and days to rerun
#then coupling compute that gives the transaction 
#then reoptimisation usinnf days to run and coupling from coupling compute
couple_min_load = {}
couple_ss_c = {}
for year in [2019, 2022]:
    couple_min_load[year] = {}
    couple_ss_c[year] = {}
    for lsa in range(5):
        couple_min_load[year][LSA_l[lsa]] = coupling(sol_sp_min_load[year][1][LSA_l[lsa]],sol_sp_min_load[year]['No Lsa'], lsa, col_em = 6)
      
couple_compute_min_load = {}
for year in [2019, 2022]:
      couple_compute_min_load[year] = {}
      for lsa in [2,3,4]:
        couple_compute_min_load[year][LSA_l[lsa]] = coupling_compute(sol_sp_min_load[year][LSA_l[lsa]], lsa)
       


couple_new = {}
couple_final_min_load = {}
for year in [2019,2022]:
    couple_new[year]={}
    couple_final_min_load[year] = {}
    for lsa in [2,3,4]:
        print(lsa)
        couple_new[year][LSA_l[lsa] ]= pd.DataFrame(columns=[ 'Day', 'Transaction Amount', 'Transaction Impact', 'Pollution rate','Period Buy', 'Period Sell'])
        for day in couple_min_load[year][LSA_l[lsa]]['to_run']:
            D_b = np.zeros((par.T,1))
            D_s = np.zeros((par.T,1))
            sol_lsa_upd = sol_sp_min_load[year]['No Lsa'][day]
            emis_total = [sum(sol_lsa_upd.var.emis_hourly)[0,0]]
            for tr in couple_compute_min_load[year][LSA_l[lsa]][day]:
                  D_b[tr[1]] = D_b[tr[1]]+tr[0]/par.eta_s[0,lsa]**2
                  D_s[tr[2]] = D_s[tr[2]] + tr[0]
                  sol_lsa_upd = optimisation_sp(lsa=lsa, year=year, mult_lsa=1, num_days=1,
                                          start_day=day, mult_d = 1, mult_s = 1, mult_w = 1, 
                                          D_b = D_b, D_s = D_s, fix_lsa = True, save = False, min_load = True)
                  emis_total.append(sum(sol_lsa_upd.var.emis_hourly))
                  pollution_rate = (emis_total[-1]-emis_total[-2])[0,0]
                  new_row = {
                        'Day': day,
                        'Transaction Amount': tr[0],
                        'Transaction Impact': pollution_rate,
                        'Pollution rate': pollution_rate/tr[0],
                        'Period Buy': tr[1],
                        'Period Sell': tr[2]
                    }
                  couple_new[year][LSA_l[lsa] ].loc[len(couple_new[year][LSA_l[lsa]])] = new_row
        couple_final_min_load[year][LSA_l[lsa]] = pd.concat([couple_new[year][LSA_l[lsa]], couple_min_load[year][LSA_l[lsa]]['df']])
        f = open(f"{path_file_result}couple_min_load_2.pkl","wb")
        pickle.dump(couple_final_min_load,f)
        
        with open(f'{path_file_result}couple_min_load_2.pkl', 'rb') as file:
            couple_final_min_load_2 =  pickle.load(file)
    

P_theoric = {}
P_theoric_min = {}
for year in [2019, 2022]:
    P_theoric[year] = pd.DataFrame(columns = [LSA_l[0], LSA_l[1], LSA_l[2], LSA_l[3], LSA_l[4]])
    P_theoric_min[year] = pd.DataFrame(columns = [LSA_l[0], LSA_l[1], LSA_l[2], LSA_l[3], LSA_l[4]])
    for obs in range(0,365):
        par_temp = sol_sp_min_load[year][1][LSA_l[0]][obs].par_temp
        P_theo = [P_theoric_f(par.eta_b[0,lsa]**2, par.c_bb[0,lsa], par_temp) for lsa in [0,1,2,3,4]]
        P_theoric[year].loc[obs] = [P_theo[lsa][1] for lsa in range(5)]
        P_theoric_min[year].loc[obs] = [P_theo[lsa][0] for lsa in range(5)]
       
     
#compute the maximum theoric pollution rate in the year per LSA 
P_max_theoric = pd.DataFrame(data = {2019: P_theoric[2019].max(), 2022: P_theoric[2022].max()})

        

#graph
#2019
theoric_list = [P_max_theoric[2019][LSA_l[lsa]]for lsa in range(5)]
theoric_min = [P_theoric_min[2019][LSA_l[lsa]].loc[0] for lsa in range(5)]
list_lsa = [LSA_l[0], LSA_l[1], LSA_l[2], LSA_l[3], LSA_l[4]]
fig, ax = plt.subplots()
values = [couple_final_min_load[2019][LSA_l[lsa]]['Pollution rate'] for lsa in range(5)]
ax.boxplot(values, labels=list_lsa)
ax.scatter(range(1, 6), theoric_list, marker='X', label='Theoretical bound', color='red', zorder=4)
ax.scatter(range(1, 6), theoric_min, marker='X', label='Lower theoretical bound', color='blue', zorder=4)
plt.ylabel('CO2 emission rate (ton/Gwh)')
plt.ylim(-1500, 3000)
plt.legend()

#2022
fig, ax = plt.subplots()
theoric_list_x = [P_max_theoric[2022][LSA_l[lsa]]for lsa in range(5)]
theoric_min = [P_theoric_min[2019][LSA_l[lsa]].loc[0] for lsa in range(5)]
values = [couple_final_min_load[2022][LSA_l[lsa]]['Pollution rate'] for lsa in range(5)]
ax.scatter(range(1, 6), theoric_list_x, marker='X', label='Theoretical bound', color='red', zorder=4)
ax.scatter(range(1, 6), theoric_min, marker='X', label='Lower theoretical bound', color='blue', zorder=4)
ax.boxplot(values, labels=list_lsa)#,  showfliers=False)
plt.ylabel('CO2 emission rate (ton/Gwh)')
#plt.ylim(-900, 2600)
plt.legend()


above_daily_threshold =  pd.DataFrame(index = [2019, 2022], columns = LSA_l)
for year in [2019, 2022]:
    for lsa in range(5):
        couple_final_min_load[year][LSA_l[lsa]]['target'] = couple_final_min_load[year][LSA_l[lsa]]['Day'].map(P_theoric[year][LSA_l[lsa]])
        above_daily_threshold.loc[year, LSA_l[lsa]] = sum(couple_final_min_load[year][LSA_l[lsa]].loc[couple_final_min_load[year][LSA_l[lsa]]['Pollution rate'] > couple_final_min_load[year][LSA_l[lsa]]['target'] , 'Transaction Amount'])/sum(couple_final_min_load[year][LSA_l[lsa]]['Transaction Amount']) *100
# Example target per day


bins = [0, 1, 1000]  # Custom bin edges
labels = ["]0,1]", "[1,["]  # Bin names


couple_min_load_new = couple_final_min_load.copy()
for year in [2019, 2022]:
    for lsa in [0,1,2,3,4]:
        couple_min_load_new[year][LSA_l[lsa]]['target'] = couple_min_load_new[year][LSA_l[lsa]]['Day'].map(P_theoric[year][LSA_l[lsa]])
        couple_min_load_new[year][LSA_l[lsa]]['difference target'] = (couple_min_load_new[year][LSA_l[lsa]]['Pollution rate'] - couple_min_load_new[year][LSA_l[lsa]]['target']).clip(lower = 0)
        couple_min_load_new[year][LSA_l[lsa]]['% above target'] = couple_min_load_new[year][LSA_l[lsa]]['difference target']/couple_min_load_new[year][LSA_l[lsa]]['target']
        couple_min_load_new[year][LSA_l[lsa]].loc[couple_min_load_new[year][LSA_l[lsa]]['difference target'] == 0, '% above target'] = 0
        couple_min_load_new[year][LSA_l[lsa]]['Bin'] = pd.cut( couple_min_load_new[year][LSA_l[lsa]]['% above target'] , bins = bins, labels = labels, right = False, include_lowest = True)
        couple_min_load_new[year][LSA_l[lsa]]['Bin'] =  couple_min_load_new[year][LSA_l[lsa]]['Bin'].astype(str)
        couple_min_load_new[year][LSA_l[lsa]].loc[couple_min_load_new[year][LSA_l[lsa]]['% above target'] == 0, 'Bin'] = "[0,0]"
        couple_min_load_new[year][LSA_l[lsa]].loc[(couple_min_load_new[year][LSA_l[lsa]]['target'] == 0) & (couple_min_load_new[year][LSA_l[lsa]]['difference target'] != 0), 'Bin'] = '[1,['
        couple_min_load_new[year][LSA_l[lsa]]['target low'] = couple_min_load_new[year][LSA_l[lsa]]['Day'].map(P_theoric_min[year][LSA_l[lsa]])
        couple_min_load_new[year][LSA_l[lsa]]['difference target low'] = (couple_min_load_new[year][LSA_l[lsa]]['target low'] - couple_min_load_new[year][LSA_l[lsa]]['Pollution rate']).apply(lambda x: max(x, 0))
        couple_min_load_new[year][LSA_l[lsa]]['% below target'] = abs(couple_min_load_new[year][LSA_l[lsa]]['difference target low']/couple_min_load_new[year][LSA_l[lsa]]['target low'])
        couple_min_load_new[year][LSA_l[lsa]].loc[couple_min_load_new[year][LSA_l[lsa]]['difference target low'] == 0, '% below target'] = 0
        couple_min_load_new[year][LSA_l[lsa]]['Bin low'] = pd.cut( couple_min_load_new[year][LSA_l[lsa]]['% below target'] , bins = bins, labels = labels, right = False, include_lowest = True)
        couple_min_load_new[year][LSA_l[lsa]]['Bin low'] =  couple_min_load_new[year][LSA_l[lsa]]['Bin low'].astype(str)
        couple_min_load_new[year][LSA_l[lsa]].loc[couple_min_load_new[year][LSA_l[lsa]]['% below target'] == 0, 'Bin low'] = "[0,0]"
        couple_min_load_new[year][LSA_l[lsa]].loc[(couple_min_load_new[year][LSA_l[lsa]]['target low'] == 0) & (couple_min_load_new[year][LSA_l[lsa]]['difference target low'] != 0), 'Bin low'] = '[1,['
        couple_min_load_new[year][LSA_l[lsa]]['combined bin'] = couple_min_load_new[year][LSA_l[lsa]]['Bin'] + couple_min_load_new[year][LSA_l[lsa]]['Bin low']
     

# Sum 'Amount' within each bin
x = couple_min_load_new[year][LSA_l[lsa]].groupby('combined bin')['Transaction Amount'].sum().reset_index()['combined bin']

in_interval = {}
for year in [2019, 2022]:
    in_interval[year] = pd.DataFrame(index = x)
    for lsa in [0,1,2,3,4]:
        in_interval[year][LSA_l[lsa]] =  couple_min_load_new[year][LSA_l[lsa]].groupby('combined bin')['Transaction Amount'].sum()
    in_interval[year] = in_interval[year].div(in_interval[year].sum())*100
    in_interval[year] = in_interval[year].fillna(0)


x = couple_min_load_new[year][LSA_l[lsa]].groupby('Bin')['Transaction Amount'].sum().reset_index()['Bin']

above_threshold = {}
for year in [2019, 2022]:
    above_threshold[year] = pd.DataFrame(index = x)
    for lsa in [0,1,2,3,4]:
        above_threshold[year][LSA_l[lsa]] =  couple_min_load_new[year][LSA_l[lsa]].groupby('Bin')['Transaction Amount'].sum()
    above_threshold[year] = above_threshold[year].div(above_threshold[year].sum())*100
    above_threshold[year] = above_threshold[year].fillna(0)

below_threshold = {}
for year in [2019, 2022]:
    below_threshold[year] = pd.DataFrame(index = x)
    for lsa in [0,1,2,3,4]:
        below_threshold[year][LSA_l[lsa]] =  couple_min_load_new[year][LSA_l[lsa]].groupby('Bin low')['Transaction Amount'].sum()
    below_threshold[year] = below_threshold[year].div(below_threshold[year].sum())*100
    below_threshold[year] = below_threshold[year].fillna(0)

#start up cost impact
sol_start_c= {}
for year in [2019, 2022]:
    sol_start_c[year] = {}
    for lsa in range(5):
        sol_start_c[year][LSA_l[lsa]] = open_file(lsa, start_cost = True, year = year)
    sol_start_c[year]['No Lsa'] = open_file(3, mult_lsa = 0, start_cost = True, year = year)

Impact_start_c = {}
for year in [2019, 2022]:
    Impact_start_c[year] = pd.DataFrame()
    for lsa in range(5):
        Impact_start_c[year][LSA_l[lsa]] = change_ss_c(sol_start_c[year][LSA_l[lsa]], sol_sp_ss_c[year][LSA_l[lsa]])
    Impact_start_c[year]['No Lsa'] = change_ss_c(sol_start_c[year]['No Lsa'], sol_sp_ss_c[year]['No Lsa'])
    Impact_start_c[year].loc['change_emis %'] =     (Impact_start_c[year].loc['emis_total']/  Impact_start_c[year].loc['emis comparison'] - 1)*100



# ==========================================================
#Additional figure : Changes in total emission
# =============================================================================
Emission[2022][1][1:].mean(axis = 1).plot(label = 2019, xlabel = 'Carbon levy level (€/ton)', ylabel = 'Total emission of the system (Ton)', title = '2019', color = 'black')
Emission[2019][1][1:].mean(axis = 1).plot(label = 2022, xlabel = 'Carbon levy level (€/ton)', ylabel = 'Total emission of the system (Ton)', title = '2022', color = 'black')

# ==========================================================
#Additional : analysis of piecewise cost. 
# =============================================================================
#this part provide similar results as what we have in the regular case but when each generators have 3 pieces of efficiency thus 3 pieces of marginal cost and marginal emissions

path_file_optimisation =  os.path.join(userpath,"Data\\result_optimisation\\All\\")

sol_piecewise= {}
for year in [2019, 2022]:
    sol_piecewise[year] = {}
    for j in range(1,2):
        sol_piecewise[year][j]={}
        for lsa in range(5):
            sol_piecewise[year][j][LSA_l[lsa]] = open_file(lsa, year = year, n_p = 3, mult_lsa=j)
    sol_piecewise[year]['No Lsa'] = open_file(lsa = 3, mult_lsa = 0, year = year, n_p =3)    

result_piecewise={}
for year in [2019, 2022]:
    result_piecewise[year]={}
    for j in range(1,7):
        print(j)
        result_piecewise[year][j]={}
        for i in  [0,1,2,3,4]:
            result_piecewise[year][j][LSA_l[i]] = agg_sol(sol_piecewise[year][j][LSA_l[i]], sol_piecewise[year]['No Lsa'], n_p = 3)


result_piecewise_period_all={}
change_result = {}
for year in [2019, 2022]:
    result_piecewise_period_all[year]={}
    for j in range(1,11):
        print(j)
        result_piecewise_period_all[year][j]=pd.DataFrame()
        for i in [0,1,2,3,4]:
            result_piecewise_period_all[year][j][LSA_l[i]] = round(result_piecewise[year][j][LSA_l[i]].sum(),5)


Emission_capacity ={}
for year in [2019, 2022]:  
    Emission_capacity[year] = pd.DataFrame()
    for j in range(1,11): 
        for i in [0,1,2,3,4]:
            Emission_capacity[year].loc[j,LSA_l[i]]=result_piecewise_period_all[year][j][LSA_l[i]]['diff emis']/result_piecewise_period_all[year][j][LSA_l[i]]['Emission']*100
    Emission_capacity[year]['index']=([10,20,30,40,50,60,70,80,90,100])
    Emission_capacity[year].set_index('index', inplace = True)
    
f = open(f'{path_file_result}piecewise_capacity.pkl',"wb")
pickle.dump(Emission_capacity,f)

#and load it
with open(f'{path_file_result}piecewise_capacity.pkl', 'rb') as file:
        Emission_capacity = pickle.load(file)
    
for i, lsa in enumerate(Emission_capacity[year].columns):
    Emission_capacity[year][lsa].plot(color=colors_dict[lsa], linestyle=linestyles_dict[lsa])
plt.legend()
plt.ylabel('Storage marginal CO2 emission (%)')
plt.xlabel('Share of daily average demand (%)')
plt.ylim(-1.35, 0.85)

impact_piecewise = {}
impact_piecewise[year]  = change_ss_c(sol_piecewise[year]['No Lsa'], sol_sp_basic[year], n_p = 3)
impact_piecewise[year].loc['change_emis %'] =     (impact_piecewise[year].loc['emis_total']/  impact_piecewise[year].loc['emis comparison'] - 1)*100


couple_piecewise = {}
for year in [2019,2022]:
    couple_piecewise[year] = {}
    for lsa in range(5):
        couple_piecewise[year][LSA_l[lsa]] = coupling(sol_piecewise[year][1][LSA_l[lsa]], sol_piecewise[year]['No Lsa'], lsa, '3_p')
       

P_theoric = {}
P_theoric_min = {}
for year in [2019, 2022]:
    P_theoric[year] = pd.DataFrame(columns = [LSA_l[0], LSA_l[1], LSA_l[2], LSA_l[3], LSA_l[4]])
    P_theoric_min[year] = pd.DataFrame(columns = [LSA_l[0], LSA_l[1], LSA_l[2], LSA_l[3], LSA_l[4]])
    for obs in range(0,365):
        par_temp = sol_piecewise[year][1][LSA_l[0]][obs].par_temp
        P_theo = [P_theoric_f(par.eta_b[0,lsa]**2, par.c_bb[0,lsa], par_temp, n_p=3) for lsa in [0,1,2,3,4]]
        P_theoric[year].loc[obs] = [P_theo[lsa][1] for lsa in [0,1,2,3,4]]
        P_theoric_min[year].loc[obs] = [P_theo[lsa][0] for lsa in [0,1,2,3,4]]
       
     

P_max_theoric = pd.DataFrame(data = {2019: P_theoric[2019].max(), 2022: P_theoric[2022].max()})
P_min_theoric =  pd.DataFrame(data = {2019: P_theoric_min[2019].min(), 2022: P_theoric_min[2022].min()})
#2019
year = 2022
theoric_list_x = [P_max_theoric[year][LSA_l[lsa]]for lsa in range(5)]
theoric_min = [P_min_theoric[year][LSA_l[lsa]] for lsa in range(5)]
list_lsa = [LSA_l[0], LSA_l[1], LSA_l[2], LSA_l[3], LSA_l[4]]
fig, ax = plt.subplots()
values = [couple_piecewise[year][LSA_l[lsa]]['df']['Pollution rate'] for lsa in range(5)]
ax.boxplot(values, labels=list_lsa)
ax.scatter(range(1, 6), theoric_list_x, marker='X', label='Upper theoretical bound', color='red', zorder=4)
ax.scatter(range(1, 6), theoric_min, marker='X', label='Lower theoretical bound', color='blue', zorder=4)
plt.ylabel('CO2 emission rate (ton/Gwh)')
#plt.ylim(-2200, 2600)
plt.legend()

#2022
fig, ax = plt.subplots()
theoric_list_x = [P_max_theoric[2022][LSA_l[lsa]]for lsa in range(5)]
values = [couple_piecewise[2022][LSA_l[lsa]]['Pollution rate'] for lsa in range(5)]
ax.scatter(range(1, 6), theoric_list_x, marker='X', label='Theoretical bound', color='red', zorder=4)
ax.boxplot(values, labels=list_lsa)#,  showfliers=False)
plt.ylabel('CO2 emission rate (ton/Gwh)')
#plt.ylim(-900, 10000)
plt.legend()


#merit order curve :
#compute the marginal cost for each day and for each fuel type' 
df_marginal_emission = {}
df_capacity_emission ={}
df_marginal_cost = {}
df_plants_count = {}
marginal_cost_gas = {}
marginal_cost_coal = {}
marginal_cost_gas_cc = {}
marginal_cost_nuclear = {}
for year in [2019,2022]:
    if year == 2019:
        l = 216 
    else:
        l = 228
    marginal_emission = np.empty(shape=(365,l))
    capacity_emission = np.empty(shape=(365,l))
    marginal_cost = np.empty(shape=(365,l))
    df_plants = pd.DataFrame(columns = range(0,l))
    df_plants_count[year] = pd.DataFrame(columns = ['Nuclear', 'Coal', 'Gas Peaker', 'Gas_cc'])
    for obs in range(0,365):
        par_temp = sol_piecewise[year]['No Lsa'][obs].par_temp
        num_plants = par_temp.shape[0]
        num_segments = 3  # We know there are 3 MC values per plant
        marginal_costs = par_temp[:,[-2,-3,-1]].astype(float).reshape(-1, 1)  # Converts to numeric
        type_g = np.repeat(par_temp[:, 1], num_segments).reshape(-1, 1)
        capacities = np.repeat(par_temp[:, 0]/3, num_segments).reshape(-1, 1)
        emission = par_temp[:, [10, 6, 9]].astype(float).reshape(-1, 1)
        segments = np.tile(np.arange(1, num_segments + 1), num_plants).reshape(-1, 1)
        reshaped_arr = np.hstack((type_g, capacities, segments, emission, marginal_costs))
        reshaped_arr = reshaped_arr[reshaped_arr[:, -1].argsort()]

        marginal_emission[obs] = np.repeat(reshaped_arr[:,3], 2)
        capacity =  np.repeat(reshaped_arr[:,1].cumsum(), 2)   
        capacity = np.insert(capacity[0:-1], 0, 0)     
        capacity_emission[obs] = capacity
        marginal_cost[obs] = np.repeat(reshaped_arr[:,-1], 2)
        df_plants.loc[obs] = np.repeat(reshaped_arr[:,0], 2)
    df_marginal_emission[year] = marginal_emission.mean(axis = 0) #yearly average
    df_capacity_emission[year] = capacity_emission.mean(axis = 0)
    df_marginal_cost[year] = marginal_cost.mean(axis = 0)
  #You need to get the cost such that it represents the share of the technology at that time
# Iterate through each column and calculate value counts
    for col in df_plants.columns:
        df_plants_count[year].loc[col] = df_plants[col].value_counts()/365
    df_plants_count[year] = df_plants_count[year].fillna(0)
    marginal_cost_nuclear[year] = df_marginal_cost[year]*df_plants_count[year]['Nuclear']
    marginal_cost_gas[year] = df_marginal_cost[year]*df_plants_count[year]['Gas Peaker']
    marginal_cost_coal[year] = df_marginal_cost[year]*df_plants_count[year]['Coal']
    marginal_cost_gas_cc[year] = df_marginal_cost[year]*df_plants_count[year]['Gas_cc']
    df_capacity_emission[year] = df_capacity_emission[year]/1000



#graph 2019
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0, height_ratios = [0.20, 0.8])
axs = gs.subplots(sharex=True)
axs[0].boxplot(Dr_2019, notch = False, sym ='', vert = False, labels = [''])
axs[0].set_axis_off()   
axs[1].scatter(df_capacity_emission[2019],df_marginal_cost[2019], alpha = 0)
axs[1].fill_between(list(df_capacity_emission[2019]), list(marginal_cost_nuclear[2019]),  facecolor='silver', label = 'Nuclear', step = 'pre')
axs[1].fill_between(list(df_capacity_emission[2019]), list(marginal_cost_coal[2019]),  facecolor='black', label = 'Coal', step = 'pre')
axs[1].fill_between(list(df_capacity_emission[2019]),  list(marginal_cost_coal[2019]), list(marginal_cost_coal[2019] + marginal_cost_gas_cc[2019]), facecolor='darkgrey', label = 'Gas_cc')
axs[1].fill_between(list(df_capacity_emission[2019]), list(marginal_cost_coal[2019] + marginal_cost_gas_cc[2019]),  list(df_marginal_cost[2019]), facecolor='dimgrey', label = 'Gas Peaker')
ax2 = axs[1].twinx() 
ax2.set_ylabel('CO2 emission (ton/Gwh)')
ax2.plot(list(df_capacity_emission[2019]), list(df_marginal_emission[2019]), color = 'black')  
axs[1].legend(loc = 'upper left')
ax2.legend()
axs[1].set_frame_on(False)
axs[1].set_xlabel('Capacity (Gwh)')
axs[1].set_ylabel('Price (1000€/GWh)')
ax2.set_frame_on(False)

#graph 2022
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0, height_ratios = [0.20, 0.8])
axs = gs.subplots(sharex=True)
axs[0].boxplot(Dr_2022, notch = False, sym ='', vert = False, labels = [''])
axs[0].set_axis_off()   
axs[1].scatter(df_capacity_emission[2022],df_marginal_cost[2022], alpha = 0)
axs[1].fill_between(list(df_capacity_emission[2022]), list(marginal_cost_nuclear[2022]),  facecolor='silver', label = 'Nuclear', step = 'pre')
axs[1].fill_between(list(df_capacity_emission[2022]), list(marginal_cost_coal[2022]),  facecolor='black', label = 'Coal', step = 'pre')
axs[1].fill_between(list(df_capacity_emission[2022]), list(marginal_cost_coal[2022]),  facecolor='black', label = 'Coal', step = 'pre')
axs[1].fill_between(list(df_capacity_emission[2022]),  list(marginal_cost_coal[2022]), list(marginal_cost_coal[2022] + marginal_cost_gas_cc[2022]), facecolor='darkgrey', label = 'Gas_cc')
axs[1].fill_between(list(df_capacity_emission[2022]), list(marginal_cost_coal[2022] + marginal_cost_gas_cc[2022]),  list(df_marginal_cost[2022]), facecolor='dimgrey', label = 'Gas')
ax2 = axs[1].twinx() 
ax2.set_ylabel('CO2 emission (ton/Gwh)')
ax2.plot(list(df_capacity_emission[2022]), list(df_marginal_emission[2022]), color = 'black', label = 'Average marginal emission')   
ax2.legend(loc = 'upper right')
axs[1].set_frame_on(False)
axs[1].set_xlabel('Capacity (Gwh)')
axs[1].set_ylabel('Price (1000€/GWh)')
ax2.set_frame_on(False)





