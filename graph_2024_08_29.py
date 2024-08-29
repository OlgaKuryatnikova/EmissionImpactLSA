# -*- coding: utf-8 -*-
"""
Created on 29/08/2024
@author: Clarisse Dupont
Code to create the graph of the paper

"""


# remove all existing variables
from IPython import get_ipython
get_ipython().magic('reset -sf')

#Change the source
path_file_source= "C:\\Users\\78160cdu\Dropbox (Erasmus Universiteit Rotterdam)\\EnergyMarket\\Data\\source\\"
path_file_optimisation = "C:\\Users\\78160cdu\Dropbox (Erasmus Universiteit Rotterdam)\\EnergyMarket\\Data\\tax\\"
path_file_result = "C:\\Users\\78160cdu\Dropbox (Erasmus Universiteit Rotterdam)\\EnergyMarket\\Data\\result\\"

#load required package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle

class Object(object):
    pass


'''load source files and some adjustemnts'''
par_temp_ori = pd.read_csv(f"{path_file_source}Parameters_plants_NL_2.csv", sep=';', usecols=[1,7,8,10,12])
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
Dr_2019[Dr_2019 > par_temp_ori['capacity_g'].cumsum().iloc[-1]/1000] = 16.771



D_2022 = D_full[1096*24:1461*24+24]
Wind = np.reshape((Q_r_full['Wind MWH'][1096*24:1461*24+24]).values, (8760, 1))
Solar = np.reshape(
    (Q_r_full['Solar MWH'][1096*24:1461*24+24]).values, (8760, 1))
Dr_2022 = D_2022 - Wind - Solar - 0.447  # Residual Demand
Dr_2022[Dr_2022 > par_temp_ori['capacity_g'].cumsum().iloc[-1]/1000]  = 16.771


'''Set up main prameters'''
LSA_l = ['PHS', 'CAES', 'Battery', 'DR', 'Hydrogen']
par = Object() # This is to store all the elements that are stable through the days
par.T = 24  # number of periods per day
par.eta_b = np.sqrt(
    np.matrix([0.76, 0.795, 0.9, 1, 0.375]))
# share of the generated energy that reaches the customer when discharged
par.eta_s = par.eta_b #assumed to be the same
par.c_bs = np.matrix([0.11, 1.55, 1.25, 0.01, 0.5])  # LSA energy sell cost
par.c_bb = par.c_bs  # LSA energy buy cost assume to be the same as sell
par.rate = np.matrix([0.12, 0.19, 0.57, 1, 1]) #how much energy (as percent of the capacity) can be charged or discharge in an hour
par.pmin = 0 # price of the renewable
par.n_f = len(par_temp_ori)  # number of fossil fuel operators, do not include renewables
par.n_tot = par.n_f + 1 #number of generators, including renewables


'''Functions used in the file'''             
def open_file(lsa,  mult_lsa = 1,  year = 2019, tax ='ETS', subsidy = 0):
    efficiency = round(par.eta_b[0,lsa],2)
    m_cost = round(par.c_bs[0,lsa],2)
    dc_rate = round(par.rate[0,lsa],2)
    if year == 2019:
        add_days = 0
    else:
        add_days = 365*2+366
    file = []
    text = f'{path_file_optimisation}sol_sp_cap_'+str(mult_lsa)+'_eff_'+','.join(str(efficiency).split('.'))+'_mcost_'+','.join(str(m_cost).split('.'))+'_dc_rate_'+','.join(str(dc_rate).split('.'))+'_'+str(year)+'_tax_'+','.join(str(tax).split('.'))+'_'+str(add_days)+'_'+str(365+add_days)
    with open(text, 'rb') as fr:
        try:
            while True:
                  file.append(pickle.load(fr))
        except EOFError:
            pass  
    return(file)
 

def diff_agg(name_input, T = 24):
    s = np.zeros((T*len(name_input),1))
    emis_wobda = np.zeros((T*len(name_input)))
    emis = np.zeros((T*len(name_input)))
    Price = np.zeros((T*len(name_input)))
    clearing = np.zeros((T*len(name_input)))
    clearing_wobda = np.zeros((T*len(name_input)))
    wb = np.zeros((T*len(name_input)))
    s_lsa = np.zeros((T*len(name_input)))
    obj = np.zeros((T*len(name_input)))
    for kk in range(len(name_input)):
        #emis[T*(kk):T*(kk+1),] = np.reshape(name_input[kk].var.emis_hourly,(24,))
        emis[T*(kk):T*(kk+1),] = [0 if np.round(i,4) == 0 else i for i in np.reshape(name_input[kk].var.emis_hourly,(24,)).tolist()[0]]
        Price[T*(kk):T*(kk+1),] = name_input[kk].var.price
        #emis_wobda[T*(kk):T*(kk+1),] = np.reshape(name_input[kk].var.emis_hourly_wobda,(24,))
        emis_wobda[T*(kk):T*(kk+1),] = [0 if np.round(i,4) == 0 else i for i in np.reshape(name_input[kk].var.emis_hourly_wobda,(24,)).tolist()[0]]
        wb[T*(kk):T*(kk+1)] = [0 if np.round(i,4) == 0 else i for i in np.reshape(name_input[kk].var.w_b,(24,)).tolist()[0]] #before you have to add [0] at the end #maybe with daily optimisation it is still the case, to be checked. 
        s_lsa[T*(kk):T*(kk+1)] = [0 if np.round(i,4) == 0 else i for i in np.reshape(name_input[kk].var.q_b,(24,)).tolist()[0]]
        #wb[T*(kk):T*(kk+1)] =  np.reshape(name_input[kk].var.w_b,(24,)).tolist()[0]
        #s_lsa[T*(kk):T*(kk+1)] = np.reshape(name_input[kk].var.q_b,(24,)).tolist()[0]
        s[T*(kk):T*(kk+1)] = name_input[kk].var.s[1:]
        obj[T*(kk):T*(kk+1)] = name_input[kk].obj   
        clearing[T*(kk):T*(kk+1)] =  name_input[kk].var.num_pc
        #days[T*(kk):T*(kk+1)]  = [name_input[kk].days]*T #does not exist for not tr
        #s(name_input[kk].var.bl!=0).argmax(axis=1).tolist()
        clearing_wobda[T*(kk):T*(kk+1)] =  name_input[kk].var.num_pc_wobda
    clearing = [i-1 if i !=0 else 'RES' for i in clearing] #before it is the first one to not produce so now it is the one that clear the market
    clearing_wobda = [i-1 if i !=0 else 'RES' for i in clearing_wobda]
    #what happens when you are at the limit, handled in the previous code ?
    emis_dif = emis - emis_wobda
    q_dif_agg = pd.DataFrame()
    q_dif_agg['Emission'] = emis.tolist()
    q_dif_agg['diff emis'] = emis_dif.tolist()
    q_dif_agg['Buy LSA']= wb.tolist()
    q_dif_agg['Sell LSA']= s_lsa.tolist()
    mask_col1_greater = q_dif_agg['Buy LSA'] > q_dif_agg['Sell LSA']
    # Set col2 to 0 where col1 is greater, and col1 to 0 where col2 is greater or equal
    q_dif_agg['Sell LSA'] = q_dif_agg['Sell LSA'].where(~mask_col1_greater, 0)
    q_dif_agg['Buy LSA'] = q_dif_agg['Buy LSA'].where(mask_col1_greater, 0)
    q_dif_agg.loc[(q_dif_agg['Buy LSA'] == 0) & (q_dif_agg['Sell LSA'] == 0), 'diff emis'] = 0
    q_dif_agg['soc']= s
    q_dif_agg['obj']= obj
    q_dif_agg['clearing_techno'] = clearing
    q_dif_agg['clearing_techno_wobda'] = clearing_wobda
    q_dif_agg['Price'] = Price.tolist()
    q_dif_agg['Profit'] = (q_dif_agg['Sell LSA']-q_dif_agg['Buy LSA'])*q_dif_agg['Price']
    day_list =[]
    for i in range(len(name_input)):
        day_list = day_list + [i]*24
    q_dif_agg['Days'] = day_list
    return(q_dif_agg)

def diff_agg_daily (name_input, sub = False):
    result_daily = pd.DataFrame()
    result_daily['Emission'] = name_input.groupby('Days')['Emission'].sum() 
    result_daily['diff emis'] = name_input.groupby('Days')['diff emis'].sum()
    result_daily['Buy LSA'] = name_input.groupby('Days')['Buy LSA'].sum()
    result_daily['Sell LSA'] = name_input.groupby('Days')['Sell LSA'].sum()
    result_daily['obj'] = name_input.groupby('Days')['obj'].mean()
    if sub:
        result_daily['Profit'] = name_input.groupby('Days')['Profit'].sum() + name_input.groupby('Days')['subsidy'].mean() 
    else:
        result_daily['Profit'] = name_input.groupby('Days')['Profit'].sum()  
    return (result_daily)


def diff_nolsa(name_input, T = 24):
    emis_wobda = np.zeros((T*len(name_input)))
    Price = np.zeros((T*len(name_input)))
    clearing_wobda = np.zeros((T*len(name_input)))
    emis_rate= np.zeros((T*len(name_input)))
    obj = np.zeros((T*len(name_input)))
    for kk in range(len(name_input)):
        Price[T*(kk):T*(kk+1),] = name_input[kk].var.price_wobda
        emis_wobda[T*(kk):T*(kk+1),] = np.reshape(name_input[kk].var.emis_hourly_wobda,(24,))
        obj[T*(kk):T*(kk+1)] = name_input[kk].obj_wobda   
        clearing_wobda[T*(kk):T*(kk+1)] =  name_input[kk].var.num_pc_wobda
        emis_rate[T*(kk):T*(kk+1)] =  name_input[kk].var.e_wobda
    clearing_wobda = [i-1 if i !=0 else 'RES' for i in clearing_wobda]
    q_dif_agg = pd.DataFrame()
    q_dif_agg['Emission'] = emis_wobda.tolist()
    q_dif_agg['diff emis'] = [0]*len(emis_wobda)
    q_dif_agg['Buy LSA']= [0]*len(emis_wobda)
    q_dif_agg['Sell LSA']= [0]*len(emis_wobda)
    q_dif_agg.loc[(q_dif_agg['Buy LSA'] == 0) & (q_dif_agg['Sell LSA'] == 0), 'diff emis'] = 0
    q_dif_agg['soc']= [float(name_input[0].var.s[0])] * len(emis_wobda)
    q_dif_agg['obj']= obj #not sure what to do 
    q_dif_agg['clearing_techno'] = clearing_wobda
    q_dif_agg['clearing_techno_wobda'] = clearing_wobda
    q_dif_agg['emis_rate_clearing_techno'] = emis_rate
    q_dif_agg['Price'] = Price.tolist()
    q_dif_agg['Profit'] = (q_dif_agg['Sell LSA']-q_dif_agg['Buy LSA'])*q_dif_agg['Price'] 
    day_list =[]
    for i in range(len(name_input)):
        day_list = day_list + [i]*24
    q_dif_agg['Days'] = day_list
    return(q_dif_agg)


def diff_agg_daily (name_input, sub = False):
    result_daily = pd.DataFrame()
    result_daily['Emission'] = name_input.groupby('Days')['Emission'].sum() 
    result_daily['diff emis'] = name_input.groupby('Days')['diff emis'].sum()
    result_daily['Buy LSA'] = name_input.groupby('Days')['Buy LSA'].sum()
    result_daily['Sell LSA'] = name_input.groupby('Days')['Sell LSA'].sum()
    result_daily['obj'] = name_input.groupby('Days')['obj'].mean()
    if sub:
        result_daily['Profit'] = name_input.groupby('Days')['Profit'].sum() + name_input.groupby('Days')['subsidy'].mean() 
    else:
        result_daily['Profit'] = name_input.groupby('Days')['Profit'].sum()  
    return (result_daily)


def coupling(df ,LSA_nb):    
    efficiency = par.eta_b[0,lsa]**2
    m_cost = par.c_bb[0,lsa]
    test_transaction_ori = df.copy()
    test_transaction_ori['LSA'] = test_transaction_ori['Sell LSA'] - test_transaction_ori['Buy LSA']
    test_transaction_ori = test_transaction_ori.drop(test_transaction_ori[test_transaction_ori.LSA== 0].index)
    test_transaction = test_transaction_ori.copy()
    transaction_amount=[]
    transaction_impact=[]  
    periods_buy = []
    periods_sell =[]
    T_clearing_buy = []
    T_clearing_sell = []
    T_clearing_buy_w = []
    T_clearing_sell_w = []
    Issue = []   
    day = []
    Profit = []
    while test_transaction.empty == False:
        if len(test_transaction) == 1 or test_transaction['Days'].iloc[0]!=test_transaction['Days'].iloc[1]:#only one thing remaining per day
            day.append(test_transaction['Days'].iloc[0])
            Issue.append(test_transaction['diff emis'].iloc[0])
            transaction_amount.append(max(test_transaction['Buy LSA'].iloc[0], test_transaction['Sell LSA'].iloc[0]))
            transaction_impact.append(0)
            periods_buy.append(int(test_transaction.iloc[0:1].index.values))
            periods_sell.append(0)
            T_clearing_buy.append(test_transaction['clearing_techno'].iloc[0]) #i put at buy but can be sell depending what is left
            T_clearing_sell.append(0)
            T_clearing_buy_w.append(test_transaction['clearing_techno_wobda'].iloc[0]) #i put at buy but can be sell depending what is left
            T_clearing_sell_w.append(0)
            Profit.append(max(test_transaction['Profit'].iloc[0], test_transaction['Profit'].iloc[0]))
            test_transaction = test_transaction.drop(test_transaction.iloc[0:1].index)
        elif sum(test_transaction['Buy LSA'])==0 or sum(test_transaction['Sell LSA'])==0:
            day.append(test_transaction['Days'].iloc[0])
            Issue.append(sum(test_transaction['diff emis'].iloc[0:-1]))
            transaction_amount.append(max(test_transaction['Buy LSA'].iloc[0], test_transaction['Sell LSA'].iloc[0]))
            transaction_impact.append(0)
            periods_buy.append(int(test_transaction.iloc[0:1].index.values))
            periods_sell.append(0)
            T_clearing_buy.append(test_transaction['clearing_techno'].iloc[0]) #i put at buy but can be sell depending what is left
            T_clearing_sell.append(0)
            T_clearing_buy_w.append(test_transaction['clearing_techno_wobda'].iloc[0]) #i put at buy but can be sell depending what is left
            T_clearing_sell_w.append(0)            
            Profit.append(test_transaction['Profit'].iloc[0])
            test_transaction = test_transaction.drop(test_transaction.iloc[0:-1].index)
        else:
            if test_transaction['LSA'].iloc[0] < 0: #'then Buy'
                l = 1
                while np.sign(test_transaction['LSA'].iloc[0]) == np.sign(test_transaction['LSA'].iloc[l]) and test_transaction['Days'].iloc[0] ==  test_transaction['Days'].iloc[l]:
                    l = l+1
                if  test_transaction['Days'].iloc[0] !=  test_transaction['Days'].iloc[l]:
                    Issue.append(sum(test_transaction['diff emis'].iloc[0:l+1]))
                    transaction_amount.append(max(test_transaction['Buy LSA'].iloc[0], test_transaction['Sell LSA'].iloc[0]))
                    transaction_impact.append(0)
                    periods_buy.append(int(test_transaction.iloc[0:1].index.values))
                    periods_sell.append(0)
                    T_clearing_buy.append(test_transaction['clearing_techno'].iloc[0]) #i put at buy but can be sell depending what is left
                    T_clearing_sell.append(0)
                    T_clearing_buy_w.append(test_transaction['clearing_techno_wobda'].iloc[0]) #i put at buy but can be sell depending what is left
                    T_clearing_sell_w.append(0)
                    day.append(test_transaction['Days'].iloc[0])
                    Profit.append(test_transaction['Profit'].iloc[0])
                    test_transaction = test_transaction.drop(test_transaction.iloc[0:l+1].index)
                    pass
                index_sell = test_transaction.iloc[l:l+1,].index
                index_buy= test_transaction.iloc[0:1].index
                periods_buy.append(int(index_buy.values))
                periods_sell.append(int(index_sell.values))
                T_clearing_buy.append(test_transaction['clearing_techno'].iloc[0])
                T_clearing_sell.append(test_transaction['clearing_techno'].iloc[l])
                T_clearing_buy_w.append(test_transaction['clearing_techno_wobda'].iloc[0]) #i put at buy but can be sell depending what is left
                T_clearing_sell_w.append(test_transaction['clearing_techno_wobda'].iloc[l]) 
                day.append(test_transaction['Days'].iloc[0])
                Issue.append(0)
                if test_transaction['Sell LSA'].iloc[l] >= test_transaction['Buy LSA'].iloc[0]*par.eta_b[0,LSA_nb]**2:
                    transaction_amount.append(test_transaction['Buy LSA'].iloc[0]*par.eta_b[0,LSA_nb]**2)
                    transaction_impact.append(test_transaction['diff emis'].iloc[0]+test_transaction['diff emis'].iloc[l]*(transaction_amount[-1]/test_transaction['Sell LSA'].iloc[l]))
                    test_transaction.loc[index_sell, 'diff emis'] = test_transaction['diff emis'].iloc[l]*(1-(transaction_amount[-1]/test_transaction['Sell LSA'].iloc[l]))
                    test_transaction.loc[index_sell, 'Sell LSA'] = test_transaction['Sell LSA'].iloc[l] - transaction_amount[-1]
                    test_transaction = test_transaction.drop(index_buy)
                else: 
                    transaction_amount.append(test_transaction['Sell LSA'].iloc[l])
                    transaction_impact.append(test_transaction['diff emis'].iloc[l]+test_transaction['diff emis'].iloc[0]*(transaction_amount[-1]/(test_transaction['Buy LSA'].iloc[0]*par.eta_b[0,LSA_nb]**2)))
                    test_transaction.loc[index_buy, 'diff emis'] = test_transaction['diff emis'].iloc[0]*(1-(transaction_amount[-1]/(test_transaction['Buy LSA'].iloc[0]*par.eta_b[0,LSA_nb]**2)))
                    test_transaction.loc[index_buy, 'Buy LSA'] = test_transaction['Buy LSA'].iloc[0] - transaction_amount[-1]/par.eta_b[0,LSA_nb]**2
                    test_transaction = test_transaction.drop(index_sell) 
            else: #Then sell
                l= 1
                while np.sign(test_transaction['LSA'].iloc[0]) == np.sign(test_transaction['LSA'].iloc[l]) and test_transaction['Days'].iloc[0] ==  test_transaction['Days'].iloc[l]:
                    l = l+1
                if  test_transaction['Days'].iloc[0] !=  test_transaction['Days'].iloc[l]:
                    Issue.append(sum(test_transaction['diff emis'].iloc[0:l+1]))
                    transaction_amount.append(max(test_transaction['Buy LSA'].iloc[0], test_transaction['Sell LSA'].iloc[0]))
                    transaction_impact.append(0)
                    periods_buy.append(int(test_transaction.iloc[0:1].index.values))
                    periods_sell.append(0)
                    T_clearing_buy.append(test_transaction['clearing_techno'].iloc[0]) #i put at buy but can be sell depending what is left
                    T_clearing_sell.append(0) 
                    T_clearing_buy_w.append(test_transaction['clearing_techno_wobda'].iloc[0]) #i put at buy but can be sell depending what is left
                    T_clearing_sell_w.append(test_transaction['clearing_techno_wobda'].iloc[l]) 
                    test_transaction = test_transaction.drop(test_transaction.iloc[0:l+1].index)
                    day.append(test_transaction['Days'].iloc[0])
                    pass
                index_buy = test_transaction.iloc[l:l+1,].index
                index_sell= test_transaction.iloc[0:1].index
                periods_buy.append(int(index_buy.values))
                periods_sell.append(int(index_sell.values))
                T_clearing_buy.append(test_transaction['clearing_techno'].iloc[l])
                T_clearing_sell.append(test_transaction['clearing_techno'].iloc[0])
                T_clearing_buy_w.append(test_transaction['clearing_techno_wobda'].iloc[l]) #i put at buy but can be sell depending what is left
                T_clearing_sell_w.append(test_transaction['clearing_techno_wobda'].iloc[0]) 
                day.append(test_transaction['Days'].iloc[0])
                Issue.append(0)             
                if test_transaction['Sell LSA'].iloc[0] <= test_transaction['Buy LSA'].iloc[l]*par.eta_b[0,LSA_nb]**2:
                    transaction_amount.append(test_transaction['Sell LSA'].iloc[0])
                    transaction_impact.append(test_transaction['diff emis'].iloc[0]+test_transaction['diff emis'].iloc[l]*(transaction_amount[-1]/(test_transaction['Buy LSA'].iloc[l]*par.eta_b[0,LSA_nb]**2)))
                    test_transaction.loc[index_buy,'diff emis'] = test_transaction['diff emis'].iloc[l]*(1-(transaction_amount[-1]/(test_transaction['Buy LSA'].iloc[l]*par.eta_b[0,LSA_nb]**2)))
                    test_transaction.loc[index_sell, 'Profit'] = test_transaction['Profit'].iloc[l]*(1-(transaction_amount[-1]/(test_transaction['Buy LSA'].iloc[l]*par.eta_b[0,LSA_nb]**2)))
                    test_transaction.loc[index_buy, 'Buy LSA'] = test_transaction['Buy LSA'].iloc[l] - transaction_amount[-1]/par.eta_b[0,LSA_nb]**2
                    test_transaction = test_transaction.drop(index_sell)
                else: 
                    transaction_amount.append(test_transaction['Buy LSA'].iloc[l]*par.eta_b[0,LSA_nb]**2)
                    transaction_impact.append(test_transaction['diff emis'].iloc[l]+test_transaction['diff emis'].iloc[0]*(transaction_amount[-1]/(test_transaction['Sell LSA'].iloc[0])))
                    test_transaction.loc[index_sell, 'diff emis'] = test_transaction['diff emis'].iloc[0]*(1-(transaction_amount[-1]/(test_transaction['Sell LSA'].iloc[0])))
                    test_transaction.loc[index_sell, 'Profit'] = test_transaction['Profit'].iloc[0]*(1-(transaction_amount[-1]/(test_transaction['Sell LSA'].iloc[0])))
                    test_transaction.loc[index_sell,'Sell LSA']= test_transaction['Sell LSA'].iloc[0] - transaction_amount[-1]
                    test_transaction = test_transaction.drop(index_buy) 
    pollution_rate = [a/b for a,b in zip(transaction_impact, transaction_amount)]
    return pd.DataFrame({'Transaction Amount': transaction_amount, 'Transaction Impact': transaction_impact, 'Period Buy' : periods_buy, 'Period Sell' : periods_sell, 'Clearing_T Buy' : T_clearing_buy, 'Clearing_T Buy_wobda' : T_clearing_buy_w, 'Clearing_T Sell': T_clearing_sell, 'Clearing_T Sell_wobda': T_clearing_sell_w, 'Day':day, 'Issue': Issue, 'Pollution rate' : pollution_rate})


def P_theoric_f(alpha, cl):
    plants_df = np.insert(par_temp,0,[0,0,0,'RES',0,0,0,0], axis = 0)
    P_possibilities_c = np.zeros((22,22), dtype=float)
    P_possibilities_c[:] = np.nan
    for i in range(0,len(plants_df)):
        for j in range(i+1, len(plants_df)):
            if plants_df[i,-1]+cl<(plants_df[j,-1]-cl)*alpha: # cm+cl>(cn-cl) alpha
                P_possibilities_c[i,j] = (plants_df[i,2]/alpha - plants_df[j,2])
    return(np.nanmax(P_possibilities_c))


#parameters used for the style of the graphs
linestyles = ['dashdot', 'dotted', 'dashed', (0, (3, 1, 1, 1, 1, 1)), 'solid']
colors = ['darkgray', 'dimgray', 'gray', 'black', 'silver']

linestyles_dict = {'PHS': 'dashdot', 'CAES': 'dotted', 'Battery':'dashed', 'DR': (0, (3, 1, 1, 1, 1, 1)), 'Hydrogen': 'solid'}
colors_dict = {'PHS': 'darkgray', 'CAES': 'dimgray', 'Battery':'gray', 'DR': 'black', 'Hydrogen': 'silver'}



# ==========================================================
# Graphs
# =============================================================================


# ==========================================================
# Figure 2 Merit order curve
# =============================================================================

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
    marginal_emission = np.empty(shape=(365,42))
    capacity_emission = np.empty(shape=(365,42))
    marginal_cost = np.empty(shape=(365,42))
    df_plants = pd.DataFrame(columns = range(0,42))
    df_plants_count[year] = pd.DataFrame(columns = ['Nuclear', 'Fossil Hard coal', 'Fossil Gas', 'Gas_cc'])
    if year == 2019:
        add_days = 0
    else:
        add_days = 365*2+366
    for obs in range(0+add_days,365+add_days):
        par_temp = par_temp_ori.copy()
    
            
            # Conditional calculations for the 'operating' column
        par_temp['operating cost'] = np.where(par_temp['type_g'] == 'Fossil Gas',
                                          Prices.loc[obs, 'Fossil Gas'] / par_temp['eff'],
                                          np.where(par_temp['type_g'] == 'Fossil Hard coal',
                                                   Prices.loc[obs, 'Fossil Hard coal'] / par_temp['eff'],
                                                            Prices.loc[obs, 'Nuclear']))
        
        condition = (par_temp['type_g'] == 'Fossil Gas') & (par_temp['Co2/Mwh (kg)'] < 450)
        par_temp.loc[condition, 'type_g'] = 'Gas_cc'
        par_temp['operating cost'] =  par_temp['operating cost'] +  par_temp['O & M cost']
        par_temp['emission cost']  = np.where(par_temp['type_g'] == 'Nuclear', 0, par_temp['Co2/Mwh (kg)']*Prices.loc[obs,'ETS']/1000)
        par_temp['marginal cost'] = par_temp['operating cost'] + par_temp['emission cost']
        par_temp =  par_temp.values
        par_temp = par_temp[par_temp[:,-1].argsort()] 
        marginal_emission[obs -add_days] = np.repeat(par_temp[:,2], 2)
        capacity =  np.repeat(par_temp[:,0].cumsum(), 2)   
        capacity = np.insert(capacity[0:-1], 0, 0)     
        capacity_emission[obs - add_days] = capacity
        marginal_cost[obs -add_days] = np.repeat(par_temp[:,-1], 2)
        df_plants.loc[obs - add_days] = np.repeat(par_temp[:,3], 2)
    df_marginal_emission[year] = marginal_emission.mean(axis = 0) #yearly average
    df_capacity_emission[year] = capacity_emission.mean(axis = 0)
    df_marginal_cost[year] = marginal_cost.mean(axis = 0)
  #You need to get the cost such that it represents the share of the technology at that time
# Iterate through each column and calculate value counts
    for col in df_plants.columns:
        df_plants_count[year].loc[col] = df_plants[col].value_counts()/365
    df_plants_count[year] = df_plants_count[year].fillna(0)
    marginal_cost_nuclear[year] = df_marginal_cost[year]*df_plants_count[year]['Nuclear']
    marginal_cost_gas[year] = df_marginal_cost[year]*df_plants_count[year]['Fossil Gas']
    marginal_cost_coal[year] = df_marginal_cost[year]*df_plants_count[year]['Fossil Hard coal']
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
axs[1].fill_between(list(df_capacity_emission[2019]), list(marginal_cost_coal[2019] + marginal_cost_gas_cc[2019]),  list(df_marginal_cost[2019]), facecolor='dimgrey', label = 'Gas')
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


#Compute result in a dataframe
result_sp_cap={}
for year in [2019, 2022]:
    result_sp_cap[year]={}
    for j in range(1,11):
        print(j)
        result_sp_cap[year][j]={}
        for i in  [0,1,2,3,4]:
            result_sp_cap[year][j][LSA_l[i]] = diff_agg(sol_sp_cap[year][j][LSA_l[i]])

        
#Compute the daily result     
result_sp_cap_daily={}
for year in [2019, 2022]:
    result_sp_cap_daily[year]={}
    for j in range(1,11):
        print(j)
        result_sp_cap_daily[year][j]={}
        for i in [0,1,2,3,4]:
            result_sp_cap_daily[year][j][LSA_l[i]] = diff_agg_daily(result_sp_cap[year][j][LSA_l[i]])
         

#compute the yearly result           
result_sp_cap_period_all={}
for year in [2019, 2022]:
    result_sp_cap_period_all[year]={}
    for j in range(1,11):
        print(j)
        result_sp_cap_period_all[year][j]=pd.DataFrame()
        for i in [0,1,2,3,4]:
            result_sp_cap_period_all[year][j][LSA_l[i]] = round(result_sp_cap_daily[year][j][LSA_l[i]].sum(),5)
   

#save the overall result
f = open(f'{path_file_result}result_sp_cap_period_all.pkl',"wb")
pickle.dump(result_sp_cap_period_all,f)

#and load it
with open(f'{path_file_result}result_sp_cap_period_all.pkl', 'rb') as file:
        result_sp_cap_period_all = pickle.load(file)

Emission_capacity_2019=pd.DataFrame()   
for j in range(1,11): 
    for i in [0,1,2,3,4]:
        Emission_capacity_2019.loc[j,LSA_l[i]]=result_sp_cap_period_all[2019][j][LSA_l[i]]['diff emis']/result_sp_cap_period_all[2019][j][LSA_l[i]]['Emission']*100
Emission_capacity_2019['index']=([10,20,30,40,50,60,70,80,90,100])
Emission_capacity_2019.set_index('index', inplace = True)
for i, lsa in enumerate(Emission_capacity_2019.columns):
    Emission_capacity_2019[lsa].plot(color=colors_dict[lsa], linestyle=linestyles_dict[lsa])
plt.legend()
plt.ylabel('CO2 emission impact (%)')
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
plt.ylabel('CO2 emission impact (%)')
plt.xlabel('Share of daily average demand (%)')
plt.ylim(-0.85, 0.5)


# ==========================================================
#Figure 4 :Transaction-wise rate of added emission (dependence: 'result_sp_cap' from figure 3)
# =============================================================================

           
couple_sp = {}
for year in [2019, 2022]:
    couple_sp[year] = {}
    for i in [0,1,3]:
        print(i)
        couple_sp[year][LSA_l[i]] = {}
        for cap in range(1,2):
            couple_sp[year][LSA_l[i]][cap]  = coupling(result_sp_cap[year][cap][LSA_l[i]], i)
            couple_sp[year][LSA_l[i]][cap]['Pollution rate'] = couple_sp[year][LSA_l[i]][cap]['Transaction Impact']/couple_sp[year][LSA_l[i]][cap]['Transaction Amount']  
            couple_sp[year][LSA_l[i]][cap] =  couple_sp[year][LSA_l[i]][cap][ couple_sp[year][LSA_l[i]][cap]['Transaction Amount']> 0.0001]
            couple_sp[year][LSA_l[i]][cap] =  couple_sp[year][LSA_l[i]][cap][ couple_sp[year][LSA_l[i]][cap]['Issue']==0]

f = open(f"{path_file_result}couple_sp_cap.pkl","wb")
pickle.dump(couple_sp,f)


with open(f'{path_file_result}couple_sp_cap.pkl', 'rb') as file:
        couple_sp = pickle.load(file)

#It gives you the max pollution rate per day 
P_theoric = {}
for year in [2019, 2022]:
    P_theoric[year] = pd.DataFrame(columns = [LSA_l[0], LSA_l[1], LSA_l[2], LSA_l[3], LSA_l[4]])
    if year == 2022: 
        add_days = 365*2+366
    else:  
        add_days = 0
        
    for obs in range(0+add_days,365+add_days):
        par_temp = par_temp_ori.copy()
    
            # Conditional calculations for the 'operating' column
        par_temp['operating cost'] = np.where(par_temp['type_g'] == 'Fossil Gas',
                                          Prices.loc[obs, 'Fossil Gas'] / par_temp['eff'],
                                          np.where(par_temp['type_g'] == 'Fossil Hard coal',
                                                   Prices.loc[obs, 'Fossil Hard coal'] / par_temp['eff'],
                                                            Prices.loc[obs, 'Nuclear']))
            
        par_temp['operating cost'] =  par_temp['operating cost'] +  par_temp['O & M cost']
        par_temp['emission cost']  = np.where(par_temp['type_g'] == 'Nuclear', 0, par_temp['Co2/Mwh (kg)']*Prices.loc[obs, 'ETS']/1000)    
         
        par_temp['marginal cost'] = par_temp['operating cost'] + par_temp['emission cost']
        par_temp =  par_temp.values
        par_temp = par_temp[par_temp[:,-1].argsort()] 
        P_theoric[year].loc[obs] = [P_theoric_f(par.eta_b[0,lsa]**2, par.c_bb[0,lsa]) for lsa in [0,1,2,3,4]]

#compute the maximum theoric pollution rate in the year per LSA 
P_max_theoric = pd.DataFrame(data = {2019: P_theoric[2019].max(), 2022: P_theoric[2022].max()})


#graph
#2019
theoric_list = [P_max_theoric[2019][LSA_l[lsa]]for lsa in range(5)]
list_lsa = [LSA_l[0], LSA_l[1], LSA_l[2], LSA_l[3], LSA_l[4]]
fig, ax = plt.subplots()
values = [couple_sp[2019]['PHS'][1]['Pollution rate'], couple_sp[2019]['CAES'][1]['Pollution rate'], couple_sp[2019]['Battery'][1]['Pollution rate'], couple_sp[2019]['DR'][1]['Pollution rate'], couple_sp[2019]['Hydrogen'][1]['Pollution rate']]
ax.boxplot(values, labels=list_lsa)
ax.scatter(range(1, 6), theoric_list, marker='X', label='Theoretical bound', color='red', zorder=4)
plt.ylabel('CO2 emission rate (ton/Gwh)')
plt.ylim(-900, 2600)
plt.legend()

#2022
fig, ax = plt.subplots()
theoric_list_x = [P_max_theoric[2022][LSA_l[lsa]]for lsa in range(5)]
values = [couple_sp[2022]['PHS'][5]['Pollution rate'], couple_sp[2022]['CAES'][5]['Pollution rate'], couple_sp[2022]['Battery'][5]['Pollution rate'], couple_sp[2022]['DR'][5]['Pollution rate'], couple_sp[2022]['Hydrogen'][5]['Pollution rate']]
ax.scatter(range(1, 6), theoric_list, marker='X', label='Theoretical bound', color='red', zorder=4)
ax.boxplot(values, labels=list_lsa)#,  showfliers=False)
plt.ylabel('CO2 emission rate (ton/Gwh)')
plt.ylim(-900, 2600)
plt.legend()


# ==========================================================
#Figure 5 : Box plots of theoretical bounds (dependence on figure 4: 'P_theoric')
# =============================================================================
fig, ax = plt.subplots()
values_bound_19 = [P_theoric[2019]['PHS'], P_theoric[2019]['CAES'], P_theoric[2019]['Battery'], P_theoric[2019]['DR'], P_theoric[2019]['Hydrogen']]
values_bound_22 = [P_theoric[2022]['PHS'], P_theoric[2022]['CAES'], P_theoric[2022]['Battery'], P_theoric[2022]['DR'], P_theoric[2022]['Hydrogen']]
# Define positions for the box plots
positions_19 = [x - 0.2 for x in range(1, len(LSA_l) + 1)]
positions_22 = [x + 0.2 for x in range(1, len(LSA_l) + 1)]

# Plot the box plots with different positions and colors
box_19 = ax.boxplot(values_bound_19, positions=positions_19, widths=0.4, patch_artist=True, boxprops=dict(facecolor="gainsboro"), medianprops=dict(color="black"))
box_22 = ax.boxplot(values_bound_22, positions=positions_22, widths=0.4, patch_artist=True, boxprops=dict(facecolor="dimgrey"), medianprops=dict(color="black"))

# Add legend
handles = [plt.Line2D([0], [0], color="gainsboro", lw=4, label="2019"),
           plt.Line2D([0], [0], color="dimgrey", lw=4, label="2022")]
ax.legend(handles=handles, loc='upper right')
ax.set_xticks(range(1, len(list_lsa) + 1))
ax.set_xticklabels(list_lsa)
plt.ylabel('CO2 emission rate (ton/GWh)')
plt.ylim(-900, 2600)



# ==========================================================
#Figure 6 : Transaction-wise and yearly rates of added emission
# =============================================================================

# Loading the results of the optimisation for different carbon levy
tax = {}
tax[2019] ={}
tax[2022] ={}
for lsa in [0,1,2,3,4]:
    tax[2019][LSA_l[lsa]] =  ['ETS'] + list(np.round(np.arange(0.00, 0.041, 0.002), 3)) +list(np.round(np.arange(0.045, 0.091, 0.005), 3))+list(np.round(np.arange(0.1, 0.27, 0.01), 2)) +  [0.28] + list(np.round(np.arange(0.3, 1.31, 0.1), 2)) + list(np.round(np.arange(1.8, 4, 0.5), 2))
    if lsa in [1,2]:
         tax [2022][LSA_l[lsa]]  =  ['ETS'] + list(np.round(np.arange(0.00, 0.88, 0.01), 2)) + list(np.round(np.arange(0.9, 3.1, 0.1), 2)) +  [3.5,4,4.5,5,5.5,6, 10,15,20,25]
    elif lsa == 0:
        tax [2022][LSA_l[lsa]]  =  ['ETS'] + list(np.round(np.arange(0.00, 0.88, 0.01), 2)) + list(np.round(np.arange(0.9, 3.1, 0.1), 2)) +  [3.5,4,4.5,5,5.5,6, 10,15,20,25, 30,35, 37]      
    elif lsa == 3:
         tax [2022][LSA_l[lsa]]  =  ['ETS'] + list(np.round(np.arange(0.00, 0.88, 0.01), 2)) + list(np.round(np.arange(0.9, 3.1, 0.1), 2)) + [3.5,4,4.5,5,5.5,6]
    else:
         tax [2022][LSA_l[lsa]]  =  ['ETS'] + list(np.round(np.arange(0.00, 0.88, 0.01), 2)) + list(np.round(np.arange(0.9, 3.1, 0.1), 2)) 



sol_sp_tax ={}
for year in [2019,2022]:
    if year == 2022: 
       add_days = 365*2+366
    else:  
        add_days = 0
    sol_sp_tax[year] = {}
    for cap in [1]:
        sol_sp_tax[year][cap] = {}
        for lsa in [0,1,2,3,4]:
            sol_sp_tax[year][cap][LSA_l[lsa]]={} 
            for t in tax[year][LSA_l[lsa]]:
                sol_sp_tax[year][cap][LSA_l[lsa]][t] =open_file(optimisation = 'daily_value', lsa = lsa, tax = t, year = year, mult_lsa = cap)


#Some corrections:
sol_sp_tax[2019][1]['CAES'][0] = sol_sp_tax[2019][1]['CAES'][0][0]
sol_sp_tax[2019][1]['DR'][0] = sol_sp_tax[2019][1]['DR'][0][0]
sol_sp_tax[2019][1]['PHS'][0] = sol_sp_tax[2019][1]['PHS'][0][0]      


#Overal impact
result_sp_tax={} 
result_sp_tax_daily={}
result_sp_tax_period_all={}
for year in [2022]:
    if year == 2022: 
       add_days = 365*2+366
    else:  
        add_days = 0
    result_sp_tax[year]={}
    result_sp_tax_daily[year]={}
    result_sp_tax_period_all[year]={}
    for cap in [1]:
        result_sp_tax[year][cap]={}
        result_sp_tax_daily[year][cap]={}
        result_sp_tax_period_all[year][cap]={}      
        for i in [0, 1,2]:
            result_sp_tax[year][cap][LSA_l[i]]={}
            result_sp_tax_daily[year][cap][LSA_l[i]]={}
            result_sp_tax_period_all[year][cap][LSA_l[i]]=pd.DataFrame()
            for ets in   tax[year][LSA_l[i]]:
                result_sp_tax[year][cap][LSA_l[i]][ets] = diff_agg(sol_sp_tax[year][cap][LSA_l[i]][ets])
                result_sp_tax_daily[year][cap][LSA_l[i]][ets] = diff_agg_daily(result_sp_tax[year][cap][LSA_l[i]][ets])
                result_sp_tax_period_all[year][cap][LSA_l[i]] = pd.concat([result_sp_tax_period_all[year][cap][LSA_l[i]],pd.DataFrame({ets: round(result_sp_tax_daily[year][cap][LSA_l[i]][ets].sum(),5)})], axis = 1)

f = open(f"{path_file_result}result_sp_tax_period_all.pkl","wb")
pickle.dump(result_sp_tax_period_all,f)

with open(f'{path_file_result}result_sp_tax_period_all.pkl', 'rb') as file:
        result_sp_tax_period_all = pickle.load(file)


result_tax_no_lsa = {}
result_tax_no_lsa_daily = {}
result_tax_no_lsa_period_all = {}
for year in [2019,2022]:
    if year == 2022: 
       add_days = 365*2+366
    else:  
        add_days = 0
    result_tax_no_lsa[year]={}
    result_tax_no_lsa_daily[year]={}
    result_tax_no_lsa_period_all[year]={}
    for cap in [1]:
        result_tax_no_lsa[year][cap]={} 
        result_tax_no_lsa_daily[year][cap]={}
        result_tax_no_lsa_period_all[year][cap]= pd.DataFrame()
        lsa = 0 
        for ets in  tax[year][LSA_l[lsa]]:
              result_tax_no_lsa[year][cap][ets]= diff_nolsa(sol_sp_tax[year][cap][LSA_l[lsa]][ets])
              result_tax_no_lsa_daily[year][cap][ets] =diff_agg_daily(result_tax_no_lsa[year][cap][ets])
              result_tax_no_lsa_period_all[year][cap] =  pd.concat([result_tax_no_lsa_period_all[year][cap], pd.DataFrame({ets: round(result_tax_no_lsa_daily[year][cap][ets].sum(),5)})], axis = 1)
        
f = open(f"{path_file_result}result_no_lsa_tax_period_all.pkl","wb")
pickle.dump(result_tax_no_lsa_period_all,f)

with open(f'{path_file_result}result_no_lsa_tax_period_all.pkl', 'rb') as file:
        result_tax_no_lsa_period_all = pickle.load(file)


        
Emission = {}
for year  in [2019, 2022]: 
    Emission[year] = {}
    for cap in [1]:
        Emission[year][cap]= pd.DataFrame()
        for i in [0, 1,2,3,4]:
            Emission[year][cap]=pd.merge(Emission[year][cap], pd.DataFrame({LSA_l[i]: result_sp_tax_period_all[year][cap][LSA_l[i]].loc['Emission'].values}, index = result_sp_tax_period_all[year][cap][LSA_l[i]].columns), left_index=True, right_index=True, how='outer')
        
 
Emission_no_lsa = {}
for year  in [2019, 2022]: 
    Emission_no_lsa[year] = {}
    for cap in [1]:
        Emission_no_lsa[year][cap]= pd.DataFrame()
        for i in [0, 1,2,3,4]:
            Emission_no_lsa[year][cap]=pd.merge(Emission_no_lsa[year][cap], pd.DataFrame({LSA_l[i]: result_tax_no_lsa_period_all[year][cap].loc['Emission'].values}, index = result_tax_no_lsa_period_all[year][cap].columns), left_index=True, right_index=True, how='outer')
        #Emission[year][cap] = Emission[year][cap].fillna(method='ffill')     

Amount = {}
for year  in [2019, 2022]: 
    Amount[year] = {}
    for cap in [1]:
        Amount[year][cap]= pd.DataFrame()
        for i in [0, 1,2,3,4]:
            Amount[year][cap]=pd.merge(Amount[year][cap], pd.DataFrame({LSA_l[i]: result_sp_tax_period_all[year][cap][LSA_l[i]].loc['Sell LSA'].values}, index = result_sp_tax_period_all[year][cap][LSA_l[i]].columns), left_index=True, right_index=True, how='outer')


Diff_emis = {}
for year  in [2019, 2022]: 
    Diff_emis[year] = {}
    for cap in [1]:
        Diff_emis[year][cap]= pd.DataFrame()
        for i in [0, 1,2,3,4]:
            Diff_emis[year][cap] = Emission[year][cap].sub(Emission_no_lsa[year][cap], axis = 0)

Overal_rate = {}
for year  in [2019, 2022]: 
    Overal_rate[year] = {}
    for cap in [1]:
        Overal_rate[year][cap]= pd.DataFrame()
        for i in [0, 1,2,3,4]:
            Overal_rate[year][cap] = Diff_emis[year][cap].div(Amount[year][cap], axis = 0)


Overal_rate[2019][1].drop('ETS', inplace = True)   
Overal_rate[2019][1].index  = np.float64(Overal_rate[2019][1].index)  *1000   

Overal_rate[2022][1].drop('ETS', inplace = True)   
Overal_rate[2022][1].index  = np.float64(Overal_rate[2022][1].index)  *1000   




## Impact per transaction ##
couple_sp_tax = {}
for year in [2022]:
    print(year)
    #couple_sp_tax[year] = {}
    for i in [3,4]:
        print(i)
        #couple_sp_tax[year][LSA_l[i]] = {}
        for t in tax[year][LSA_l[i]][0:32] : 
                couple_sp_tax[year][LSA_l[i]][t]  = coupling(result_sp_tax[year][1][LSA_l[i]][t], i)
                couple_sp_tax[year][LSA_l[i]][t] =  couple_sp_tax[year][LSA_l[i]][t][couple_sp_tax[year][LSA_l[i]][t]['Transaction Amount']> 0.0001]
                couple_sp_tax[year][LSA_l[i]][t] =  couple_sp_tax[year][LSA_l[i]][t][couple_sp_tax[year][LSA_l[i]][t]['Issue']==0]

        
f = open(f"{path_file_result}couple_sp_tax.pkl","wb")
pickle.dump(couple_sp_tax,f)

with open(f'{path_file_result}couple_sp_tax.pkl', 'rb') as file:
        couple_sp_tax = pickle.load(file)
        


#transaction impact    
tax_tr = {}
tax_tr[2019] =  [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08 ,0.09, 0.3, 1.3]
tax_tr [2022]  =   [0.0, 0.05, 0.1, 0.3,0.4 ,0.6, 0.8, 1.0, 1.2,3]  
  


P_theoric_tax={} 
for year in [2019, 2022]:
    if year == 2022: 
        add_days = 365*2+366
    else:  
        add_days = 0
   
    P_theoric_tax[year]={}
    for t in  tax_tr[year]:
        P_theoric_tax[year][tax]=pd.DataFrame(columns = range(5))
        for obs in range(0+add_days,365+add_days):
            par_temp = par_temp_ori.copy()
            
                    
                    # Conditional calculations for the 'operating' column
            par_temp['operating cost'] = np.where(par_temp['type_g'] == 'Fossil Gas',
                                                  Prices.loc[obs, 'Fossil Gas'] / par_temp['eff'],
                                                  np.where(par_temp['type_g'] == 'Fossil Hard coal',
                                                           Prices.loc[obs, 'Fossil Hard coal'] / par_temp['eff'],
                                                                    Prices.loc[obs, 'Nuclear']))
                    
            par_temp['operating cost'] =  par_temp['operating cost'] +  par_temp['O & M cost']
            tax_value = [tax if tax!= 0 else 0.00001]
            par_temp['emission cost']  = np.where(par_temp['type_g'] == 'Nuclear', 0, par_temp['Co2/Mwh (kg)']*t)    
                 
            par_temp['marginal cost'] = par_temp['operating cost'] + par_temp['emission cost']
            par_temp =  par_temp.values
            par_temp = par_temp[par_temp[:,-1].argsort()] 
            P_theoric_tax[year][t].loc[obs] = [P_theoric_f(par.eta_b[0,lsa]**2, par.c_bb[0,lsa]) for lsa in [0,1,2,3,4]]



P_max_theoric_tax ={}
P_max_theoric_tax[2019] = pd.DataFrame(data = [P_theoric_tax[2019][tax].max() for tax in tax_tr[2019]])
P_max_theoric_tax[2022] = pd.DataFrame(data = [P_theoric_tax[2022][tax].max() for tax in tax_tr[2022]])
P_max_theoric_tax[2019].columns = LSA_l
P_max_theoric_tax[2019].index = [x*1000 for x in tax_tr[2019]]
P_max_theoric_tax[2022].columns = LSA_l
P_max_theoric_tax[2022].index = [x*1000 for x in tax_tr[2022]]

position_ets = {}
position_ets[2019] = 13
position_ets[2022] = 8
tax_graph = {}
tax_graph[2019] = [0, 20,40, 90,300,1300]
tax_graph[2022] = [0, 100, 400,800, 1000,3000]
tax_overal = {}
tax_overal[2019] = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 110.0, 130.0, 150.0, 170.0, 190.0, 220.0, 240.0, 260.0, 280.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300]
tax_overal[2022] = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 130.0, 160.0, 190.0, 220.0, 250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 440.0, 480.0, 520.0, 560.0, 600.0, 640.0, 680.0, 720.0, 760.0, 800.0, 820.0, 840.0, 860.0, 870.0, 900.0, 900.0, 900.0, 900.0, 1000.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0, 2200.0, 2400.0, 2600.0, 2800.0, 3000]
Overal_rate_l = {}
values = {}
for year in [2019, 2022]:
    Overal_rate_l[year]={}
    values[year] ={}
    for lsa in range(5):
        Overal_rate_l[year][LSA_l[lsa]] = [Overal_rate[year][1].loc[i,LSA_l[lsa]] for i in tax_overal[year]]
        values[year][LSA_l[lsa]] = [couple_sp_tax[year][LSA_l[lsa]][i/1000]['Pollution rate'] for i in tax_graph[year]]
       

for year in [2019]:         
    for lsa in range(5):
        fig, ax = plt.subplots()
        ax.scatter([0,10,20,30,40,50], P_max_theoric_tax[year][LSA_l[lsa]][tax_graph[year]],  marker = 'x', label = 'Theoretical bound', color = 'red', zorder=4)
        ax.plot(range(51), Overal_rate_l[year][LSA_l[lsa]], label = 'Yearly rate of added emission', color = 'black')
        ax.boxplot(values[year][LSA_l[lsa]], labels=tax_graph[year],  showfliers=True, positions = [0,10,20,30,40,50], widths = 3)
        ax.axvline(x=position_ets[year], color='red', linestyle='-', label = 'ETS Mean')
        plt.ylabel('Co2 emission rate (Ton/Gwh)')
        plt.xlabel('Carbon levy level (€/ton)')
        #plt.ylim(-700, 2650)
        #plt.xlim(-10, 320)
        if year == 2019:
            plt.legend()


# ==========================================================
#Figure 7 : Minimal carbon levy for different maximal allowed rates of added emission
# =============================================================================

#compute the theoretical tax
rows = []
Co2_Tax = {}
Ax_tax_indices = {}
for lsa in [0,1,2,3,4]:
    Co2_tax[LSA_l[lsa]] = pd.DataFrame()
    Ax_tax_indices[LSA_l[lsa]] = pd.DataFrame()
#find couple to look at that would need an higher tax

for year in [2022]:
    if year == 2022: 
       add_days = 365*2+366
    else:  
        add_days = 0
    for obs in range(0+add_days,365+add_days):
        par_temp = par_temp_ori.copy()
    
            
            # Conditional calculations for the 'operating' column
        par_temp['operating cost'] = np.where(par_temp['type_g'] == 'Fossil Gas',
                                          Prices.loc[obs, 'Fossil Gas'] / par_temp['eff'],
                                          np.where(par_temp['type_g'] == 'Fossil Hard coal',
                                                   Prices.loc[obs, 'Fossil Hard coal'] / par_temp['eff'],
                                                            Prices.loc[obs, 'Nuclear']))
            
        par_temp['operating cost'] =  par_temp['operating cost'] +  par_temp['O & M cost']
        par_temp =  par_temp.values
        par_temp = par_temp[par_temp[:,-1].argsort()] 
        for lsa in [4]:
            alpha = par.eta_b[0,lsa]**2
            Tax_max = []
            index_max = []
            for P in range(0,150):
                aa = pd.DataFrame()
                rows.clear()
                threshold = P
                indices = np.argwhere((par_temp[:, 2, None] / alpha - par_temp[:, 2] > threshold))
                #indices = indices[indices[:, 0] < indices[:, 1]]
                for idx in indices:
                   i, j = idx[0], idx[1]
                   tax = (alpha * par_temp[j,-1]-par_temp[i,-1] - par.c_bb[0,lsa]*(1+alpha))/(par_temp[i,2]-alpha*par_temp[j,2])
                   rows.append(tax)
                aa['Tax'] = rows
                aa['indices'] = list(indices)
                aa = aa.sort_values(by='Tax', ascending=False)

                if len(rows) != 0:
                    Tax_max.append(max(0,max(rows)) )
                    index_max.append(list(indices[np.where(rows == max(rows))][0]))
                else:
                    Tax_max.append(0)
            Co2[LSA_l[lsa]][obs] =  Tax_max
            Ax_tax_indices[LSA_l[lsa]][obs] = index_max


f = open(f"{path_file_result}Ax_tax_P_LSA.pkl","wb")
pickle.dump(Co2_tax,f) 

with open(f'{path_file_result}Ax_tax_P_LSA.pkl', 'rb') as file:
    Co2_tax =  pickle.load(file)
    

#compute average
Ax_mean = {}
for year in [2019, 2022]:
    if year == 2022: 
       add_days = 365*2+366
    else:  
        add_days = 0
    Ax_mean[year] = pd.DataFrame(columns = range(0,900))
    for key,df in Ax_tax.items():
        Ax_mean[year].loc[key] = Ax_tax[key].loc[0+add_days:364+add_days].mean(axis = 0)*1000
 
        
#graph
for year in [2019, 2022]:
    plt.figure()
    for i, column in enumerate(Ax_mean[year].T.columns):
        Ax_mean[year].T[column].plot(color=colors[i], linestyle=linestyles[i])
    plt.plot(range(9000), [ets[year]]*9000, linestyle='solid', label='ETS mean' , color='red')
    plt.xlabel('Maximum allowed rate of added emission (ton/Gwh)')
    plt.ylabel('Average carbon levy (€/ton)')
    plt.legend()
    if year == 2019:
        plt.ylim(0,500)
        plt.xlim(0,800)
    else:
        plt.ylim(0,5000)
        plt.xlim(0,800)
    plt.show() 




# ==========================================================
#Table 5 and 6 : Maximum ration phi and 90th quantile of ratio phi, dependence Co2_tax from figure 7 and couple_sp_tax from figure 6
# =============================================================================

Max_Pollution_rate = {}
for year in [2019, 2022]:
    Max_Pollution_rate[year]  = {}
    for i in [0,1,2,3,4]:
        Max_Pollution_rate[year][LSA_l[i]] = pd.DataFrame(index = range(365))
        for t in tax[year][LSA_l[i]]:
            Max_Pollution_rate[year][LSA_l[i]] = pd.concat([Max_Pollution_rate[year][LSA_l[i]], pd.DataFrame({t: couple_sp_tax[year][LSA_l[i]][t].groupby('Day')['Pollution rate'].max()})], axis = 1)    
for year in [2019, 2022]:
    for i in [0,1,2,3,4]:
        Max_Pollution_rate[year][LSA_l[i]].fillna(0, inplace=True)
        Max_Pollution_rate[year][LSA_l[i]] = Max_Pollution_rate[year][LSA_l[i]].applymap(lambda x: 0 if round(x, 2) == 0 else x)




def find_index(series ,P ):
    # Find the last occurrence of a positive value
    last_positive_index = -1
    for i in range(len(series)):
        if series.iloc[i] > P:
            last_positive_index = i
    # Check if all values after last_positive_index are 0
    if last_positive_index != -1 and all(value <= P for value in series[last_positive_index+1:]):
            # Find the first 0 after last_positive_index
            first_zero_index = Max_Pollution_rate[year][LSA_l[lsa]].loc[d].index[last_positive_index+1]
            return first_zero_index
    else:
        return 0


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
            Th_su[year][LSA_l[lsa]][P]['Avg_price'] = Co2_tax[LSA_l[lsa]].loc[0+add_days:365+add_days,P]
            Th_su[year][LSA_l[lsa]][P]['tax achieving this target']  = [find_index(Max_Pollution_rate[year][LSA_l[lsa]].loc[d], P) for d in range(0,365)]
            Th_su[year][LSA_l[lsa]][P]['ratio'] = Th_su[year][LSA_l[lsa]][P]['tax achieving this target']/Th_su[year][LSA_l[lsa]][P]['Avg_price'] 
            Th_su[year][LSA_l[lsa]][P]['ratio'] = Th_su[year][LSA_l[lsa]][P]['ratio'].clip(upper=1)
            Th_su[year][LSA_l[lsa]][P] = Th_su[year][LSA_l[lsa]][P].dropna()
              

#the two different tables
Th_su_qt = {}
for q in [90,100]:
    Th_su_qt[q]= {}
    for P in [0,100,500]:
        Th_su_qt[q][P]= pd.DataFrame(index = [2019,2022], columns = LSA_l)
        for lsa in [0,1,2,3,4]:
            if lsa != 4:
                Th_su_qt[q][P][LSA_l[lsa]] = [np.percentile(Th_su[year][LSA_l[lsa]][P]['ratio'], q) for year in [2019,2022]]
            else:
                Th_su_qt[q][P][LSA_l[lsa]].loc['2022'] = [np.percentile(Th_su[year][LSA_l[lsa]][P]['ratio'], q) for year in [2022]]

       
       



# ==========================================================
# Figure 8 : LSA changes of profit compared to the case without any carbon levy, dependence on 'couple_sp_tax from figure 6'
# =============================================================================

change = {}
change[2019] = pd.DataFrame()
for i in [0,1,2,3,4]:
    change[2019][LSA_l[i]] = (result_sp_tax_period_all[2019][1][LSA_l[i]].loc['Profit']/result_sp_tax_period_all[2019][1][LSA_l[i]].loc['Profit', 0] -1)*100
change[2019].drop('ETS',  inplace = True)
change[2019] = change[2019].sort_index()
change[2019].index = np.float64(change[2019].index)*1000

change[2022] = pd.DataFrame()
for i in [0,1,2,3,4]:
    change[2022][LSA_l[i]] = (result_sp_tax_period_all[2022][1][LSA_l[i]].loc['Profit']/result_sp_tax_period_all[2022][1][LSA_l[i]].loc['Profit', 0] -1)*100
change[2022].drop('ETS',  inplace = True)
change[2022] = change[2022].sort_index()
change[2022].index = np.float64(change[2022].index)*1000

for i, column in enumerate(change[2022].columns):
    change[2022][column].plot(color=colors[i], linestyle=linestyles[i], legend = True)
plt.xlabel ('Carbon levy level (€/ton)')
plt.xlim(0,500)
plt.ylim(-20, 80)
plt.ylabel( 'Profit change (%)')
plt.axvline(x=ets[2022], color='red', linestyle='-', label='ETS mean')
plt.legend()
plt.show()

for i, column in enumerate(change[2019].columns):
    change[2019][column].plot(color=colors[i], linestyle=linestyles[i], legend = True)
plt.xlabel ('CO2 Tax (€/ton)')
plt.xlim(0,90)
plt.ylim(0, 600)
plt.ylabel( 'Profit change (%)')
plt.axvline(x=ets[2019], color='red', linestyle='-', label='ETS mean')
plt.legend()
plt.show()


