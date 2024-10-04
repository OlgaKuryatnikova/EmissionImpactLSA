# -*- coding: utf-8 -*-
"""
2022
A model for the behavior of a day-ahead market: social planner
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import timeit
import pickle
import pandas as pd
from picos import Problem, RealVariable, BinaryVariable, SymmetricVariable
import picos
import numpy as np


path_file_source= "path\\source\\"
# the folder source is in the same github package
path_file_optimisation = "path_to_save_optimisation_output"

class Object(object):
    pass
[
# define a utility to save an object
]

def save_object(obj, filename):
    with open(f'{path_file_optimisation}'+filename, 'ab') as outp:
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


'''
Load data
'''

D_full = pd.read_csv(f"{path_file_source}Load_2019_2022.csv", skipinitialspace=True, usecols=[2])
D_full = D_full.values/1000  # GwH

Q_r_full = pd.read_csv(f"{path_file_source}Renewable_2019_2022.csv", skipinitialspace=True, usecols=[1, 2])
Q_r_full = Q_r_full/1000


'''Replace nans with approximations '''
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

par_temp_ori = pd.read_csv(f"{path_file_source}Parameters_plants_NL_2.csv", sep=';', usecols=[1,7,8,10,12])
Prices = pd.read_csv(f"{path_file_source}Prices.csv", sep=',', index_col=0)

'''Set up main prameters'''
# All further parameters are given per LSA type, there are 10 LSA types  in the order as below
LSA_l = ['PHS', 'CAES', 'Battery', 'DR', 'Hydrogen']
par = Object() # This is to store all the elements that are stable through the days
par.T = 24  # number of periods per day
par.gamma = 1-np.matrix([0, 0, 0, 0, 0]) #self- discharge, assume to be 0
# share of the bought energy that is effectively chanrged
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



'''Parameters of the optimisation, are part of the function'''
start_day = 0
num_days = 365
lsa = 1 #which lsa participate, nb is equal to the position in LSA_l
year = 2019
LSA_Capacity = 'dynamic' #the LSA capacity is set as 10% percentage of the daily average demand, can also be set to a number
mult_lsa = 1#multiplier of the lsa capacity
mult_d = 1 #multiplier of the demand
mult_s = 1 #multiplier of the sun generation
mult_w = 1 #multiplier of the wind generation
co2_tax = 3.8 #what co2 tax to use, for transaction tax, A1 is alpha 1, can also be set to 'ETS' or to any number
Pollution_rate = 0 #what is the maximum pollution rate desired to apply the co2 tax alpha 1

def optimisation_sp(lsa,  year=2019, mult_lsa=1, num_days=365, LSA_Capacity='dynamic',
                     start_day=0, co2_tax='ETS', mult_d = 1, mult_s = 1.0, mult_w = 1.0, Pollution_rate = 0):

    self_discharge = par.gamma[0, lsa]
    efficiency = par.eta_b[0, lsa]
    m_cost = par.c_bs[0, lsa]
    dc_rate = par.rate[0, lsa]

    if year == 2022:
        add_days = 365*2+366
    else:
        add_days = 0

    value = Object()
    

    if type(LSA_Capacity) == str:
        LSA_capacity = np.sum(
            D_full[add_days*24:(add_days+365)*24])/(365*24)*0.1
        Capacity = mult_lsa*LSA_capacity  # GwH
        par.S_max = 0.9*Capacity  # max SOC
        par.S_min = 0.1*Capacity  # min SOC
        par.S_init = par.S_max/2  # half loaded batteries at the beginning
# if we just have an input LSA_capacity = number, we can use it, otherwise LSA capacity is computed based on depamd
# will produce an error if there are nans, that's good

    if type(LSA_Capacity) == float:
        Capacity = mult_lsa*LSA_Capacity  # GwH
        par.S_max = 0.9*Capacity  # max SOC
        par.S_min = 0.1*Capacity  # min SOC
        par.S_init = par.S_max/2

    for obs in range(start_day + add_days, start_day + num_days + add_days):
        value = Object() #that are the parameters that are changing daily
        day_year = obs - add_days
        par_temp = par_temp_ori.copy()
        # Conditional calculations for the 'operating' column
        par_temp['operating cost'] = np.where(par_temp['type_g'] == 'Fossil Gas',
                                          Prices.loc[obs, 'Fossil Gas'] /
                                              par_temp['eff'],
                                          np.where(par_temp['type_g'] == 'Fossil Hard coal',
                                                   Prices.loc[obs, 'Fossil Hard coal'] /
                                                       par_temp['eff'],
                                                            Prices.loc[obs, 'Nuclear']))

        par_temp['operating cost'] = par_temp['operating cost'] + \
            par_temp['O & M cost']

        if co2_tax == 'ETS':
            co2_tax_value = Prices.loc[obs, co2_tax]/1000
        else:
            co2_tax_value = co2_tax
        if co2_tax_value == 0:
            par_temp['emission cost'] = np.where(
                par_temp['type_g'] == 'Nuclear', 0, par_temp['Co2/Mwh (kg)']*0.00001)
        else:
            par_temp['emission cost'] = np.where(
                par_temp['type_g'] == 'Nuclear', 0, par_temp['Co2/Mwh (kg)']*co2_tax_value)

        par_temp['marginal cost'] = par_temp['operating cost'] + \
            par_temp['emission cost']

        par_temp = par_temp.values
        par_temp = par_temp[par_temp[:, -1].argsort()]

        for i in range(len(par_temp)-1):
            if round(par_temp[i, -1], 4) == round(par_temp[i+1, -1], 4):
                par_temp[i+1, -1] += 0.0001

        # after ordering by marginal cost
        # Maximal available quantities, without the battery, in the same order as costs
        value.quant = np.matrix(
            np.round(np.insert(par_temp[:, 0], 0, 0).astype(float)/1000, 4))  # GwH
        value.quant = np.tile(value.quant, (par.T, 1))
        value.cum_quant = np.cumsum(value.quant, axis=1)
        value.emis = np.matrix(np.insert(par_temp[:, 2], 0, 0)) # Ton CO2 per GwH, or kg CO2 per MwH
        value.emis = np.tile(value.emis, (par.T, 1))
        value.c_f = np.matrix(par_temp[:, -1])
        value.c_f = np.tile(value.c_f, (par.T, 1))
        value.price = np.concatenate((par.pmin*np.ones((par.T, 1)), value.c_f), 1)

        ind_t = range(24*obs, 24*obs + par.T)  # range of hours
        D = D_full[ind_t]*mult_d
        Wind = np.reshape((Q_r_full['Wind MWH'][ind_t]*mult_w).values, (24, 1))
        Solar = np.reshape(
            (Q_r_full['Solar MWH'][ind_t]*mult_s).values, (24, 1))
        D_r = D - Wind - Solar - 0.447  # Residual Demand
        D_r[D_r > value.cum_quant[0, -1]] = 16.771

        # Generate 24 random values between -16.7 and 16.772
        #D_r = np.array([random.uniform(-16.7, 16.772) for _ in range(24)], ndmin = 2).T
        '''Define optimization variables'''
        P = picos.Problem()  # defining a problem P
        q_f = RealVariable("q_f", (par.T, par.n_f), lower=0,
                           upper=value.quant[range(par.T), 1:])

        s = RealVariable("s", (par.T+1), lower=par.S_min,
                         upper=par.S_max)  # state of charge variables
        # contracts can start in all but the last periods
        q_b = RealVariable("q_b", (par.T), lower=0, upper=Capacity*dc_rate)
        w_b = RealVariable("w_b", (par.T), lower=0, upper=Capacity*dc_rate)

        '''Add objective and constraints'''
        # Objective
        # if emission_obj: # minimize the total emission, comes from fossil fuel only
   
           # minimize the total cost
        obj = (picos.sum(q_f[t, j]*value.c_f[t, j] for j in range(par.n_f) for t in range(par.T))
                   + picos.sum(q_b[t]*m_cost+w_b[t]*m_cost for t in range(par.T)))  # objective

        # obj=obj/1000000
        P.set_objective("min", obj)

        # Ramp up and down constraints if they are present
        # assume that in the zero period anything can be generated within the given limits
        # if ramp:
            # P += [q_f[t,j] >= q_f[t-1,j]*(1+value.ramp_down[t,j]*60) for j in range(value.n_f) for t in range(1,par.T)]
            # P += [q_f[t,j] <= q_f[t-1,j]*(1+value.ramp_up[t,j]*60) for j in range(value.n_f) for t in range(1,par.T)]

        # Initial SOC
        P += [s[0] == par.S_init]

        # Final SOC
        P += [s[-1] == par.S_init*((self_discharge)**par.T)]

        # Flow conservation constraints  for BDAs
        P += [s[t+1] == self_discharge*s[t]+efficiency*w_b[t]-q_b[t]/efficiency
              for t in range(par.T)]

            # Demand satisfaction constraints
        # # min generation for each period
        P += [picos.sum(q_f[t, j] for j in range(par.n_f)) + picos.sum(q_b[t] - w_b[t])
              >= D_r[t] for t in range(par.T)]
            # # # max generation for each period
            # P += [Q_r[t] + picos.sum(q_f[t,j] for j in range(value.n_f)) +
                                      # picos.sum(cum_q[t][j] - cum_w[t][j] for j in range(value.n_b)) <=  D[t] + value.delta*Q_r[t]
                                      # for t in range(par.T) ]

        '''Solve'''
        P.solve(solver='mosek') # any other LP solver could be used

        '''Extract optimization results'''
        sol_sp = Object()
        sol_sp.status = P.status
        sol_sp.obj = obj.value
        sol_sp.var = Object()
        sol_sp.var.s = np.matrix(s.value)  # state of charge
        sol_sp.var.q_f = np.matrix(q_f.value)  # fossil fuel generation
        sol_sp.var.q_b = np.matrix(q_b.value)  # LSA sold
        sol_sp.var.w_b = np.matrix(w_b.value)  # LSA bought
        sol_sp.var.q_f_tot = np.sum(sol_sp.var.q_f, 1)
        q_f_tot_wobda = np.multiply(D_r > 0, D_r)
        sol_sp.var.q_f_tot_wobda = q_f_tot_wobda
        #sol_sp.D = D  # demand
        #sol_sp.Q_r = Wind+Solar  # renewable generation
        sol_sp.D_r = D_r  # residual demand, D-Qr
        sol_sp.days = obs
        
        '''Price '''
        # Notice that bl is equal to zero if the interval is covered exactly. The explanation is as follows:
        # if some interval can be exactly covered by fossil fuel, the battery would produce just 
        # little enough to make sure the price is set up by the next interval, and if the battery buys, 
        # it will also buy just little enough to make sure that the price stays lower, optimization provides the limiting cases
        # In the case of the social planner we repeat this pattern to avoid purely technical differences in the profit of the battery
        '''New part: I adjusted computation of some indices to make it simpler and cleaner'''  
        q_f_tot_pos = np.matrix(np.multiply(sol_sp.var.q_f_tot>0,sol_sp.var.q_f_tot))
        bl_buy = np.round(q_f_tot_pos,4) > np.round(value.cum_quant,4)
        bl_sell = np.round(q_f_tot_pos,4) >= np.round(value.cum_quant,4)
        ind_buy = np.round(sol_sp.var.w_b,4)>0 
        sol_sp.var.bl = (bl_buy & ind_buy) | (bl_sell & ~ind_buy) #
        num_pc = np.sum(sol_sp.var.bl,1)
        sol_sp.var.num_pc = num_pc
        sol_sp.var.price = value.price[range(par.T),num_pc.T]
        bb_wo_lsa =np.round( q_f_tot_wobda,4) > np.round(value.cum_quant,4)
        num_pc_wobda = np.sum(bb_wo_lsa,1)
        sol_sp.var.num_pc_wobda = num_pc_wobda
        sol_sp.var.price_wobda = value.price[range(par.T),num_pc_wobda.T]
        '''Emission '''
        sol_sp.par_temp = par_temp      
        list_clearing_wobda = [i-1 if i !=0 else 0 for i in sol_sp.var.num_pc_wobda] #since nuclear is zero, we can also attribute zero to RES
        sol_sp.var.e_wobda = par_temp[list_clearing_wobda, 2]
        list_clearing = [i-1 if i !=0 else 0 for i in sol_sp.var.num_pc]
        sol_sp.var.e = par_temp[list_clearing, 2]        
        sol_sp.obj_wobda = np.sum(sol_sp.var.q_f.T*sol_sp.var.price_wobda.T)
        sol_sp.var.emis_hourly = np.sum(np.multiply(sol_sp.var.q_f, value.emis[:, range(1, par.n_tot)]), 1)  # emission per hour in ton
        q_f_last_wobda = np.where((num_pc_wobda.T == 0), 0, q_f_tot_wobda.T - value.cum_quant[range(par.T), num_pc_wobda-1])
        q_f_temp = value.quant + 0  # to prevent them from being the same object
        q_f_temp[~bb_wo_lsa] = 0
        emis_temp = value.emis + 0
        emis_temp[~bb_wo_lsa] = 0
        sol_sp.var.emis_hourly_wobda = np.multiply(q_f_last_wobda, value.emis[range(
            par.T), num_pc_wobda]).T + np.sum(np.multiply(q_f_temp, emis_temp), 1)  # emission per hour in KG without LSA
        #sol_sp.obj_wobda = np.sum(sol_sp.var.q_f.T*sol_sp.var.price_wobda.T)
      
        # print(obs)
        if LSA_Capacity == 0:
            save_object(sol_sp, 'sol_sp_cap_0_'+str(year)+'_tax_'+','.join(str(co2_tax).split(
                '.'))+'_'+str(start_day + add_days)+'_'+str(add_days + start_day + num_days))
        elif type(LSA_Capacity) == pd.core.series.Series:
            save_object(sol_sp, 'sol_sp_cap_'+str(mult_lsa)+'_eff_'+','.join(str(round(efficiency, 2)).split('.'))+'_mcost_'+','.join(str(round(
                m_cost, 2)).split('.'))+'_dc_rate_'+','.join(str(round(dc_rate, 2)).split('.'))+'_'+str(year)+'_cap_'+str(LSA_Capacity.name)+'_'+str(start_day + add_days)+'_'+str(add_days + start_day + num_days))
        else:
           save_object(sol_sp, 'sol_sp_cap_'+str(mult_lsa)+'_eff_'+','.join(str(round(efficiency, 2)).split('.'))+'_mcost_'+','.join(str(round(m_cost, 2)
                        ).split('.'))+'_dc_rate_'+','.join(str(round(dc_rate, 2)).split('.'))+'_'+str(year)+'_tax_'+','.join(str(co2_tax).split('.'))+'_'+str(start_day + add_days)+'_'+str(add_days + start_day + num_days))
    
