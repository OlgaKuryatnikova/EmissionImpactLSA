# -*- coding: utf-8 -*-
"""

This is the file to generate optimisation results: It takes files from source, then there is a function that generate the optimisation for the full year with the parameters chosen 
To compile this file, the folder needs to be set like explain in readme plus you need a licence of mosek
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import time
import pickle
import pandas as pd
from picos import Problem, RealVariable, BinaryVariable, SymmetricVariable
import picos
import numpy as np
import os


#To complete:
userpath = "C:\\Users\\"


path_file_source= os.path.join(userpath,"Data\\submission\\source\\")
path_file_optimisation_save = os.path.join(userpath,"Data\\result_optimisation\\new_file\\") #this is a folder for new files, then needs to be copy in the right folder, I do that to not overwrite some files
class Object(object):
    pass
[
# define a utility to save an object
]

def save_object(obj, filename):
    with open(f'{path_file_optimisation_save}'+filename, 'ab') as outp:
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



par_temp_ori = {}
par_temp_ori[2019] = pd.read_csv(f"{path_file_source}Parameters_plants_2019.csv", sep=',', index_col = 0, usecols=[2,3,4,7, 8,9,12,14,15,16, 17, 18, 19,20])
par_temp_ori[2022] = pd.read_csv(f"{path_file_source}Parameters_plants_2022.csv", sep=',', index_col = 0, usecols=[2,3,4,7, 8,9,12,14,15,16,17,18, 19, 20])
# load, min load, r-up, r-down, emis_avg, emis_max, emis_min, c_min, c_max, c_avg
Prices = pd.read_csv(f"{path_file_source}Prices.csv", sep=',', index_col=0)



'''Set up main prameters'''
# All further parameters are given per LSA type, there are 10 LSA types  in the order as below
LSA_l = ['Hydrogen', 'PHS', 'CAES', 'Battery', 'Perfect ESS']

par = Object() # This is to store all the elements that are stable through the days
par.T = 24  # number of periods per day
par.gamma = 1-np.matrix([0, 0, 0, 0, 0]) #self- discharge, assume to be 0
# share of the bought energy that is effectively chanrged
par.eta_b = np.sqrt(
    np.matrix([0.375, 0.76, 0.795, 0.9, 1])
)  
# share of the generated energy that reaches the customer when discharged
par.eta_s = par.eta_b #assumed to be the same
par.c_bs = np.matrix([0.5, 0.11, 1.55, 1.25, 0.01]) # LSA energy sell cost
par.c_bb = par.c_bs  # LSA energy buy cost assume to be the same as sell
par.rate = np.matrix([1, 0.12, 0.19, 0.57, 1]) #how much energy (as percent of the capacity) can be charged or discharge in an hour
par.pmin = 0 # price of the renewable

# parameters defining additional problem complexities



'''Parameters of the optimisation, are part of the function, here just to run cases without function '''
start_day = 0 #in the year which day start the optimization
num_days = 365 #for how mnay days to optimize
lsa =3 # which lsa participate, nb is equal to the position in LSA_l, provide a list if several LSAs are involved
year = 2019
LSA_Capacity = 'dynamic' #the LSA capacity is set as 10% percentage of the daily average demand, can also be set to a number
#LSA_Capacity = 0.0
mult_lsa = 1 #multiplier of the lsa capacity
mult_d = 1.0 #multiplier of the demand
mult_s = 1.0 #multiplier of the sun generation
mult_w = 1.0 #multiplier of the wind generation
co2_tax = 'ETS' #what co2 tax to use, for transaction tax, can also be set to 'ETS' or to any number
Pollution_rate = 0 #what is the maximum pollution rate desired to apply the theoric co2 tax
ramp = False #ramping constraint on generators
min_load =False #minimum load on generators
start_cost = False #start up cost of generators 
n_p = 1 #number of marginal cost pieces of generators
fix_lsa = False #if we want to have fix operations in the LSA that are already there, then D_b and D_s need a vector input
D_b = np.zeros((par.T,1)) #if you want to run the optimization with fix buying operation of the storage agent
D_s = np.zeros((par.T,1)) #If you want to run the optimization with fix selling operation of the storage agent
par.n_lsa = 1# the number of storage agent
''' the function that runs over all the day of the year'''

def optimisation_sp(lsa,  year=2019, mult_lsa=1, num_days=365, LSA_Capacity='dynamic',
                     start_day=0, co2_tax='ETS', mult_d = 1.0, mult_s = 1.0, mult_w = 1.0, Pollution_rate = 0, 
                     ramp = False, min_load = False, start_cost = False, n_p = 1, D_b = D_b, D_s = D_s,  fix_lsa = False, save = True):
    par.n_f = len(par_temp_ori[year])+1  # number of fossil fuel operators,  including renewables
    par.n_tot = par.n_f + 1 #number of generators, including renewables
    efficiency = par.eta_b[0, lsa]
    m_cost = par.c_bs[0, lsa]
    dc_rate = par.rate[0, lsa]
    if year == 2022:
        add_days = 365*2+366
    else:
        add_days = 0
    value = Object()    
    if type(LSA_Capacity) == str:#capacity based ob demand
        LSA_cap = np.sum(
            D_full[add_days*24:(add_days+365)*24])/(365*24)*0.1
        Capacity = np.tile(mult_lsa*LSA_cap/par.n_lsa, (1,1))  # GwH, divide the total max capacity equally among all storage


    if type(LSA_Capacity) == float: #given capacity
        Capacity = np.tile(mult_lsa*LSA_Capacity/par.n_lsa, (1,n_p))  # GwH, divide the total max capacity equally among all LSAs      
    
    Capacity = np.matrix(Capacity)    
    par.S_max = np.tile(0.9*Capacity, (par.T+1,1))  # max SOC
    par.S_min = np.tile(0.1*Capacity, (par.T+1,1))  # min SOC
    par.S_init = Capacity/2  # half loaded batteries at the beginning, this one makes more sense, but until now I have used the second one. 

    nuclear_price = Prices.loc[0+add_days, 'Nuclear']  
    for obs in range(start_day + add_days, start_day + num_days + add_days): 
        day_year = obs - add_days
        value = Object() #that is the object with the parameters that are changing daily
        par_temp = par_temp_ori[year].values
        prices_day = Prices.loc[obs].values
        coal_price = prices_day[0]
        gas_price = prices_day[1] 
        co2_price = prices_day[3]
        is_nuclear = par_temp[:, 1] == "Nuclear"
        is_coal = par_temp[:, 1] == "Coal"
        mc = np.where(
           is_nuclear, nuclear_price, 
           np.where(is_coal, coal_price / par_temp[:, 5], gas_price / par_temp[:, 5]))
        mc += par_temp[:, 11]
        co2_tax_value = (co2_price / 1000 if co2_tax == 'ETS' else 0.00001 if co2_tax == 0 else co2_tax) #the price we have is in eur/ton, and we have obj in 1000 eur.  
        em_c = np.where(is_nuclear, 0, par_temp[:, 6] * co2_tax_value)
        par_temp = np.concatenate((par_temp, np.reshape(mc+em_c, (len(mc),1))), axis = 1)
        if n_p == 3: #if three pieces oer generators
             par_temp = np.concatenate((par_temp, np.reshape(par_temp[:, -1]/1.1, (len(par_temp),1))), axis = 1)
             par_temp = np.concatenate((par_temp, np.reshape(par_temp[:, -2]/0.9, (len(par_temp),1))), axis = 1)
       
        if n_p == 1: 
            par_temp = par_temp[par_temp[:, -1].argsort()]
            
            
            eps = 0.001
            for i in range(1, len(par_temp)):
                while par_temp[i, -1] - par_temp[i - 1, -1] < eps: #this is done to avoid python to consider cost equal
                    par_temp[i, -1] += eps     
            par.c_f = np.matrix(par_temp[:, -1]).T# costs ordered by value, you need to change -1 in -3 when you will put the code for 3 pieces
           
        else:
            flattened = par_temp[:,-3:].flatten()
            original_indices = np.argsort(flattened)  # Store original order

            # Step 2: Sort the flattened array
            sorted_indices = np.argsort(np.argsort(flattened))
            sorted_flattened = np.sort(flattened)
    
            eps = 0.001
            for i in range(1, len(sorted_flattened )):
                while sorted_flattened [i] - sorted_flattened [i -1] < eps:
                    sorted_flattened [i] += eps    

            # Step 4: Restore original order
            restored_flattened = np.zeros_like(sorted_flattened)
            restored_flattened[original_indices] = sorted_flattened  # Undo sorting
            
            # Step 5: Reshape back into 3 columns
            reshaped = restored_flattened.reshape(-1, 3)
            par_temp[:,-3:] = reshaped
            par_temp = par_temp[par_temp[:, -3].argsort()]        
            par.c_f = np.concatenate((np.matrix(par_temp[:, -2]).T, np.matrix(par_temp[:, -3]).T, np.matrix(par_temp[:, -1]).T), axis=1) # costs ordered by value

        par.c_f = np.insert(par.c_f, 0, 0, axis=0) # add renewables
        par.quant = np.matrix(np.round(par_temp[:, 0].astype(float)/1000, 5))  # GwH
        par.quant = np.tile(par.quant, (par.T, 1))
        par.emis = np.matrix(np.insert(par_temp[:, 6], 0, 0)) # Ton CO2 per GwH, or kg CO2 per MwH, # add renewables
        par.emis = np.tile(par.emis, (par.T, 1))

        if n_p == 3:
            par.emis = np.tile(par_temp[:, [10, 6, 9]].T, (24, 1))
            par.emis[:, 0] = 0  # Set the first column to [0, 0, 0]
            par.emis = np.hstack((np.zeros((par.emis.shape[0], 1)), par.emis))

        ind_t = range(24*obs, 24*obs + par.T)  # range of hours
        Wind = np.reshape((Q_r_full['Wind MWH'][ind_t]*mult_w).values, (24, 1))
        Solar = np.reshape(
            (Q_r_full['Solar MWH'][ind_t]*mult_s).values, (24, 1))
        
        value.quant = np.insert(par.quant, 0, (Wind + Solar).T, axis=1)  # GwH, add renewables
        value.cum_quant = np.cumsum(value.quant, axis=1)

        # compute the demand
        D = D_full[ind_t]*mult_d
        D_r = D - 0.447  # I remove a fix baseload of waste generation in the NL
        if ramp: #operation to remove when demand is larger than the total cumulated capacity of the generators
            x = (D_r > value.cum_quant[:, -1] -2.6)
            y = np.where(x==True)[0]
            D_r[y] = value.cum_quant[y, -1] -2.6 # I remove a bit more otherwise with ramp constraint, it creates an infeasibility (ex: 2019, no lsa, day 9 0.15, 17 1.3)
        else:
            x = (D_r > value.cum_quant[:, -1]   )
            y = np.where(x==True)[0]
            D_r[y] = value.cum_quant[y, -1]  

        if (mult_lsa == 0) and (ramp == False) and (min_load == False) and (start_cost == False) and (n_p ==1): #simple case without battery 
           sol_sp = Object()
           sol_sp.var = Object()
           sol_sp.par_temp = par_temp
           sol_sp.var.q_f_tot = np.multiply(D_r - (Wind +Solar) > 0 > 0, D_r )
           sol_sp.D = D_r # demand
           sol_sp.Q_r = Wind+Solar  # renewable generation
           sol_sp.days = obs
           bb_wo_lsa =np.round(   sol_sp.var.q_f_tot ,4) > np.round(value.cum_quant,4)
           curt = D_r < Wind+Solar
           num_pc_wobda = np.sum(bb_wo_lsa,1)
           sol_sp.var.num_pc = num_pc_wobda
           q_f_last_wobda = np.where((num_pc_wobda.T == 0), 0,  sol_sp.var.q_f_tot.T - value.cum_quant[range(par.T), num_pc_wobda-1])
           q_f_temp = value.quant + 0  # to prevent them from being the same object
           q_f_temp[~bb_wo_lsa] = 0
           emis_temp = par.emis + 0
           emis_temp[~bb_wo_lsa] = 0
           sol_sp.var.emis_hourly = np.multiply(q_f_last_wobda, par.emis[range(
                        par.T), num_pc_wobda]).T + np.sum(np.multiply(q_f_temp, emis_temp), 1)
           cols = num_pc_wobda.flatten()
           q_f_temp[np.arange(24), cols] = q_f_last_wobda.flatten()
           q_f_temp[:,0][curt] = D_r[curt]
           sol_sp.var.q_f = q_f_temp     
        else: #when there is a battery 
            '''Define optimization variables'''
            P = picos.Problem()  # defining a problem P
            q_f = [RealVariable("q_f[" + str(i) + "]", (par.n_f,n_p), lower=0) for i in range(par.T)]   
            s = RealVariable("s", (par.T+1, par.n_lsa), lower=par.S_min,upper=par.S_max)  # state of charge variables
            q_b = RealVariable("q_b", (par.T, par.n_lsa), lower=0, upper=np.tile(np.multiply(Capacity,dc_rate), (par.T,1)))
            w_b = RealVariable("w_b", (par.T, par.n_lsa), lower=0, upper=np.tile(np.multiply(Capacity,dc_rate), (par.T,1)))
            '''Add objective and constraints'''
               # minimize the total cost
            obj = (picos.sum(q_f[t][j,k]*par.c_f[j,k] for k in range(n_p) for j in range(par.n_f) for t in range(par.T)) + picos.sum(q_b[t]*m_cost+w_b[t]*m_cost for t in range(par.T)))  # objective
            P.set_objective("min", obj)
    
            '''Constraints about fixing the operation of the LSA'''
            if fix_lsa:
                P += [w_b[t,j]==np.floor(D_b[t]*1e4)/1e4 for t in range(par.T) for j in range(par.n_lsa)]
                P += [q_b[t,j]==np.floor(D_s[t]*1e4)/1e4 for t in range(par.T) for j in range(par.n_lsa)]

            else:                 
                # Initial SOC
                P += [s[0,j] == par.S_init[0,j] for j in range(par.n_lsa)]
    
                # Final SOC
                P += [s[-1,j] == par.S_init[0,j] for j in range(par.n_lsa)]
                # Flow conservation constraints  for LSAs
                P += [s[t+1,j] == s[t,j]+efficiency*w_b[t,j]-q_b[t,j]/efficiency
                      for t in range(par.T) for j in range(par.n_lsa)]
    
            if not(min_load or start_cost or ramp):
                pass
            else:
            # min load and start-up constraints
                if start_cost and min_load:
                    ep = RealVariable("ep", (par.T,par.n_f), lower=0) # epigraph variable, defines the final cost ep=c(q)= c1*q1+c2*q2+c3*q3
                    obj = (picos.sum(ep[t,j] for j in range(par.n_f) for t in range(par.T))
                               + picos.sum(q_b[t,0]*m_cost+w_b[t,0]*m_cost for t in range(par.T)))  # objective
                    P.set_objective("min", obj)
                    par.c_s = np.matrix(np.insert(par_temp[:, 12], 0, 0)) 
                    par.q_min = np.multiply(par_temp[:, 0],par_temp[:, 2])
                    par.q_min=np.matrix(
                        np.round(np.insert(par.q_min, 0, 0).astype(float)/1000, 5))  # min load, assume the first entry is renewables
                    par.q_min=np.tile(par.q_min,(par.T,1))
                    # min load
                    b_f = BinaryVariable("b_f", (par.T,par.n_f)) # 1 if generator j is on in period t, 0 otherwise
                    con_min_load1 = P.add_list_of_constraints([picos.sum(q_f[t][j,k] for k in range(n_p)) <= value.quant[t,j]*b_f[t,j] 
                                                         for j in range(par.n_f) for t in range(par.T)])
                    con_min_load2 = P.add_list_of_constraints([picos.sum(q_f[t][j,k] for k in range(n_p)) >= par.q_min[t,j]*b_f[t,j] 
                                                           for j in range(par.n_f) for t in range(par.T)])
                
                    # start up
                    con_start_up1 = P.add_list_of_constraints([ep[t,j] >= par.c_s[0,j]*(b_f[t,j]-b_f[t-1,j]) + picos.sum(q_f[t][j,k]*par.c_f[j,k] for k in range(n_p)) 
                          for j in range(par.n_f) for t in range(1,par.T)])
                    con_start_up2 = P.add_list_of_constraints([ep[0,j] >= par.c_s[0,j]*b_f[0,j] + picos.sum(q_f[0][j,k]*par.c_f[j,k] for k in range(n_p)) 
                          for j in range(par.n_f)])
                    P += [ep[t,j] >= picos.sum(q_f[t][j,k]*par.c_f[j,k] for k in range(n_p)) for j in range(par.n_f) for t in range(par.T)]
                
                # min load only constraints
                if min_load and not start_cost:
                    par.q_min = np.multiply(par_temp[:, 0],par_temp[:, 2])
                    par.q_min=np.matrix(
                        np.round(np.insert(par.q_min, 0, 0).astype(float)/1000, 5))           
                    par.q_min=np.tile(par.q_min,(par.T,1))
                 
                    b_f = BinaryVariable("b_f", (par.T,par.n_f)) # 1 if generator j is on in period t, 0 otherwise
                    con_min_load1 = P.add_list_of_constraints([picos.sum(q_f[t][j,k] for k in range(n_p)) <= value.quant[t,j]*b_f[t,j] 
                                                         for j in range(par.n_f) for t in range(par.T)])
                    con_min_load2 = P.add_list_of_constraints([picos.sum(q_f[t][j,k] for k in range(n_p)) >= par.q_min[t,j]*b_f[t,j] 
                                                           for j in range(par.n_f) for t in range(par.T)])
                            
                # start-up only constraints
                if start_cost and not min_load:
                    ep = RealVariable("ep", (par.T,par.n_f), lower=0) # epigraph variable, defines the final cost ep=c(q)= c1*q1+c2*q2+c3*q3
                    obj = (picos.sum(ep[t,j] for j in range(par.n_f) for t in range(par.T))
                               + picos.sum(q_b[t,0]*m_cost+w_b[t,0]*m_cost for t in range(par.T)))  # objective
                    P.set_objective("min", obj)
                    par.c_s = np.matrix(np.insert(par_temp[:, 12], 0, 0)) 
                    # min load
                    b_f = BinaryVariable("b_f", (par.T,par.n_f)) # 1 if generator j is on in period t, 0 otherwise       
                    # start up
                    con_start_up1 = P.add_list_of_constraints([ep[t,j] >= par.c_s[0,j]*(b_f[t,j]-b_f[t-1,j]) + picos.sum(q_f[t][j,k]*par.c_f[j,k] for k in range(n_p)) 
                          for j in range(par.n_f) for t in range(1,par.T)])
                    con_start_up2 = P.add_list_of_constraints([ep[0,j] >= par.c_s[0,j]*b_f[0,j] + picos.sum(q_f[0][j,k]*par.c_f[j,k] for k in range(n_p)) 
                          for j in range(par.n_f)])
                    P += [ep[t,j] >= picos.sum(q_f[t][j,k]*par.c_f[j,k] for k in range(n_p)) for j in range(par.n_f) for t in range(par.T)]
                
                # ramp up and down constraints if they are present
                # assume that in the zero period anything can be generated within the given limits
                if ramp:
                    par.ramp_up=np.matrix(np.insert(par_temp[:, 3], 0, 1).astype(float)) # assume the first entry is renewables
                    par.ramp_down=np.matrix(np.insert(par_temp[:, 4], 0, -1).astype(float))     # assume the first entry is renewables  
                    P += [picos.sum(q_f[t][j,k] for k in range(n_p)) >= picos.sum(q_f[t-1][j,k] for k in range(n_p))*(1+par.ramp_down[0,j]*60) 
                          for j in range(par.n_f) for t in range(1,par.T)]
                    P += [picos.sum(q_f[t][j,k] for k in range(n_p)) <= picos.sum(q_f[t-1][j,k] for k in range(n_p))*(1+par.ramp_up[0,j]*60) 
                          for j in range(par.n_f) for t in range(1,par.T)]
            
                   
            # constraints on q_f, could be changed if we adjust the length of the intervals
            # assume here that each piece in the cost function is of the same length
            P += [q_f[t][j,k] <= value.quant[t,j]/n_p for k in range(n_p) for j in range(par.n_f) for t in range(par.T)]
    
           
            # # Demand satisfaction constraints
            # without the complicating constraints we could have an inequality >=,
            # but with those constraints we need an equality for each period, otherwise we will have to curtail
            con_equil = P.add_list_of_constraints([picos.sum(q_f[t][j,k] for k in range(n_p) for j in range(par.n_f)) 
                                             + picos.sum(q_b[t,j] - w_b[t,j] for j in range(par.n_lsa)) == D_r[t] for t in range(par.T)])
    
            '''Solve'''
            P.solve(solver='mosek')
            '''Extract optimization results'''
            sol_sp = Object()
            sol_sp.status = P.status
            sol_sp.obj = obj.value
            sol_sp.var = Object()
            sol_sp.var.s = np.matrix(s.value)  # state of charge
            sol_sp.var.q_b = np.matrix(q_b.value)  # LSA sold
            sol_sp.var.w_b = np.matrix(w_b.value)  # LSA bought
            sol_sp.var.q_f = np.concatenate([np.matrix(q_f[t].value).T for t in range(par.T)])
            sol_sp.var.q_f_tot = np.sum(sol_sp.var.q_f, 1)
            sol_sp.D = D_r # demand
            sol_sp.Q_r = Wind+Solar  # renewable generation
            sol_sp.days = obs
            if min_load or start_cost:
                sol_sp.var.b_f = np.matrix(b_f.value)  # binary: generator on or off
            sol_sp.par_temp = par_temp   
            sol_sp.var.emis_hourly = np.sum(np.multiply(sol_sp.var.q_f, par.emis), 1)  # emission per hour in ton
    
    
            # find the prices
            if not(min_load or start_cost or ramp) and n_p ==1:
                sol_sp.var.price_dual =  [con_equil[t].dual for t in range(par.T)] 
                q_f_tot_pos = np.matrix(np.multiply(sol_sp.var.q_f_tot>0,sol_sp.var.q_f_tot))
                bl_buy = np.round(q_f_tot_pos,4) > np.round(value.cum_quant,4)
                bl_sell = np.round(q_f_tot_pos,4) >= np.round(value.cum_quant,4)
                ind_buy = np.round(sol_sp.var.w_b,4)>0 
                sol_sp.var.bl = (bl_buy & ind_buy) | (bl_sell & ~ind_buy) #
                num_pc = np.sum(sol_sp.var.bl,1)
                #sol_sp.var.num_pc = num_pc
                modified_num_pc = [x if x!= len(par.c_f) else x - 1 for x in num_pc]
                sol_sp.var.price = par.c_f[modified_num_pc]
                curt = D_r < Wind+Solar
                #try to add this part
                q_f_tot_wobda = np.multiply(D_r - (Wind +Solar) > 0, D_r)
                #q_f_tot_wobda = np.multiply(D_r  > 0, D_r )
                bb_wo_lsa =np.round( q_f_tot_wobda,4) > np.round(value.cum_quant,4)
                num_pc_wobda = np.sum(bb_wo_lsa,1)
                sol_sp.var.num_pc_wobda = num_pc_wobda
               
                q_f_last_wobda = np.where((num_pc_wobda.T == 0), 0, q_f_tot_wobda.T - value.cum_quant[range(par.T), num_pc_wobda-1])
                
                q_f_temp = value.quant + 0  # to prevent them from being the same object
                q_f_temp[~bb_wo_lsa] = 0
                q_f_temp[:,0][curt] = D_r[curt]
                emis_temp = par.emis + 0
                emis_temp[~bb_wo_lsa] = 0
                sol_sp.var.emis_hourly_wobda = np.multiply(q_f_last_wobda, par.emis[range(
                         par.T), num_pc_wobda]).T + np.sum(np.multiply(q_f_temp, emis_temp), 1)
                
                cols = num_pc_wobda.flatten()
                q_f_temp[np.arange(24), cols] = q_f_last_wobda.flatten()
                sol_sp.var.q_f_wobda = q_f_temp 
                
            elif n_p == 3:
                sol_sp.var.price_dual =  [con_equil[t].dual for t in range(par.T)] 
            else:
                if min_load and start_cost:
                    P.remove_constraint((0,))
                    P.remove_constraint((0,))
                    P.remove_constraint((0,))
                    P.remove_constraint((0,))
                    bb_f = RealVariable("bb_f", (par.T,par.n_f)) # 1 if generator j is on in period t, 0 otherwise
                    P += [bb_f == sol_sp.var.b_f]
                    # min load
                    con_min_load1 = P.add_list_of_constraints([picos.sum(q_f[t][j,k] for k in range(n_p)) <= value.quant[t,j]*bb_f[t,j] 
                                                         for j in range(par.n_f) for t in range(par.T)])
                    con_min_load2 = P.add_list_of_constraints([picos.sum(q_f[t][j,k] for k in range(n_p)) >= par.q_min[t,j]*bb_f[t,j] 
                                                           for j in range(par.n_f) for t in range(par.T)])
                
                    # start up
                    con_start_up1 = P.add_list_of_constraints([ep[t,j] >= par.c_s[0,j]*(bb_f[t,j]-bb_f[t-1,j]) + picos.sum(q_f[t][j,k]*par.c_f[j,k] for k in range(n_p)) 
                          for j in range(par.n_f) for t in range(1,par.T)])
                    con_start_up2 = P.add_list_of_constraints([ep[0,j] >= par.c_s[0,j]*bb_f[0,j] + picos.sum(q_f[0][j,k]*par.c_f[j,k] for k in range(n_p)) 
                          for j in range(par.n_f)])
                if start_cost and not min_load:
                    P.remove_constraint((0,))
                    P.remove_constraint((0,))
                    bb_f = RealVariable("bb_f", (par.T,par.n_f)) # 1 if generator j is on in period t, 0 otherwise
                    P += [bb_f == sol_sp.var.b_f]
                    # start up
                    con_start_up1 = P.add_list_of_constraints([ep[t,j] >= par.c_s[0,j]*(bb_f[t,j]-bb_f[t-1,j]) + picos.sum(q_f[t][j,k]*par.c_f[j,k] for k in range(n_p)) 
                          for j in range(par.n_f) for t in range(1,par.T)])
                    con_start_up2 = P.add_list_of_constraints([ep[0,j] >= par.c_s[0,j]*bb_f[0,j] + picos.sum(q_f[0][j,k]*par.c_f[j,k] for k in range(n_p)) 
                          for j in range(par.n_f)])
                if min_load and not start_cost:
                    P.remove_constraint((0,))
                    P.remove_constraint((0,))
                    bb_f = RealVariable("bb_f", (par.T,par.n_f)) # 1 if generator j is on in period t, 0 otherwise
                    P += [bb_f == sol_sp.var.b_f]
                    # min load
                    con_min_load1 = P.add_list_of_constraints([picos.sum(q_f[t][j,k] for k in range(n_p)) <= value.quant[t,j]*bb_f[t,j] 
                                                         for j in range(par.n_f) for t in range(par.T)])
                    con_min_load2 = P.add_list_of_constraints([picos.sum(q_f[t][j,k] for k in range(n_p)) >= par.q_min[t,j]*bb_f[t,j] 
                                                           for j in range(par.n_f) for t in range(par.T)])                   
                     #Resolve the linearized version and obtain prices from the respective dual
                P.solve(solver='mosek')
                # dual prices 
                sol_sp.var.price_dual = pr_dual = [con_equil[t].dual for t in range(par.T)] 

        if save:
            save_object(sol_sp, 'sol_sp_cap_'+','.join(str(mult_lsa).split('.'))+'_'+'_'.join(LSA_l[lsa].split(' '))+'_'+str(year)+'_tax_'+','.join(str(co2_tax).split('.'))+
                    '_solar_'+','.join(str(mult_s).split('.'))+'_wind_'+','.join(str(mult_w).split('.'))+
                    '_ramp_'+str(ramp)+'_min_load_'+str(min_load)+'_start_c_'+str(start_cost)+'_n_piece_'+str(n_p)+'_'+str(start_day + add_days)+'_'+str(add_days + start_day + num_days))
        else:
          return(sol_sp)  




                
''' to run the function, the parameters needs to be adjusted as desired and it will save it in the folder 'new_file', it write for each day in the file '''

#Below are all the files that are needed to do the graphs, they are stored in new file and need to be transfered into the correct folder afterwards
for year in [2019,2022]:
    for cap in range(1,11):
        for lsa in [0,1,2,3,4]:
            optimisation_sp(lsa=lsa, year=year, n_p=1, mult_lsa=cap)
            optimisation_sp(lsa=lsa, year=year, n_p=1, mult_lsa=cap, min_load = True)
            optimisation_sp(lsa=lsa, year=year, n_p=1, mult_lsa=cap, ramp = True)
    optimisation_sp(lsa = 3, mult_lsa = 0, year = year)
    optimisation_sp(lsa = 3, mult_lsa = 0, year = year, min_load = True)
    optimisation_sp(lsa = 3, mult_lsa = 0, year = year, ramp = True)
    optimisation_sp(lsa = 3, mult_lsa = 0, year = year, start_cost=True)
    optimisation_sp(lsa = 3, mult_lsa = 0, year = year, n_p = 3)

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


RES_value = {}
RES_value[2019] = np.concatenate([np.arange(1, 5.1, 0.5),np.arange(6,10.1,1)])
RES_value[2022] = np.arange(1,5.1,0.5)

for year in [2019,2022]:
        for lsa in [0,1,2,3,4]:
            optimisation_sp(lsa=lsa, year=year, n_p=1, start_c = True)
            optimisation_sp(lsa=lsa, year=year, n_p=3)
            for t in tax[year][LSA_l][lsa]:
                optimisation_sp(lsa=lsa, year=year, n_p=1, co2_tax = t)
            
        for t in tax[year]['No Lsa']:
            optimisation_sp(lsa=3, year=year, n_p=1, co2_tax = t, mult_lsa = 0)
        for res in RES_value[year][1:]:
            optimisation_sp(lsa=3, year=year, n_p=1, mult_w = res)
            optimisation_sp(lsa=3, year=year, n_p=1, mult_s = res)
            optimisation_sp(lsa=3, year=year, n_p=1, mult_s = res, mult_w = res)




