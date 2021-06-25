#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:14:25 2020

@author: hujie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize,LinearConstraint
import datetime
import numba as nb
import multiprocessing as mp
#from probability import loss
from numba import jit

#===================== read file =========================================
date = '3_2'
#read components file, the result of max posterior model from BayesMixQt 
norm_data = pd.read_excel('map_'+date+'.xlsx',sheet_name = None)
#read age data
file_name = 'basin_use_op_2021_2_25.xlsx'
name_use = file_name[:-5]
age_all = pd.read_excel(file_name,sheet_name = None)

#make file to save results 
sqp_writer = pd.ExcelWriter(date+'_sqp_random_'+name_use+'.xlsx') # all values

#===================== read file end  =========================================

#=====================  functions  =========================================

# Gauss model equation 3
@jit(nopython=True)
def gaus(x,mu,sig):
    return 1/sig/(2*np.pi)**0.5*np.exp(-(x-mu)**2/2/sig**2)

# resample age function  equation 4
def produce_random_age(age,error):
    random_age = []
    for i in range(len(age)):
        random_age_temp = np.random.normal(age[i],age_error[i],1)
        random_age.append(random_age_temp)
    random_age = np.array(random_age)
    return random_age

#probability function equation 5
#note: to speed up the program, we use 2*∂x*pdf(x) to replace the cdf(x+∂x)-cdf(x-∂x), the result is nearly same.
@jit(nopython=True)
def probability_function(W,rand_age,norm_para_all_in_one,node_number_id,delta = 2):
    P = np.array([np.float64(x) for x in range(0)])
    for x in range(len(rand_age)):
        P_river = np.array([np.float64(x) for x in range(0)])
        for j in range(len(node_number_id)-1):
            test_mu = norm_para_all_in_one[0][node_number_id[j]:node_number_id[j+1]]
            test_sig = norm_para_all_in_one[1][node_number_id[j]:node_number_id[j+1]]
            test_pi = norm_para_all_in_one[2][node_number_id[j]:node_number_id[j+1]]
            p_1 = gaus(rand_age[x],test_mu,test_sig)*test_pi*delta*2
            p_1 = np.sum(p_1)
            P_river = np.append(P_river,p_1) #the probability of one point in each river
        haha = np.sum(P_river*W) #probability of each river multiply the weight of each river for one point
        P = np.append(P,haha) 
    P_all = np.sum(-np.log(P)) #防止溢出
    return P_all

# use parallel computing to speedup the process of the calculation
def do_mpi(ran_age,norm_para):
    nw = len(norm_para)
    #把所有的参数合成一个2D数组3*8，id 为分隔点
    node_number = []
    for i in norm_para:
        node_number.append(len(i[0]))
    node_number_id = np.cumsum(np.array(node_number))
    node_number_id = np.insert(node_number_id,0,0)
    norm_para_all_in_one = np.concatenate(norm_para,axis=1)

    W0 = np.linspace(1/nw,1/nw,nw)  # initial vaules
    #开始优化过程
    #up and bottom bounds
    lb = 0.0
    ub = 1.0
    bounds = [(lb,ub)]*nw        
    #set constrain
    def constraint(x):
    # the sum of all weights of opened (more the 0) positions should equal 1
        # also add '-1' to execute maximization 
        return (1-sum(x))
    
    con = ({'type':'eq','fun':constraint})
    res = minimize(probability_function,W0,(ran_age,norm_para_all_in_one,node_number_id),\
                    method = 'SLSQP',bounds = bounds,\
                    constraints=con,options={'disp': False})        
    return list(np.append(res.x,res.fun))  

#=====================  functions end  =========================================

# ===================== main functions  =========================================

if __name__ == '__main__':
    for river_used in list(age_all.keys()):
    #    age_all.keys():
    #for river_used in age_model_again:
        
        age_data = age_all[river_used]
        print('working on',river_used)
     
        #numba doesn't support pandas，change to np.array
        age = np.array(age_data['Age'])
        age_error = np.array(age_data['Error'])
        norm_para = []
        river_names = np.sort(list(norm_data.keys())) #sort the keys
        for i in river_names:
            Mu = np.array(norm_data[i]['Mu'])
            Sigma = np.array(norm_data[i]['Sigma'])
            Pi = np.array(norm_data[i]['Pi'])
            temp_para = [Mu,Sigma,Pi]
            norm_para.append(temp_para)
        
        rd_age_list = []
        for i in range(1000):
            #produce 1000 random age 
            rand_age = produce_random_age(age,age_error)
            # rand_age = age 
            rd_age_list.append(rand_age)
            
        start = datetime.datetime.now() 
        pool = mp.Pool(mp.cpu_count())
        result_objects = [pool.apply_async(do_mpi,args=(ran_age,norm_para)) for ran_age in rd_age_list]
        results = [r.get() for r in result_objects]
        end = datetime.datetime.now()
        print(end-start)

        pool.close()
        pool.join()
        #save all results
        columns = np.append(river_names,'Func')
        all_propotion = pd.DataFrame(results,columns=columns)    
        all_propotion.to_excel(sqp_writer,sheet_name = river_used,index = False)
    sqp_writer.save()

    print('Finish')


