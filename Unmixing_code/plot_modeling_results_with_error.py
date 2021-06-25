#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:15:51 2020

@author: hujie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import copy

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'

#===================== read file =========================================
date = '3_2'
file_name = 'basin_use_op_2021_2_25.xlsx'
name_use = file_name[:-5]
basin_data = pd.read_excel(file_name,sheet_name = None)

norm_data = pd.read_excel('map_'+date+'.xlsx',sheet_name = None)
sqp_name = date+'_sqp_random_'+name_use+'.xlsx'
Wi_sqp = pd.read_excel(sqp_name,sheet_name = None)
bayes_MAP = pd.read_excel('basin_map_3_9.xlsx',sheet_name = None)

#===================== read file end =========================================

#===================== functions  =========================================
def river_function(x,W,norm_para):
    #不能进行ufunc
    #
    P_1 = [] #一条河概率，权值之前的
    river_names = np.sort(list(norm_data.keys())) #排序防止出现错位
    for i in river_names:
        p_temp = (stats.norm.pdf(x,norm_para[i]['Mu'],norm_para[i]['Sigma'])*norm_para[i]['Pi']).sum()
        P_1.append(p_temp)
    return np.sum(np.array(P_1)*np.array(W))  #W为河流的权值，返回值x在最终拟合曲线的p

def combined_river_function(x,norm_para):
    #不能进行ufunc
    p_1 = stats.norm.pdf(x,norm_para['Mu'],norm_para['Sigma'])*norm_para['Pi']
    return p_1.sum()

#计算每个点在河流的概率
def probability_function(age,norm_para,delta = 2):
    P = []
    for x in range(len(age)):
        P_age= np.array([])
        for j in range(len(norm_para)):
            test = norm_para[j]
#            print(test)
            p_1 = ((stats.norm.cdf(age[x]+delta,test[0],test[1])-stats.norm.cdf(age[x]-delta,test[0],test[1]))*test[2]).sum()
            P_age = np.append(P_age,p_1) #每个点在十四条河的概率
        P.append(P_age)
    return P

norm_para = []
river_names = np.sort(list(norm_data.keys())) #对keys进行排序，防止后面对不上
for i in river_names:
    Mu = np.array(norm_data[i]['Mu'])
    Sigma = np.array(norm_data[i]['Sigma'])
    Pi = np.array(norm_data[i]['Pi'])
    temp_para = [Mu,Sigma,Pi]
    norm_para.append(temp_para)

#===================== functions end  =========================================


#===================== plots  =========================================
#plot histogram
figtt, ax_hah = plt.subplots(len(Wi_sqp), 8,figsize=(16,3.5*len(Wi_sqp)),tight_layout=True)
temp_river_list = ['Dong River','Bei River','He River','Gui River','Liu River','Upper Hongshui River','You River','Zuo River']
for pt in range(len(Wi_sqp.keys())):
    pt_temp = Wi_sqp[list(Wi_sqp.keys())[pt]]
    for jjk in range(len(temp_river_list)):
        ax_hah[pt,jjk].hist(pt_temp[temp_river_list[jjk]])
        if pt == 0:
            ax_hah[pt,jjk].set_title(temp_river_list[jjk])# 河流的名字
        if jjk == 4:
            ax_hah[pt,jjk].set_title(list(Wi_sqp.keys())[pt]) # 年代
        
plt.savefig('Hist_all.pdf')
# plt.savefig('Hist_all.svg') 
plt.savefig('Hist_all.jpg',dpi=300) 
  
#%%
#process the result, transfer to pandas DataFrame
Wi_for_stat = {}
for sf in Wi_sqp.keys():
    Wi_st = Wi_sqp[sf].sort_values(by = 'Func') #sort 
    Wi_st = Wi_st.reset_index(drop=True)    #reindex
    
    ma = Wi_st[0:1].T   # the max probability model
    Wi_len = len(Wi_st)
    # Wi_st = Wi_st[int(0.025*Wi_len):int(0.975*Wi_len)]    # 删除前2.5%和后2.5%
    # Wi_st = Wi_st[:]    #不删除

    means = Wi_st.mean()
    sd = Wi_st.std()
    # ma = Wi_st.max()
    wi_stat = pd.concat([means,sd,ma] ,axis = 1)
    wi_stat.columns = ['Me','SD','Ma']
    #再加一列做为表格
    wi_stat = wi_stat.round(2)
    # wi_stat['char'] = wi_stat['Me'].apply(str)+'±'+wi_stat['SD'].apply(str)
    
    Wi_for_stat[sf] = wi_stat.T

#handling data
X = np.linspace(0,3500,100)
Y_me_all = pd.DataFrame([])
Y_me_all['X'] = X
Y_ma_all = pd.DataFrame([])
Y_ma_all['X'] = X

#computer the fitted curve
Me_comp = []
Me_sd = []
Ma_comp = []
for sp in Wi_for_stat.keys():
    Wi_tmp = Wi_for_stat[sp]
    Wi_tmp.drop(['Func'],axis=1,inplace=True)

    Wi_Me = Wi_tmp.iloc[0:1]   #mean value 
    Wi_SD = Wi_tmp.iloc[1:2]  #SD 
    Wi_Ma = Wi_tmp.iloc[2:] #best model
        
    Me_haha = Wi_Me.rename(index ={'Me':sp},inplace=False)
    Me_comp.append(Me_haha) #parameters for plot
    
    sd_haha = Wi_SD.rename(index ={'SD':sp},inplace=False)
    Me_sd.append(sd_haha)  #parameters for plot
    
    Ma_haha = Wi_Ma.rename(index ={'Ma':sp},inplace=False)
    Ma_comp.append(Ma_haha) #parameters for plot
    
    Y_me = np.array([])  # Mean model
    Y_ma = np.array([])  # Best model
    for x in X:
        Y_me = np.append(Y_me,river_function(x,Wi_Me,norm_data))
        Y_ma = np.append(Y_ma,river_function(x,Wi_Ma,norm_data))
    Y_me_all[sp] = Y_me
    Y_ma_all[sp] = Y_ma

#combine to a dataframe
Me_comp = pd.concat(Me_comp)
Me_sd = pd.concat(Me_sd)
Ma_comp = pd.concat(Ma_comp)
#calculate the Bayes model curve
river_names = np.sort(list(bayes_MAP.keys())) #sort to avoid dislocation
fitted_data_basin_Y = {}
for i in river_names:
    #test data
    river_temp = i   
    #BayesMix fitted curve
    norm_para = bayes_MAP[river_temp]
    X_basin = np.linspace(0,3500,1000) 
    Y_basin =[]
    for x in X_basin:
        Y_basin.append(combined_river_function(x,norm_para))
    fitted_data_basin_Y[i] = Y_basin 

#%%
#plot the relative contribution figure
fig, ax_1 = plt.subplots(1, 1,figsize=(8,7))
#对年代逆序
index_new = list(Wi_sqp.keys())
index_new.reverse()
Me_comp = Me_comp.reindex(index = index_new)
# Me_comp = Me_comp.apply(lambda x : x/np.sum(x))
Me_comp = Me_comp.div(Me_comp.sum(axis=1),axis=0) #normalize
Me_comp = Me_comp.round(2)
Me_comp = Me_comp[['Dong River','Bei River','He River','Gui River','Liu River','Upper Hongshui River','You River','Zuo River']]
Me_sd = Me_sd.reindex(index = index_new)
Me_sd = Me_sd.div(Me_comp.sum(axis=1),axis=0)    #normalize
Me_sd = Me_sd.round(2)

Ma_comp = Ma_comp.reindex(index = index_new)
Ma_comp = Ma_comp[['Dong River','Bei River','He River','Gui River','Liu River','Upper Hongshui River','You River','Zuo River']]
Ma_comp.plot(marker = '.', ax=ax_1,linestyle='--',legend = False,alpha = 0.8)
plt.gca().set_prop_cycle(None)  # reset the color 
Me_comp.plot(marker='.',yerr = Me_sd ,ax = ax_1, alpha = 0.8,table =False) #

plt.title('Relative contribution',fontsize = 20)
# ax_1.get_xaxis().set_visible(False)
ax_1.set_ylim([0,1])
ax_1.set_ylabel('Relative contribution',fontsize = 15)
plt.xticks(fontsize=11)
plt.yticks(fontsize=12)
# plt.legend(fontsize=5)
plt.savefig(date+'Contribution'+name_use+'.pdf',bbox_inches='tight')
# plt.savefig(date+'Contribution'+name_use+'.svg',bbox_inches='tight')
plt.savefig(date+'Contribution'+name_use+'.jpg',bbox_inches='tight',dpi=300)

#plot area figure
Me_comp.plot.area(stacked=True,alpha = 0.8)
plt.savefig(date+'area.pdf')
# plt.savefig(date+'area.svg')

#plot pie figure
# fig_pie1, ax_pi1 = plt.subplots(1, 1,figsize=(20,8))
# Me_comp_for_pie = Me_comp.T*10  # *10 to normalize
# Me_comp_for_pie[Me_comp_for_pie<0.5] = 0 #删除5%以下的
# Me_comp_for_pie.plot.pie(subplots=True,ax=ax_pi1)
# plt.savefig(date+'Pie_Me'+name_use+'.pdf',bbox_inches='tight')
# # plt.savefig(date+'Pie_Me'+name_use+'.svg',bbox_inches='tight')

# fig_pie2, ax_pi2 = plt.subplots(1, 1,figsize=(20,8))
# Ma_comp_for_pie = Ma_comp.T*10
# Ma_comp_for_pie.plot.pie(subplots=True,ax=ax_pi2)
# # Ma_comp_for_pie[Ma_comp_for_pie<5] = 0 #删除5%以下的
# plt.savefig(date+'Pie_Ma'+name_use+'.pdf',bbox_inches='tight')
# # plt.savefig(date+'Pie_Ma'+name_use+'.svg',bbox_inches='tight')

#KDE,Sug,Normal fitted function figure
import seaborn as sns
sns.reset_defaults()
sample_names = Wi_sqp.keys()
f, axes = plt.subplots(len(sample_names), 1, figsize=(8,3.5*len(sample_names)), sharex=False)
j = 0 #for the 
for i in sample_names:
#    sns.despine(left=True)
    stp = 40
    bins = np.arange(0,3500+stp,stp)
    sns.distplot(basin_data[i]['Age'], rug=True, \
                  rug_kws={'height':0.03,'color':'orange'},\
                  bins = bins,kde=True,axlabel = False,
                  kde_kws={"shade": False,'bw':40,'label': 'KDE bw: 40','legend':True},\
                  color="b", ax=axes[j])
    axes[j].legend(loc=2,shadow = False,facecolor = 'w')
    axes[j].set_ylabel('For test data')
    axes[j].yaxis.get_major_formatter().set_powerlimits((1,2)) #change Y-axis to scientific notation
    axes[j].set_xlim(xmin = -100,xmax = 3600)
    if j  == (len(list(sample_names))-1):
        axes[j].set_xlabel('Age (Ma)')
    n = len(basin_data[i]['Age']) #number of data
    
    name_temp = i
    samples = basin_data[i].drop_duplicates(subset=['Sample'],keep='first')
    samples['Sample'] = samples['Sample'].astype(str)
    samples_name = ", ".join(list(samples['Sample']))
    txt = str(name_temp)+'\n'+'n='+str(n)+'\n'+'Samples: '+samples_name
    axes[j].text(0.77,0.5,txt,horizontalalignment='center',\
        verticalalignment='center', transform=axes[j].transAxes)
   
    # plot table
    comp_map_use = bayes_MAP[i]
    comp_map_use = comp_map_use.round(2)
    the_table2 = axes[j].table(cellText = comp_map_use.values,rowLabels=comp_map_use.index+1,\
                      colLabels= comp_map_use.columns,loc = 'upper center',cellLoc='center'\
                      ,colWidths=[0.1]*3,bbox=[0.35,0.47,0.3,0.53])
    the_table2.set_fontsize(9)
    #plot fitted Gauss model
    ax2 = axes[j].twinx() 
    ax2.plot(X,Y_me_all[i],color = 'r',alpha = 0.5,label = 'Fitted SQP Mean')
    ax2.plot(X,Y_ma_all[i],color = 'purple',alpha = 0.5,label = 'Fitted SQP Max')
    ax2.plot(X_basin,fitted_data_basin_Y[i],color = 'orange',label = 'Fitted normal function')
    ax2.set_ylabel('For fitted function')
    ax2.yaxis.get_major_formatter().set_powerlimits((1,2))
    ax2.legend(loc = 1,shadow = False,facecolor = 'w')
    j = j+1

plt.savefig(date+'fitted_curve_vs_KDE_max'+name_use+'.pdf')
# plt.savefig(date+'fitted_curve_vs_KDE_max'+name_use+'.svg')
plt.savefig(date+'fitted_curve_vs_KDE_max'+name_use+'.jpg',dpi=300)
