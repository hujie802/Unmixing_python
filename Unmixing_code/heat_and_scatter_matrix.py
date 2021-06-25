#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:55:30 2020

@author: jiehu
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# sns.set_theme(style="whitegrid")
# sns.set(style="darkgrid")

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'


date = '3_2'
file_name = 'basin_use_op_2021_2_25.xlsx'
name_use = file_name[:-5]

# MCMC_name = date+'_MCMC_random_9_22'+name_use+'.xlsx'
MCMC_name = date+'_sqp_random_basin_use_op_2021_2_25.xlsx'
Wi_MCMC = pd.read_excel(MCMC_name,sheet_name = None)

# f, axes = plt.subplots(1,len(Wi_MCMC.keys()), figsize=(11*len(Wi_MCMC.keys()),11))
col_n = 3
row_n = 2

f, axes = plt.subplots(row_n,col_n,figsize=(5*col_n,5*row_n))
# f, axes = plt.subplots(1,len(Wi_MCMC.keys()))
cbar_ax = f.add_axes([1, 0.3, 0.015, 0.4])   #x,y of bottom left point, width, heigth of the colorbar

p = 0 #位置参数
#one by one for heat corr
for MC_id in list(Wi_MCMC.keys())[::-1]:

    MCMC_all = Wi_MCMC[MC_id][:]
    MCMC_use = MCMC_all.drop('Func',axis = 1)
    # MCMC_use = MCMC_all[-1000:]
    MCMC_use.columns = ['Bei','Dong','Gui','He','Liu','Upper \n Hongshui','You','Zuo']
    MCMC_use = MCMC_use[['Dong','Bei','He','Gui','Liu','Upper \n Hongshui','You','Zuo']]
    colormap = plt.cm.RdBu
    # plt.figure(figsize=(8,7))
    # plt.title(MC_id+' Correlation of Rivers', y=1.05, size=15)
    corr = MCMC_use.astype(float).corr()
    corr = corr.round(2)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    i = p//col_n #0-axis
    j = p%col_n #1-axis
    
    axes[i][j].set_title(MC_id,size=15)
    sns.heatmap(corr, mask=mask, ax = axes[i][j],linewidths=0.1,vmax=1.0, vmin = -1.0,
                square=True, cmap=colormap, linecolor='white', annot=True,cbar=False)
    p = p+1
# f.colorbar()
f.colorbar(axes[0][1].collections[0], cax=cbar_ax)
plt.tight_layout()
plt.savefig('all_heat.pdf' ,bbox_inches = 'tight')
# plt.savefig('all_heat.svg' ,bbox_inches = 'tight')
plt.savefig('all_heat.jpg' ,bbox_inches = 'tight',dpi=300)

# for scatter
# #one by one
sns.set_style("darkgrid")
# sns.set_theme(style="darkgrid")

for MC_id in list(Wi_MCMC.keys()):

    MCMC_all = Wi_MCMC[MC_id][1000:]
    MCMC_use = MCMC_all.drop('Func',axis = 1)  
    MCMC_use.columns = ['Bei','Dong','Gui','He','Liu','Upper \n Hongshui','You','Zuo']
    MCMC_use = MCMC_use[['Dong','Bei','He','Gui','Liu','Upper \n Hongshui','You','Zuo']]
    g = sns.PairGrid(MCMC_use)
    g.map(sns.scatterplot)
    g.map(sns.kdeplot)
    # g.set_title(MC_id)
    # plt.subplots_adjust(top=0.9)
    g.fig.suptitle(MC_id,y = 1.01,fontsize=22)
    plt.tight_layout()
    # plt.title(MC_id)
    # plt.savefig(date+MC_id+'_cor_sc.pdf')
    g.savefig(date+MC_id+'_cor_sc.pdf')
    # g.savefig(date+MC_id+'_cor_sc.svg')
    g.savefig(date+MC_id+'_cor_sc.jpg',dpi = 300)

    
