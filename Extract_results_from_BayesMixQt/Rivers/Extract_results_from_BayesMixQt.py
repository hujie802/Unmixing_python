"""
Spyder Editor

This is a temporary script file.
"""
import os 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

#===================== read file =========================================
l = os.getcwd()
names_all = os.listdir(l)
names_txt = [name for name in names_all if (name.endswith('.txt'))]

names_txt.sort()
date = '3_2'  # set the date on the file name for easy finding.
writer_map = pd.ExcelWriter('map_'+date+'.xlsx')

#read Zircon U-Pb data
Data_mixed_rivers = pd.read_excel('Data_for_river_in_optimization.xlsx',sheet_name = None)
#===================== read file end =========================================

#=====================  functions  =========================================
def river_function(x,norm_para):
    #no ufunc
    p_1 = stats.norm.pdf(x,norm_para['Mu'],norm_para['Sigma'])*norm_para['Pi']
    return p_1.sum()
#=====================  functions end  =========================================

#=====================  plots  =========================================
fig_number = len(names_txt)
fig,axes=plt.subplots(2*fig_number,1,figsize=(9,6*fig_number))
#set the sequence of the rivers according the geography map, note: the name must be as same as the file name in the dir
name_geo = ['Dong.txt','Bei.txt','He.txt','Gui.txt','Liu.txt','Upper Hongshui.txt','You.txt','Zuo.txt']
j = 0 
for file_name in name_geo:
    data = pd.read_csv(file_name,error_bad_lines=False, encoding='utf-8')
    sample_name = file_name[:-4] + ' River'
    
    #extract component data，Max Posterior probability
    comp_map_start = data[data.iloc[:,0].str.contains('MAP no of components')].index.tolist()[0]
    comp_map_end = data[data.iloc[:,0].str.contains('Alpha Error scale for MAP model')].index.tolist()[0]
    
    comp_map_data = data.iloc[comp_map_start+1:comp_map_end-1,0]
    comp_map_data = comp_map_data.str.split(' ', expand=True)
    comp_map_use = pd.DataFrame()
    comp_map_use[['Mu','Sigma','Pi']] = comp_map_data[[5,10,16]]
    comp_map_use.reset_index(inplace = True,drop = True)
    comp_map_use = comp_map_use.apply(pd.to_numeric, errors='ignore')
    comp_map_use.to_excel(writer_map,sheet_name = sample_name,index = False)
    
    #extract cluster data Max Posterior probability
    cluster_probability_start = data[data.iloc[:,0].str.contains('Cluster Inferred Ages kcomp')].index.tolist()[0]
    cluster_probability_end = data[data.iloc[:,0].str.contains('END MAP no of components')].index.tolist()[0]
    cluster_data = data.iloc[cluster_probability_start+1:cluster_probability_end,0]
    cluster_data = cluster_data.str.split('\t',expand = True)
    cluster_data.reset_index(inplace = True,drop = True)
    cluster_data = cluster_data.apply(pd.to_numeric,errors = 'ignore')

    #drop last line
    cluster_data.drop(cluster_data.columns[len(cluster_data.columns)-1], axis=1, inplace=True)
    
    #prepare fitted curve
    X = np.linspace(0,3500,1000)    
    Y = []
    for x in X:
        Y.append(river_function(x,comp_map_use))
    
    name_temp = file_name[:-4]+' River'
    stp = 40
    bins = np.arange(0,3500+stp,stp)
    #plot histogram,sug and KDE 
    sns.distplot(Data_mixed_rivers[name_temp]['Age'], rug=True, \
                  rug_kws={'height':0.03,'color':'orange'},\
                  bins = bins,kde=True,axlabel = False,
                  kde_kws={"shade": True,'bw':40,'label': 'KDE bw: 40','legend':True},\
                  color="b", ax=axes[2*j])

    axes[2*j].legend(loc=2,shadow = False,facecolor = 'w')
    axes[2*j].set_ylabel('For observed data')
    axes[2*j].set_title(name_temp)
    axes[2*j].yaxis.get_major_formatter().set_powerlimits((1,2)) #Y-axis change to scientific format
    n = len(Data_mixed_rivers[name_temp]['Age'])
    samples = Data_mixed_rivers[name_temp].drop_duplicates(subset=['Sample'],keep='first')
    samples['Sample'] = samples['Sample'].astype(str)
    samples_name = ", ".join(list(samples['Sample']))
    txt = 'n='+str(n)+'\n'+'Samples: '+samples_name
    axes[2*j].text(0.82,0.75,txt,horizontalalignment='center',\
        verticalalignment='center', transform=axes[2*j].transAxes)
    #plot fitted normal function
    ax2 = axes[2*j].twinx() 
    ax2.plot(X,Y,color = 'r',label = 'Fitted normal function')
    ax2.set_ylabel('For fitted function')
    ax2.yaxis.get_major_formatter().set_powerlimits((1,2))
    ax2.legend(loc = 1,shadow = False,facecolor = 'w')

    comp_map_use = comp_map_use.round(2)
    the_table2 = axes[2*j].table(cellText = comp_map_use.values,rowLabels=comp_map_use.index+1,\
                     colLabels= comp_map_use.columns,loc = 'upper center',cellLoc='center'\
                     ,colWidths=[0.1]*3,bbox=[0.35,0.4,0.3,0.6])
    the_table2.set_fontsize(8.5)
    
    axes[2*j].set_xlim([0,3500])
    #plot classfication 
    cluster_data.plot(x =0, marker = 'x',ax = axes[2*j+1])
    axes[2*j+1].set_xlabel('Age (Ma)')
    axes[2*j+1].set_ylabel('Probability')
    axes[2*j+1].set_xlim([0,3500])
    
    j = j+1
plt.tight_layout()
plt.savefig(date+'_all.pdf')
writer_map.save()
print('Finish')