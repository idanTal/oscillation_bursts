# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 05:57:57 2021

@author: Idan Tal
"""


# plot distributions of transfer ratio analysis 
import numpy as np
import seaborn as sns
import scipy.io as io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as stat
from scipy.fft import fft,  fftfreq
import random


sns.set(font_scale=1.2)
sns.set_style("whitegrid")

fig_save_dir = r'K:\slowFluctuationsNHP\output\figures_FWRcorrected_allGoodChnNewBL_filt_allBands_v3_layer4_layer3_noThr_vars'
a4_dims = (11.7, 8.27)
# Define file names
file_names = ['JO031014','JO031017','JO032014', 'JO032017','JO033015','JO033019',
              'JO037038017','JO037038021','JO040041015','JO040041018','JO043044019',
              'JO043044021','JO048049023','JO048049029','JO050051016','JO050051019']

layer1 = '4'
layer2 = '3'

for file_name in file_names:
    data = io.loadmat(r'K:\slowFluctuationsNHP\output\figures_FWRcorrected_allGoodChnNewBL_filt_theta_noThr_vars/'+file_name+'_TR_layer'+layer1+'_elec1_layer'+layer2+'_elec1')
    auc_ratio_trl = np.array(data['aucRatioTrl']['trl'][0,0])
    avg_auc_ratio = data['aucRatioTrl']['avg'][0][0][0]
    
    # get the eye tracking data (average for each trial)
    num_et_events = data['numETevents'][0]
    mean_pup_sz = data['meanPupSzTrl'][0]
    sd_pup_sz = data['sdPupSzTrl'][0]
    
    num_trials = len(num_et_events)
    
    '''
    # create data frame with the auc (at the single trial level)
    '''
    
    d_bands = {'auc ratio':[], 'num. fixations':[],'mean pupil diameter':[],'sd pupil diameter':[],'trial num':[]}
    for i in np.arange(0,num_trials):    
        temp_auc = auc_ratio_trl[0,i]
        trial_num = 'trial_' + str(i+1)
        for j in np.arange(0,len(temp_auc)):
            d_bands['auc ratio'].append(temp_auc[j][0])
            d_bands['trial num'].append(trial_num)
            d_bands['num. fixations'].append(num_et_events[i])
            d_bands['mean pupil diameter'].append(mean_pup_sz[i])
            d_bands['sd pupil diameter'].append(sd_pup_sz[i])
            
            
    d_bands['auc norm'] = stat.zscore(d_bands['auc ratio theta'])
    df_bands = pd.DataFrame(data=d_bands)
    
    
    # plot the number of fixations for each trial
    plt.figure()
    plt.plot(np.arange(1,41),num_et_events, 'k-')
    plt.show()
    plt.xlabel('Trial num')
    plt.ylabel('Pupil Diameter, microM')
    plt.title(file_name)
    
    # plot the number of fixations for each trial
    plt.figure()
    plt.plot(np.arange(1,41),mean_pup_sz, 'k-')
    plt.fill_between(np.arange(1,41),mean_pup_sz-sd_pup_sz, mean_pup_sz+sd_pup_sz)
    plt.show()
    plt.xlabel('Trial num')
    plt.ylabel('Pupil Diameter, microM')
    plt.title(file_name)
    
    
    '''
    # create dataframe with the average values (auc) for each trial
    
    '''
    
    # first get the normalized (z-scored) values of the auc
    norm_auc = [] 
    for i in np.arange(1,num_trials+1):
        ind = df_bands['trial num'] == 'trial_' + str(i)
        norm_auc.append(np.mean(df_bands['auc norm'][ind]))
    
    
    d_avg = {'norm auc': norm_auc,
             'num. fixations':num_et_events,
             'mean pupil diameter':mean_pup_sz,
             'sd pupil diameter':sd_pup_sz,
             'trial num':np.arange(1,num_trials+1)}
    
    df_avg = pd.DataFrame(data=d_avg)
    
    g = sns.pairplot(df_avg,kind='kde',vars = ['norm auc','num. fixations','mean pupil diameter','sd pupil diameter','trial num'])
    plt.gcf().subplots_adjust(bottom=0.15,left=0.15)
    for ax in g.axes.flatten():
        # rotate x axis labels
        ax.set_xlabel(ax.get_xlabel(), rotation = 90)
        # rotate y axis labels
        ax.set_ylabel(ax.get_ylabel(), rotation = 0)
        # set y labels alignment
        ax.yaxis.get_label().set_horizontalalignment('right')
    
    plt.savefig(fig_save_dir  + '/' + file_name + '_allvarsDistr.png',resolution = 600) 
    plt.close()  
        
    # spearman correlation coefficient
    df_avg = df_avg.dropna(axis=0)
    r = np.zeros([df_avg.shape[1],df_avg.shape[1]])
    pval = np.zeros([df_avg.shape[1],df_avg.shape[1]])
    for i in range(df_avg.shape[1]): 
        for j in range(df_avg.shape[1]):
            r[i,j], pval[i,j] = stat.spearmanr(df_avg[df_avg.columns[i]], df_avg[df_avg.columns[j]])
    
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.heatmap(pval,annot=True, linewidths=.5,xticklabels = df_avg.columns, yticklabels = df_avg.columns)
    plt.gcf().subplots_adjust(bottom=0.3,left=0.15)
    plt.savefig(fig_save_dir  + '/' + file_name + '_allvars_pval.png',resolution = 300) 
    plt.close()  
    
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.heatmap(r,annot=True, linewidths=.5,xticklabels = df_avg.columns, yticklabels = df_avg.columns)
    plt.gcf().subplots_adjust(bottom=0.3,left=0.15)
    plt.savefig(fig_save_dir  + '/' + file_name + '_allvars_rval.png',resolution = 300) 
    plt.close()  
    
    
    # calculate spectrum
    T = 7
    data_temp = np.array(df_avg['norm auc'])
    N = len(data_temp)
    yf = fft(data_temp)
    xf = fftfreq(N, T)
    
    d_freq = {'spect auc theta': np.abs(yf[0:N//2]),
              'freq, Hz': xf[0:N//2]}
    df_freq = pd.DataFrame(data = d_freq)
    
    '''
    Shuffle the auc data across trials to get a null distribution for frequency analysis
    '''
    shuffle_ind = np.arange(0,1000)
    p_band = []
    for i in shuffle_ind:
        data_temp = np.array(df_avg['norm auc'])
        random.shuffle(data_temp)
        yf_shuffle = fft(data_temp)
        yf_shuffle = np.abs(yf_shuffle[0:N//2])
        p_band.append(np.percentile(yf_shuffle, 95))
    
    
    
    # plot spectrum with 95th percentile thereshold
    sns.lineplot(x="freq, Hz", y="spect auc",
                 data=df_freq, color = 'r', label = 'AUC ratio')
    plt.plot([0, np.max(xf)], [np.percentile(p_band,95), np.percentile(p_band,95)], 'r--', lw=2)
    plt.ylabel('normalized auc ratio')
    plt.savefig(fig_save_dir  + '/' + file_name + '_spect.png') 
    plt.close()  
    
    
    df_freq_fn = fig_save_dir  + '/' + file_name + '_df_freq'
    df_freq.to_pickle(df_freq_fn)
    
    
