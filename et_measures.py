# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:48:55 2021

plot results of transfer ratio analysis 
and spectra of eye-tracking measures

@author: Idan Tal
"""



import os
import numpy as np
import seaborn as sns
import scipy.io as io
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as stat
from scipy.fft import fft,  fftfreq
from scipy.signal import coherence as coh
from scipy.signal import savgol_filter
import random

sns.set(font_scale=1.2)
sns.set_style("whitegrid")
a4_dims = (11.7, 8.27)
sd_thr = 3

# load the data (saved by matlab)
file_names = ['JO031014','JO031017','JO032014', 'JO032017','JO033015','JO033019',
              'JO037038017','JO037038021','JO040041015','JO040041018','JO043044019',
              'JO043044021','JO048049023','JO048049029','JO050051016','JO050051019']
fbands = ['mua']
layer1 = '4'
layer2 = '3'
smooth_data = True

db_dir_et = r'K:\slowFluctuationsNHP\output'
db_dir_mua = r'K:\slowFluctuationsNHP\output\figures_FWRcorrected_ongoing_rem_outliers_'+ str(sd_thr) + '_sdThr_'
fig_save_dir = r'K:\slowFluctuationsNHP\output\figures_ongoing_rem_outliers_3_sdThr_mua'

if not(os.path.isdir(fig_save_dir)):
    os.mkdir(fig_save_dir)



num_sac_all = []
pup_sz_all = []
mua_ratio_all = []
yf_num_sac_all = []
yf_mean_pup_sz_all = []
yf_auc_ratio_all = []
coh_mua_sac_all = []
coh_mua_pup_all = []

for file_name in file_names:
    # load auc of the MUA data
    data = io.loadmat(db_dir_mua + fbands[0] + '/' + file_name + '_aucImg')
    auc_ratio_trl_mua = np.array(data['aucImg'])
    avg_auc_ratio_mua = np.nanmean(auc_ratio_trl_mua,axis = 1)
    len_mua = np.shape(auc_ratio_trl_mua)
    se_auc_ratio_mua = np.nanstd(auc_ratio_trl_mua, axis = 1)/np.sqrt(len_mua[1])
    
    
    
    # load eye tracking information
    et_file_name = db_dir_et + '/' + file_name + '_trialData'
    et_data = io.loadmat(et_file_name)
    good_trials = et_data['trialData']['goodTrials'][0][0][0]
    sac_start = et_data['trialData']['et'][0,0]['sacS'][0,0][0]
    pup_sz = et_data['trialData']['et'][0,0]['pupSizeImg'][0,0][0]
    num_sac = []
    mean_pup_sz = []
    sd_pup_sz = []
    for trl_ind in np.arange(len(sac_start)):
        num_sac.append(len(sac_start[trl_ind][0]))
        mean_pup_sz.append(np.mean(pup_sz[trl_ind]))
        sd_pup_sz.append(np.std(pup_sz[trl_ind]))
        
    num_sac = np.array(num_sac)
    num_sac = num_sac[good_trials-1]
    num_sac_smooth = savgol_filter(num_sac,9,3)
    mean_pup_sz = np.array(mean_pup_sz)
    mean_pup_sz = mean_pup_sz[good_trials-1]
    mean_pup_sz_smooth = savgol_filter(mean_pup_sz,9,3)
    sd_pup_sz = np.array(sd_pup_sz)
    sd_pup_sz = sd_pup_sz[good_trials-1]
    
    # remove 'bad' trials from MUA
    avg_auc_ratio_mua = avg_auc_ratio_mua[good_trials-1]
    se_auc_ratio_mua = se_auc_ratio_mua[good_trials-1]
    avg_auc_ratio_mua_smooth = savgol_filter(avg_auc_ratio_mua,9,3)
    
    num_trials = len(num_sac)
    
    # plot the avg auc ratio
    plt.figure()
    plt.plot(np.arange(1,num_trials+1),avg_auc_ratio_mua, 'b-')
    plt.plot(np.arange(1,num_trials+1),avg_auc_ratio_mua_smooth, 'k-')
    plt.fill_between(np.arange(1,num_trials+1),avg_auc_ratio_mua-se_auc_ratio_mua, avg_auc_ratio_mua+se_auc_ratio_mua)
    plt.show()
    plt.xlabel('Trial num')
    plt.ylabel('MUA transfer ratio (layer4/layer3)')
    plt.title(file_name)
    plt.savefig(fig_save_dir  + '/' + file_name + '_auc_ratio_mua.png',resolution = 600)
    plt.close()
    
    # plot the number of saccades for each trial
    plt.figure()
    plt.plot(np.arange(1,num_trials+1),num_sac, 'b-')
    plt.plot(np.arange(1,num_trials+1),num_sac_smooth, 'k-')
    plt.show()
    plt.xlabel('Trial num')
    plt.ylabel('Number of saccades')
    plt.title(file_name)
     
    plt.savefig(fig_save_dir  + '/' + file_name + '_numSaccades_mua.png',resolution = 600) 
    plt.close() 
    
    # plot the number of fixations for each trial
    plt.figure()
    plt.plot(np.arange(1,num_trials+1),mean_pup_sz, 'b-')
    plt.plot(np.arange(1,num_trials+1),mean_pup_sz_smooth, 'k-')
    plt.fill_between(np.arange(1,num_trials+1),mean_pup_sz-sd_pup_sz, mean_pup_sz+sd_pup_sz)
    plt.show()
    plt.xlabel('Trial num')
    plt.ylabel('Pupil Diameter, microM')
    plt.title(file_name)
    plt.savefig(fig_save_dir  + '/' + file_name + '_pupilDiameter_mua.png',resolution = 600) 
    plt.close() 
    
    '''
    # calculate the fft of the measures
    '''
    if smooth_data:
        num_sac = num_sac_smooth
        mean_pup_sz = mean_pup_sz_smooth
        avg_auc_ratio_mua = avg_auc_ratio_mua_smooth
    # number of saccades
    T = 7
    N = len(num_sac)
    yf_num_et_events = fft(stat.zscore(num_sac))
    yf_num_et_events  = np.abs(yf_num_et_events[0:N//2])
    xf = fftfreq(N, T)
    
    # shuffle to get threshold
    shuffle_ind = np.arange(0,1000)
    freq_vec = xf[0:N//2]
    p_num_et_events = []
    yf_shuffle = np.zeros([len(freq_vec),len(shuffle_ind)])
    for i in shuffle_ind:
        data_temp = np.array(num_sac)
        random.shuffle(data_temp)
        yf_shuffle_temp = fft(stat.zscore(data_temp))
        yf_shuffle[:,i] = np.abs(yf_shuffle_temp[0:N//2])
        p_num_et_events.append(np.percentile(yf_shuffle[:,i], 75))
        
    
    # plot spectrum
    plt.figure()
    plt.plot(xf[0:N//2],yf_num_et_events, 'k-')
    plt.plot([0, np.max(xf)], [np.percentile(p_num_et_events,95), np.percentile(p_num_et_events,95)], 'r--', lw=2)
    plt.show()
    plt.xlabel('Frequency, Hz')
    plt.ylabel('Amplitude')
    plt.title(file_name + ' number of saccades spectra')
    plt.savefig(fig_save_dir  + '/' + file_name + '_numSaccadesSpect_mua.png',resolution = 600) 
    plt.close()
    
    # mean pupil size
    T = 7
    N = len(num_sac)
    yf_mean_pup_sz = fft(stat.zscore(mean_pup_sz))
    yf_mean_pup_sz  = np.abs(yf_mean_pup_sz[0:N//2])
    xf = fftfreq(N, T)
    
    # shuffle to get threshold
    shuffle_ind = np.arange(0,1000)
    freq_vec = xf[0:N//2]
    p_mean_pup_sz = []
    yf_shuffle = np.zeros([len(freq_vec),len(shuffle_ind)])
    for i in shuffle_ind:
        data_temp = np.array(mean_pup_sz)
        random.shuffle(data_temp)
        yf_shuffle_temp = fft(stat.zscore(data_temp))
        yf_shuffle[:,i] = np.abs(yf_shuffle_temp[0:N//2])
        p_mean_pup_sz.append(np.percentile(yf_shuffle[:,i], 75))
        
    
    # plot the pupil diameter spectrum for each trial
    plt.figure()
    plt.plot(xf[0:N//2],yf_mean_pup_sz, 'k-')
    plt.plot([0, np.max(xf)], [np.percentile(p_mean_pup_sz,95), np.percentile(p_mean_pup_sz,95)], 'r--', lw=2)
    plt.show()
    plt.xlabel('Frequency, Hz')
    plt.ylabel('Amplitude')
    plt.title(file_name + ' pupil diameter spectra')
    plt.savefig(fig_save_dir  + '/' + file_name + '_pupilDiameterSpect_mua.png',resolution = 600) 
    plt.close()
    
    # auc ratio MUA
    T = 7
    N = len(avg_auc_ratio_mua)
    yf_auc_ratio = fft(stat.zscore(avg_auc_ratio_mua))
    yf_auc_ratio  = np.abs(yf_auc_ratio[0:N//2])
    xf = fftfreq(N, T)
    
    # shuffle to get threshold
    shuffle_ind = np.arange(0,1000)
    freq_vec = xf[0:N//2]
    p_auc_ratio = []
    yf_shuffle = np.zeros([len(freq_vec),len(shuffle_ind)])
    for i in shuffle_ind:
        data_temp = np.array(avg_auc_ratio_mua)
        random.shuffle(data_temp)
        yf_shuffle_temp = fft(stat.zscore(data_temp))
        yf_shuffle[:,i] = np.abs(yf_shuffle_temp[0:N//2])
        p_auc_ratio.append(np.percentile(yf_shuffle[:,i], 75))
        
    
    # plot spectrum
    plt.figure()
    plt.plot(xf[0:N//2],yf_auc_ratio, 'k-')
    plt.plot([0, np.max(xf)], [np.percentile(p_auc_ratio,95), np.percentile(p_auc_ratio,95)], 'r--', lw=2)
    plt.show()
    plt.xlabel('Frequency, Hz')
    plt.ylabel('Amplitude')
    plt.title(file_name + ' MUA auc ratio spectra')
    plt.savefig(fig_save_dir  + '/' + file_name + '_aucSpect_mua.png',resolution = 600) 
    plt.close()
    

    
    sampling_rate = 1/T
    # coherence between MUA and number of saccades
    n_points = 10
    n_overlap = 5
    f, coh_mua_sac = coh(stat.zscore(avg_auc_ratio_mua), stat.zscore(num_sac), fs=sampling_rate, window='hann', nperseg=n_points, noverlap=n_overlap)
    plt.figure()
    plt.plot(f, coh_mua_sac)
    plt.title(file_name + '-Coherence auc MUA-num. saccades')
    plt.xlabel('Frequency, Hz')
    plt.ylabel('Coherence')
    plt.savefig(fig_save_dir  + '/' + file_name + '_cohMUAnumSac.png',resolution = 600) 
    plt.close()
    
    # coherence between MUA and mean pupil diameter
    n_points = 10
    n_overlap = 5
    f, coh_mua_pup = coh(stat.zscore(avg_auc_ratio_mua), stat.zscore(mean_pup_sz), fs=sampling_rate, window='hann', nperseg=n_points, noverlap=n_overlap)
    plt.figure()
    plt.plot(f, coh_mua_pup)
    plt.title(file_name + '-Coherence auc MUA-pupil diameter')
    plt.xlabel('Frequency, Hz')
    plt.ylabel('Coherence')
    plt.savefig(fig_save_dir  + '/' + file_name + '_cohMUApupSz.png',resolution = 600) 
    plt.close()
    
    # concatenate measures across recordings
    num_sac_all = np.append(num_sac_all,stat.zscore(num_sac),axis = 0)
    pup_sz_all = np.append(pup_sz_all,stat.zscore(mean_pup_sz),axis = 0)
    mua_ratio_all = np.append(mua_ratio_all, stat.zscore(avg_auc_ratio_mua), axis = 0)
    yf_num_sac_all = np.append(yf_num_sac_all, yf_num_et_events, axis = 0)
    yf_mean_pup_sz_all = np.append(yf_mean_pup_sz_all, yf_mean_pup_sz, axis = 0)
    yf_auc_ratio_all = np.append(yf_auc_ratio_all, yf_auc_ratio, axis = 0)
    coh_mua_sac_all.append(coh_mua_sac)
    coh_mua_pup_all.append(coh_mua_pup)
'''
# calculate the correlation between MUA and ET measures across all trials and recordings
'''
# correlation between auc ratio and number of saccades
r, p = stat.pearsonr(mua_ratio_all,num_sac_all) 
fig, ax = plt.subplots()
plt.plot(mua_ratio_all, num_sac_all, 'o')
m, b = np.polyfit(mua_ratio_all, num_sac_all, 1)
plt.plot(mua_ratio_all, m*mua_ratio_all + b)
plt.title('Correlation MUA-num. saccades')
plt.xlabel('Normalized MUA')
plt.ylabel('Normalized Num. saccades')


# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr = 'r = ' + str(r) + ', p = ' + str(p)
# place a text box in upper left in axes coords
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.savefig(fig_save_dir  + '/' + 'gaCorr_aucMUA_numSac.png',resolution = 600) 


# correlation between auc ratio and pupil size
r, p = stat.pearsonr(mua_ratio_all,pup_sz_all) 

fig, ax = plt.subplots()
plt.plot(mua_ratio_all, pup_sz_all, 'o')
m, b = np.polyfit(mua_ratio_all, pup_sz_all, 1)
plt.plot(mua_ratio_all, m*mua_ratio_all + b)
plt.title('Correlation MUA-pupil diameter')
plt.xlabel('Normalized MUA')
plt.ylabel('Normalized Pupil Diameter, microM')

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr = 'r = ' + str(r) + ', p = ' + str(p)
# place a text box in upper left in axes coords
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.savefig(fig_save_dir  + '/' + 'gaCorr_aucMUA_pupSz.png',resolution = 600) 


# correlation between fft of auc and fft of number of saccades
r, p = stat.pearsonr(yf_auc_ratio_all,yf_num_sac_all) 

fig, ax = plt.subplots()
plt.plot(yf_auc_ratio_all, yf_num_sac_all, 'o')
m, b = np.polyfit(yf_auc_ratio_all, yf_num_sac_all, 1)
plt.plot(yf_auc_ratio_all, m*yf_auc_ratio_all + b)
plt.title('Spectrum Correlation MUA-num. saccades')
plt.xlabel('Normalized MUA spectrum magnitude')
plt.ylabel('Normalized num. saccades spectrum magnitude')

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr = 'r = ' + str(r) + ', p = ' + str(p)
# place a text box in upper left in axes coords
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.savefig(fig_save_dir  + '/' + 'gaSpectCorr_aucMUA_numSac.png',resolution = 600) 


# correlation between fft of auc and fft of pupil size
r, p = stat.pearsonr(yf_auc_ratio_all,yf_mean_pup_sz_all) 

fig, ax = plt.subplots()
plt.plot(yf_auc_ratio_all, yf_mean_pup_sz_all, 'o')
m, b = np.polyfit(yf_auc_ratio_all, yf_mean_pup_sz_all, 1)
plt.plot(yf_auc_ratio_all, m*yf_auc_ratio_all + b)
plt.title('Spectrum Correlation MUA-pupil diameter')
plt.xlabel('Normalized MUA spectrum magnitude')
plt.ylabel('Normalized Pupil Diameter spectrum magnitude')

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr = 'r = ' + str(r) + ', p = ' + str(p)
# place a text box in upper left in axes coords
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.savefig(fig_save_dir  + '/' + 'gaSpectCorr_aucMUA_pupSz.png',resolution = 600) 


# plot coherence across all recordings
mean_coh_mua_pup = np.mean(coh_mua_pup_all, axis = 0)
se_coh_mua_pup = np.std(coh_mua_pup_all, axis = 0)/np.sqrt(len(coh_mua_pup_all))
plt.figure()
plt.plot(f,mean_coh_mua_pup, 'k-')
plt.fill_between(f,mean_coh_mua_pup-se_coh_mua_pup, mean_coh_mua_pup+se_coh_mua_pup)
plt.show()
plt.xlabel('Frequency, Hz')
plt.ylabel('Coherence')
plt.title('Coherence between MUA and pupil diameter')
plt.savefig(fig_save_dir  + '/gaCoherenc_mua_pupilDiameter.png',resolution = 600) 


mean_coh_mua_sac = np.mean(coh_mua_sac_all, axis = 0)
se_coh_mua_sac = np.std(coh_mua_sac_all, axis = 0)/np.sqrt(len(coh_mua_sac_all))
plt.figure()
plt.plot(f,mean_coh_mua_sac, 'k-')
plt.fill_between(f,mean_coh_mua_sac-se_coh_mua_sac, mean_coh_mua_sac+se_coh_mua_sac)
plt.show()
plt.xlabel('Frequency, Hz')
plt.ylabel('Coherence')
plt.title('Coherence between MUA and number of saccades')
plt.savefig(fig_save_dir  + '/gaCoherenc_mua_numSac.png',resolution = 600) 

