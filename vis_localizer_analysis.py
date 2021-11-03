# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:42:58 2019

Analyze audio-visual ECoG experiment

@author: Idan Tal
"""

import sys
print('Python:{}'.format(sys.version))

import pandas
import numpy as np
from scipy import stats
from scipy.signal import hilbert as hilb
from scipy.fftpack import next_fast_len as nflen
from statsmodels.stats.multitest import multipletests as mt
from math import ceil
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn
import mne
import pickle
import time 

################################### FUNCTIONS #############################

def define_trials(mne_info, triggers, log_events):
    trl = np.zeros((len(triggers),3), dtype = int)
    trl[:,0] = triggers
    #trl[:,1] = trl[:,0] + trial_len
    trl[:,1] = 0
    
    # define image categories 
    # remove the underscores to unite categories
    categories_cut = log_events.copy()
    for category_ind in range(0,len(categories_cut)):
        temp_str = categories_cut[category_ind]
        ind_substr1 = temp_str.find('_')
        ind_substr2 = temp_str.find('_',ind_substr1+1)
        categories_cut[category_ind] = categories_cut[category_ind][ind_substr1+1:ind_substr2]
    
    unique_categories, ic = np.unique(categories_cut, return_inverse = True)    
    num_categories = len(unique_categories)
    
    trl[:,2] = ic
    
    # create a dictionary for event_id
    event_id = dict(zip(unique_categories, range(num_categories)))
    return trl, categories_cut, event_id


########################### Main script ###################################
###########################################################################
###########################################################################    
subjects_info = pandas.read_excel(r"H:\\HFbursts\\OneDrive_1_5-24-2019\\data\\vis_localizer_patients.xlsx")
subjects_info = subjects_info[subjects_info['Analysis Flag'] == 1]
thr = 5
for ctsubj in subjects_info.index:
    data_dir = r"H:\\HFbursts\\OneDrive_1_5-24-2019\\data\\" + subjects_info.loc[ctsubj,'Patient ID']
    file_name = "\\" + subjects_info.loc[ctsubj, 'Patient ID'] + "_" +  subjects_info.loc[ctsubj, 'Patient init'] + "_rawCommonRemoved_eventRelatedNatural.mat"
    info_file_name = "\\" + subjects_info.loc[ctsubj, 'Patient ID'] + "_" +  subjects_info.loc[ctsubj, 'Patient init'] + "_info_eventRelatedNatural.mat"
    save_dir = r"H:\\HFbursts\\OneDrive_1_5-24-2019\\data\\" + subjects_info.loc[ctsubj, 'Patient ID'] + "\\matlabOut\\trialsTransientsAllFreq_thr" + str(thr) + "\\"

    # load the info file for the current patient
    sfreq = 500
    temp_info = sio.loadmat(data_dir + info_file_name, struct_as_record=False, squeeze_me=True)
    info = temp_info["info"]
    del temp_info
    good_electrodes = info.good_electrodes-1 # note that indexing here starts at 1 (reduce 1 to fit Python indexing)
    elec_list = np.ndarray.tolist(info.elec_list[good_electrodes])
    log_events = np.ndarray.tolist(info.log_events_noResp[1:-1])
    triggers = info.triggers[1:-1]
    # create mne info file 
    mne_info = mne.create_info(ch_names = elec_list, sfreq = sfreq)
    
    # load the matlab matrix with the raw data
    temp = sio.loadmat(data_dir + file_name)
    temp = temp["data_commonRef"]
    data = np.array(temp[good_electrodes,:])
    del temp
    
    ########################### find channels with increase in HFA ##############
    # filter the data 
    l_freq = 70
    h_freq = 150
    data_filt = mne.filter.filter_data(data, sfreq, l_freq, h_freq, copy = True)
    # get the hilbert analytic amplitude
    hilbert3 = lambda data_filt: hilb(data_filt, nflen(len(data_filt[0])))[:len(data_filt[0])]
    data_env = abs(hilbert3(data_filt))
    data_env = data_env[:, 0 : len(data[0])]
    
#    # check output for one channel
#    plt.figure()
#    plt.plot(data_filt[1,:])
#    plt.hold(True)
#    plt.plot(data_env[1,:],'r')
    
    # segment the data into trials
    trl, categories_cut, event_id = define_trials(mne_info, triggers, log_events)
    
    
    # create raw data object
    raw_env = mne.io.RawArray(data_env, mne_info, first_samp=0, copy='auto', verbose=None)
    # plot the raw data
    #plt.figure()
    #raw.plot(n_channels=4, title='Data from arrays',
      #   show=True)
    
    epochs = mne.Epochs(raw_env, events = trl,
                         event_id = event_id, tmin = -2, tmax = 2, preload = True)
    
    epochs_bl = epochs.copy()
    epochs_bl = epochs_bl.crop(tmin = -0.5, tmax = 0.)
    epochs_act = epochs.copy()
    epochs_act = epochs_act.crop(tmin = 0., tmax = 0.5)
    
    data_bl = epochs_bl.get_data()
    data_act = epochs_act.get_data()

    mean_data_bl = np.average(data_bl, axis = 2)
    mean_data_act = np.average(data_act, axis = 2)
    
    t, prob = stats.ttest_ind(mean_data_act, mean_data_bl, axis = 0)
    # correct formultiple comparisons using FDR'
    reject = mt(prob, method = 'fdr_bh', alpha = 0.05)
    ind_sig_chn = np.argwhere(reject[0] & np.asarray(t > 0))
    sig_HFA_chn = np.asarray(elec_list)
    sig_HFA_chn = sig_HFA_chn[ind_sig_chn]
    
    with open(data_dir + '\\matlabOut\\sig_HFA_chn.pickle', 'wb') as f:
        pickle.dump([sig_HFA_chn, reject], f)
        
    with open(data_dir + 'matlabOut\\sig_HFA_chn.pickle', 'rb') as f:
        sig_HFA_chn, reject = pickle.load(f)
    # plot the epoched data
    epochs.plot(picks = 'all')
    
# =============================================================================
# =============================================================================
    ######################### start time frequency analysis ##################
    
    # cut desired segments and channels from the original data
    epochs_cut = epochs.copy()
    epochs_cut.crop(tmin = -1, tmax = 1)
    epochs_cut.pick_channels(sig_HFA_chn[:,0])
    epochs_cut.resample(300., npad = 'auto')
    
    picks = mne.pick_types(raw_env.info, misc = True, stim=False)
    chn_ind = 0
    power = mne.time_frequency.tfr_morlet(epochs_cut, np.arange(70,150), 5, use_fft=False, return_itc=False, decim=1, n_jobs=1, picks = 'all', zero_mean=True, average=False, output='power', verbose=None)
    power_fname = save_dir + "sigHFAchn_tfr.h5"
    power.save(power_fname)
    
    mne.time_frequency.EpochsTFR.save(power,power_fname, overwrite=True)
    # read the power data
    power = mne.time_frequency.read_tfrs(power_fname)  
    avg_power = power.average()
    # plot all the TFRs
    fig = plt.figure()
    for chn_ind in np.arange(0,len(sig_HFA_chn[:,0])):
        ax2plt = fig.add_subplot(ceil(len(sig_HFA_chn[:,0])/3), 3,chn_ind+1)
        avg_power.plot(chn_ind, title='Significant channels TFR', tmin = -0.1, tmax = 0.5
                       , baseline = (-0.1, 0.), mode = 'percent', axes = ax2plt)
        ax2plt.title.set_text(sig_HFA_chn[chn_ind,0])
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    fig.set_size_inches(19.2, 10.8)
    fig_save_name = save_dir + "sigHFAchn_tfr.pdf"
    plt.show()
    plt.savefig(fig_save_name, bbox_inches='tight', dpi = 300)
    plt.close(fig)
# =============================================================================
# 
# =============================================================================



    
    