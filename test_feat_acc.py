# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:40:28 2020

Classification based on features of transient oscillatory activity

@author: Idan Tal
"""



import sys
print('Python:{}'.format(sys.version))
import os

import pandas
import numpy as np
from scipy.io import loadmat
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class Features:
    def __init__(self,ID,feat_name, feat_code, mean_acc, sd_acc,conf_mat):
        self.ID = ID
        self.feat_name = feat_name
        self.feat_code = feat_code
        self.mean_acc = mean_acc
        self.sd_acc = sd_acc
        self.conf_mat = conf_mat
        
        
            
        
        

#chn_ind = 29
#subject_IDs = ['LIJ092', 'LIJ079', 'LIJ080', 'LIJ078_imp2', 'LIJ070_imp2', 'LIJ071', 'LIJ072', 'LIJ073', 'LIJ074', 'LIJ075', 'LIJ076', 'LIJ077','LIJ078']

subjects_info = pandas.read_excel(r"H:\\HFbursts\\OneDrive_1_5-24-2019\\data\\vis_localizer_patients.xlsx")
to_analyze = subjects_info['Analysis Flag']
subject_IDs = list(subjects_info.values[to_analyze == 1,0])
categories = ['animals', 'patterns', 'people', 'places', 'tools', 'words']
num_random_states = 10
temp = loadmat(r"H:\\HFbursts\\OneDrive_1_5-24-2019\\data\\LIJ071\\matlabOut\\trialsTransientsHFA_BOSC_powThr95_durThr3\\transientFeatureFiles\\featureCombinations.mat")
feature_combinations = temp['featureCombinations']
feat_list = []
feat_name_list = []
mean_sum_diag = [[]]*len(subject_IDs)
for subject_ind in range(0, len(subject_IDs)):
    for feat_ind1 in range(0, feature_combinations.size):
        for feat_ind2 in range(0,len(feature_combinations[0,feat_ind1])):
            conf_mat_all = np.zeros((num_random_states, len(categories),len(categories)))
            sum_diag = np.zeros(num_random_states)
            comb_temp = str(feature_combinations[0,feat_ind1][feat_ind2])
            comb_temp = comb_temp.replace(' ','__')
            save_dir_name = r"H:\\HFbursts\\OneDrive_1_5-24-2019\\data\\" + subject_IDs[subject_ind] + "\\matlabOut\\trialsTransientsHFA_BOSC_powThr95_durThr3\\transientFeatureFiles\\"
            file_name = save_dir_name + "featuresBOSC_" + comb_temp[1:-1] + ".txt"
            fig_save_dir = r"H:\\HFbursts\\figures\\" + subject_IDs[subject_ind] + "\\trialsTransientsHFA_BOSC_powThr95_durThr3\\figures_combinations"
            if not os.path.isdir(fig_save_dir):
                os.mkdir(fig_save_dir)
            temp_sum_diag = np.load(save_dir_name+"sum_diag_" + comb_temp[1:-1] + ".npy")
            temp_conf_mat = np.load(save_dir_name+"conf_mat_" + comb_temp[1:-1] + ".npy")
            feat_list.append(Features(subject_IDs[subject_ind], comb_temp[1:-1],comb_temp[1:-1], np.mean(temp_sum_diag), np.std(temp_sum_diag), np.mean(temp_conf_mat, axis = 0)))


features_name = np.array(['Count', 'Duration', 'Amplitude', 'Freq. span', 'Frequency', 'Latency', 'IBI', 'Num. cycles', 'maxLFphase'])
mean_acc_all = []
sd_acc_all = []
conf_mat_all = []
feat_val_list = []
feat_name_list = []
for feat_ind1 in range(0, feature_combinations.size):
        for feat_ind2 in range(0,len(feature_combinations[0,feat_ind1])):
            mean_acc = []
            conf_mat = []
            comb_temp = str(feature_combinations[0,feat_ind1][feat_ind2])
            comb_temp = comb_temp.replace(' ','__')
            feat_val_list.append(comb_temp[1:-1])
            features_name_temp = features_name[np.array(feature_combinations[0,feat_ind1][feat_ind2]-1)]
            feat_name_list.append(features_name_temp)
            for f in feat_list:
                if f.feat_code == comb_temp[1:-1]:
                    mean_acc.append(f.mean_acc)
                    conf_mat.append(f.conf_mat)
                    
            mean_acc_all.append(np.nanmean(mean_acc))
            sd_acc_all.append(np.nanstd(mean_acc))
            conf_mat_all.append(np.nanmean(conf_mat, axis = 0))



x = np.arange(len(mean_acc_all))
y = np.array(mean_acc_all)
err = np.divide(np.array(sd_acc_all),np.sqrt(len(subject_IDs)))
 
    
plt.plot(x,y,'k-')
plt.fill_between(x,y-err, y+err)
plt.xticks(x, feat_name_list)
plt.show() 

max_acc_ind = np.argmax(y)
f_max = feat_name_list[max_acc_ind]

# sort the features based on accuracy
indSort = y.argsort()
feat_name_list_sorted = feat_name_list.copy()
feat_name_list_sorted = [feat_name_list_sorted[i] for i in indSort]
err_sort = [err[i] for i in indSort]

y.sort()
plt.plot(x,y,'k-')
plt.fill_between(x,y-err, y+err)
plt.xticks(x, feat_name_list_sorted)
plt.ylabel('Sum of diagonal')
plt.show() 

            
'''         
# plot examples of confusion matrix with the best and worst classification
'''
# average the confusion matrix across subjects
ind_min = np.argmin(np.array(mean_acc_all))
mean_conf_mat_all = conf_mat_all[ind_min]
fig = plt.figure()
fig.suptitle('LR classification accuracy fetures: ' + str(feat_name_list[ind_min]))
plt.imshow(mean_conf_mat_all)
plt.colorbar()
plt.xticks(np.arange(6), categories)
plt.yticks(np.arange(6), categories)
plt.show()

ind_max = np.argmax(np.array(mean_acc_all))
mean_conf_mat_all = conf_mat_all[ind_max]
fig = plt.figure()
fig.suptitle('LR classification accuracy fetures: ' + str(feat_name_list[ind_max]))
plt.imshow(mean_conf_mat_all)
plt.colorbar()
plt.xticks(np.arange(6), categories)
plt.yticks(np.arange(6), categories)
plt.show()
