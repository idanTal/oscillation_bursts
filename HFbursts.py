# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:40:26 2019

Classification of image category based on high-frequency activity in each 
electrode location

@author: Idan Tal
"""


import sys
print('Python:{}'.format(sys.version))

import pandas
import numpy as np
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



# read subject info
subjects_info = pandas.read_excel(r"H:\\HFbursts\\OneDrive_1_5-24-2019\\data\\vis_localizer_patients.xlsx")
subject_IDs = list(subjects_info.values[:,0])
# define image categories
categories = ['animals', 'patterns', 'people', 'places', 'tools', 'words']
conf_mat_all = np.zeros((len(subject_IDs),len(categories),len(categories)))
for subject_ind in range(0, len(subject_IDs)):
    sum_diag = 0
    # define individual subject dirs
    file_name = r"H:\\HFbursts\\OneDrive_1_5-24-2019\\data\\" + subject_IDs[subject_ind] + "\\matlabOut\\trialsTransients\\features_allChnNumTrlFeat_imresize.txt"
    save_dir_name = r"H:\\HFbursts\\OneDrive_1_5-24-2019\\data\\" + subject_IDs[subject_ind] + "\\matlabOut\\trialsTransients\\"
    fig_save_dir = r"H:\\HFbursts\\figures\\" + subject_IDs[subject_ind] + "\\trials\\"
    # load features
    dataset = pandas.read_csv(file_name)
    names = []
    for i in range(len(dataset.columns)-1):
        names.append("samp" + str(i))
    names.append("class")
    dataset = pandas.read_csv(file_name, names = names)
    
    for random_states in range(0, 100):
        
        # define training and test sets
        array = dataset.values
        X = array[:,0:len(dataset.columns)-1]
        Y = array[:,-1]
        validation_size = 0.2
        random_state= random_states
        print("random state seed = " + str(random_state))
        X_train,X_validation, Y_train, Y_validation = \
            model_selection.train_test_split(X,Y,test_size = validation_size, random_state = random_state)
            
        scoring = 'accuracy'
        
        # append models for comparison (training)
        models = []
        solver = 'sag'
        penalty = 'l2'
        tol = 0.001
        random_state_LR= 0
        C = 1
        models.append(('LR',LogisticRegression(random_state = random_state_LR, C = C, solver = solver, multi_class = 'multinomial', penalty = penalty, tol = tol)))
        models.append(('LDA',LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 'auto')))
        models.append(('KNN',KNeighborsClassifier(n_neighbors = 10, algorithm = 'auto')))
        models.append(('CART',DecisionTreeClassifier(criterion = 'entropy')))
        models.append(('NB',GaussianNB()))
        models.append(('SVM',SVC(kernel='linear')))
            
        results = []
        names = []
        
        # evaluate model performance 
        for name,model in models:
            kfold = model_selection.KFold(n_splits = 10)
            cv_results = model_selection.cross_val_score(model,X_train,Y_train,cv = kfold, scoring = scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s : %f(%f)"%(name,cv_results.mean(),cv_results.std())
            print(msg)    
        
        # =============================================================================
        # plot models accuracy (box)
        fig = plt.figure()
        fig.suptitle('Algorithm comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.ylabel('Accuracy')
        plt.show()  
        # =============================================================================
        
        # =============================================================================
        # evaluate KNN performance
        knn = KNeighborsClassifier()
        knn.fit(X_train,Y_train)
        predictions = knn.predict(X_validation)
        print(accuracy_score(Y_validation,predictions))
        print(confusion_matrix(Y_validation, predictions))
        print(classification_report(Y_validation,predictions))
        # plot confusion matrix
        fig = plt.figure()
        plt.imshow(confusion_matrix(Y_validation, predictions))
        plt.colorbar()
        plt.show()
        # =============================================================================
        
        # evaluate logistic regression performance
        LR = LogisticRegression(random_state = random_state, C = C, solver = solver, multi_class = 'multinomial', penalty = penalty, tol = tol)
        LR.fit(X_train, Y_train)
        predictions = LR.predict(X_validation)
        print(accuracy_score(Y_validation,predictions))
        print(confusion_matrix(Y_validation, predictions))
        # plot confusion matrix
        conf_mat = confusion_matrix(Y_validation, predictions)
        # normalize the confusion matrix to show probabilities
        norm_factor = conf_mat.sum(axis = 0)
        norm_conf_mat = conf_mat/norm_factor[None,:]
        sum_diag_temp = np.trace(norm_conf_mat)
        if sum_diag_temp > sum_diag:
            print("previous sum_diag = " + str(sum_diag) + " current sum_diag = " + str(sum_diag_temp))
            sum_diag = sum_diag_temp       
            fig = plt.figure()
            fig.suptitle('Logistic Regression classification accuracy')
            plt.imshow(norm_conf_mat)
            plt.colorbar()
            plt.xticks(np.arange(6), categories)
            plt.yticks(np.arange(6), categories)
            plt.show()
            plt.savefig(fig_save_dir + "LRclassification_" + subject_IDs[subject_ind] + "seed" + str(random_state) + ".png")
            conf_mat_all[subject_ind,:,:] = norm_conf_mat
            

        # =============================================================================
         # evaluate SVM performance
        svm = SVC(kernel = 'linear')
        svm.fit(X_train, Y_train)
        predictions = svm.predict(X_validation)
        print(accuracy_score(Y_validation, predictions))
        print(confusion_matrix(Y_validation, predictions))
        # plot confusion matrix
        conf_mat = confusion_matrix(Y_validation, predictions)
        norm_factor = conf_mat.sum(axis = 0)
        norm_conf_mat = conf_mat/norm_factor[None,:]
        fig = plt.figure()
        fig.suptitle('SVM classification accuracy')
        plt.imshow(norm_conf_mat)
        plt.colorbar()
        plt.xticks(np.arange(6), categories)
        plt.yticks(np.arange(6), categories)
        plt.show
        # =============================================================================
        
        # =============================================================================
        # evaluate LDA performance
        lda = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 'auto')
        lda.fit(X_train, Y_train)
        predictions = lda.predict(X_validation)
        print(accuracy_score(Y_validation, predictions))
        print(confusion_matrix(Y_validation, predictions))
        # plot confusion matrix
        conf_mat = confusion_matrix(Y_validation, predictions)
        norm_factor = conf_mat.sum(axis = 0)
        norm_conf_mat = conf_mat/norm_factor[None,:]
        fig = plt.figure()
        fig.suptitle('LDA classification accuracy')
        plt.imshow(norm_conf_mat)
        plt.colorbar()
        plt.xticks(np.arange(6), categories)
        plt.yticks(np.arange(6), categories)
        plt.show
         
        # =============================================================================
        
        # =============================================================================
        # write the weights of the logistic regression to a csv file
        np.savetxt(save_dir_name + "LRweights.csv", coef, delimiter=",")
        # =============================================================================
        plt.close('all')
        
np.save(r"H:\\HFbursts\\OneDrive_1_5-24-2019\\data\\meanConfMatAll", conf_mat_all)
