from __future__ import division
import read_massbank
import read_gnps
import mz_value_list
from sklearn import svm
#from sklearn.metrics import confusion_matrix
import file_handler
import latent_n_class
import numpy as np
import pandas as pd
import sklearn.model_selection as skm
import sklearn.metrics as skmetrics
import os

#%%


#using trained data with massbank on gnps data

classtrain, mz_value_train, smilerow_train,rowindex_train, unique_smiles_train = read_massbank.get_class_vectors()

classtest, mz_value_test, smilerow_test,rowindex_test, unique_smiles_test = read_gnps.get_class_vectors()

smile_latent_dict_train = file_handler.load_latent_dict('massbank_smiles_2_latent.csv')

smile_latent_dict_test = file_handler.load_latent_dict('gnps_smiles_2_latent.csv')

#%%
# using trained partial data with gnps on massbank data

#classtrain, mz_value_train, smilerow_train,rowindex_train, unique_smiles_train = read_gnps.get_class_vectors()
#
#classtest, mz_value_test, smilerow_test,rowindex_test, unique_smiles_test = read_massbank.get_class_vectors()
#
#smile_latent_dict_train = file_handler.load_latent_dict('gnps_smiles_2_latent.csv')
#
#smile_latent_dict_test = file_handler.load_latent_dict('massbank_smiles_2_latent.csv')
##  


#%%

smile_binary_true ={}
smile_binary_predict ={}
smile_confusion = {}
smile_sen_spec=[]
smile_sensitivity={}
smile_specificity={}
sorted_sen_spec=[]
# for column permutation
mzvalue_binarycolumn ={}
smile2no_sen_spec={}

#%%
        
# the input is the latent_vectors(2d) and the class vector with each column being the class label
def make_fitted_model(latent_reps, class_vector):
    model = svm.SVC(kernel='rbf',probability=True,C=0.1,gamma=10,class_weight='balanced')
    fitted_model= model.fit(latent_reps,class_vector)
    return fitted_model

#  get a list of smiles that were not used for trainning and return a dictionary
def get_unique_smiles():
    unique_smiles = {}
    for key in unique_smiles_test.keys():
        if unique_smiles_train.get(key) == None:
            unique_smiles[key] = 1
    return unique_smiles

# input one mz_value at a time to get one model, for mz_value with high auc sore
def make_model(mz_value):
    latent_matrix, class_vector = latent_n_class.get_latent_class(classtrain,mz_value_train,mz_value,rowindex_train, smile_latent_dict_train)
    fitted_model = make_fitted_model(latent_matrix, class_vector)
    return fitted_model

# the input is the latent_vectors(2d) and the class vector with each column being the class label
def svm_model(pca2d, classvector, fitted_model):
#    using the shuffle split
    stratified = skm.StratifiedKFold(n_splits=5)
    score_list=[]
#    this loop will be iterated 5 times, cuz the number of split is number from above
    for train, test in stratified.split(pca2d,classvector):
        try:
            predict= fitted_model.fit(pca2d[train],classvector[train])
        except ValueError:
            auc_score = 0.0
            # print auc_score
            score_list.append(auc_score)
            continue
#        if the class labels have only one class, then the auc-score ill be set to 0.5
        if predict.classes_.shape[0] <2:
            auc_score = 0.0
            # print auc_score
            score_list.append(auc_score)
        else:
#            tempsparse= sp.csr_matrix(pca2d[test])
#            predict=model.predict_proba(tempsparse)
            predict = fitted_model.predict_proba(pca2d[test])
    #        only input the predicted probability of the positive classes
            try:
#     sometimes the positive class is too few, its not possible get the auc score
#     so this is to catch that error and set auc score to 0.5
                auc_score = skmetrics.roc_auc_score(classvector[test],predict[:,1])
            except ValueError:
                auc_score = 0.0
            score_list.append(auc_score)
    return score_list


# make score list with all the auc score of all the bins
def score_list(classes, fitted_model , latent_rep):
    length=classes.shape[0]
    tempcol=classes.reshape(length,)
    temp_score=svm_model(latent_rep, tempcol,fitted_model)
    score_list=temp_score
    return score_list
    

def get_auc_score(latent_rep,classes,fitted_model):
    scorelist =score_list(classes,fitted_model,latent_rep)
#    print type(scorelist), len(scorelist), 'scorelist'
    mean_auc = np.mean(scorelist)
    return mean_auc
    
def writetofile(result_file,score,binn):
  line = str(binn)  + ',' +str(score)
  with open (result_file,'a') as www:
    www.write(line+'\n')

def make_file(result_file):
        mode = 'w'
        first_line = 'm/z_value,auc_score\n'
        with open(result_file,mode) as www:
            www.write(first_line)
        

#%%
# iterate each mz_value (each column)
def get_auc_score_test(filename,num_of_mol,num_of_unique,auc_score,filename2save):
  make_file(filename2save)
  mz_values = mz_value_list.get_mz_list(filename,num_of_mol,num_of_unique,auc_score)
  print len(mz_values), 'number of mz_values -------'
  for mz_value in mz_values:
    if mz_value_test.get(mz_value) !=None:
      print 'mz_value', mz_value
      fitted_model = make_model(mz_value)
      latent_matrix, class_vector = latent_n_class.get_latent_class(classtest,mz_value_test,mz_value,rowindex_test,smile_latent_dict_test)
      auc_score = get_auc_score(latent_matrix, class_vector, fitted_model)
      print auc_score
      writetofile(filename2save, auc_score, mz_value)


get_auc_score_test('massbank_result2.csv',20,10,0.7,'auc_massbank2gnps.txt')