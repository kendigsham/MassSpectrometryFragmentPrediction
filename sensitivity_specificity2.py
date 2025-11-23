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
import p_values
import math
import pickle


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

# 
def save_sen_spec(new_sen,new_spec,smile_string):
      temp_list = [smile_string, new_sen, new_spec]
      smile_sen_spec.append(temp_list)
      smile_sensitivity[smile_string] = new_sen
      smile_specificity[smile_string] = new_spec


def sort_save_list(filename_to_save):
  header = ['smile_string','sensitivity','specificity']
#  temp_dataframe = sorted(smile_sen_spec, key = operator.itemgetter(1, 2),reverse = True)
  temp_dataframe = pd.DataFrame(smile_sen_spec,columns = header)
  global sorted_sen_spec
  sorted_sen_spec= temp_dataframe.sort_values(by=['sensitivity', 'specificity'], ascending = [0,0])
  sorted_sen_spec.to_csv(filename_to_save, index=False)
  
def get_predict_binary(smile_string,fitted_model):
  temp_latent_reps = smile_latent_dict_test.get(smile_string)
  latent_reps = np.array(temp_latent_reps).reshape(1,-1)
  predicted_value = fitted_model.predict(latent_reps)
#  print predicted_value
  if smile_binary_predict.get(smile_string) ==None:
    smile_binary_predict[smile_string] = list(predicted_value)
  else:
    smile_binary_predict[smile_string].append(predicted_value)
    
def get_true_binary(mz_value, smile_string):
      # if it doesnt exist in the mz_value to column index dictoinary, the true value must be zero
    # if it exists, it shall be retrieved from teh classmatrix of tehe testing data
    if mz_value_test.get(mz_value) == None or mz_value_test.get(mz_value) == 0 or math.isnan(mz_value_test.get(mz_value)):
      if smile_binary_true.get(smile_string) ==None:
        smile_binary_true[smile_string] = [0]
      else:
        smile_binary_true[smile_string].append(0)
      return 0
    else:
      colindex = mz_value_test.get(mz_value)
#      print colindex, 'colindex'
      rowindex = smilerow_test.get(smile_string)
#      print rowindex, 'rowindex'
      if smile_binary_true.get(smile_string) ==None:
        smile_binary_true[smile_string] = [classtest[rowindex,colindex]]
      else:
        smile_binary_true[smile_string].append(classtest[rowindex,colindex])
      return classtest[rowindex,colindex]
    
#print len(smile_binary_true.keys()), 'number of strings'
def get_sen_spec_values():
  for smile_string, true_vectors in smile_binary_true.iteritems():
    if true_vectors.count(1) >=1:
      p_values.get_confusion_matrix(smile_string, smile_binary_true, smile_binary_predict,smile_confusion)
      new_sen, new_spec , boo = p_values.sensitivity_specificity(smile_string,smile_confusion)
      if boo:
        save_sen_spec(new_sen,new_spec,smile_string)
    else:
      smile2no_sen_spec[smile_string] = 1

#%%
# iterate each mz_value (each column)
def everything(filename,num_of_mol,num_of_unique,auc_score,ranked_filename):
  mz_values = mz_value_list.get_mz_list(filename,num_of_mol,num_of_unique,auc_score)
  print len(mz_values), 'number of mz_values -------'
#  unique_smiles_all = get_unique_smiles()
#  for index in range(len(mz_values)):
#    mz_value = mz_values[index]
#    fitted_model = make_model(mz_value)
#    # iterate each smile string (each row)
#    for index2 in range(len(unique_smiles_all.keys())):
#      temp_unique = unique_smiles_all.keys()
#      smile_string = temp_unique[index2]
#      true_binary = get_true_binary(mz_value, smile_string)
#      get_predict_binary(smile_string,fitted_model)
#    #   making a list of true binary values for each mz_value (column)
#      if index2 ==0:
#        mzvalue_binarycolumn[mz_value] = [true_binary]
#      else:
#        mzvalue_binarycolumn[mz_value].append(true_binary)
#
#  get_sen_spec_values()
#  sort_save_list(ranked_filename)
      
    
#%%

everything('massbank_result2.csv', 20, 10, 0.7,'ranked_sen_spec_0.7massbank.csv')
#
  
#with open('smile_binary_true.txt','r') as f1:
#  smile_binary_true=pickle.load(f1)
#
#with open('smile_binary_predict.txt','r') as f2:
#  smile_binary_predict = pickle.load(f2)
#  
#with open('mzvalue_binarycolumn.txt','r') as f3:
#  mzvalue_binarycolumn = pickle.load(f3)
#  
#with open('smile_sensitivity.txt','r') as f4:
#  smile_sensitivity =pickle.load(f4)
#  
#with open('smile_specificity.txt','r') as f5:
#  smile_specificity = pickle.load(f5)
#  

#sen_row,spec_row = p_values.permutate_row(500,smile_binary_true,smile_binary_predict)
#p_values.save_lists(sen_row,spec_row,'sen_row_0.7massbank.txt','spec_row_0.7massbank.txt')
#print 'done permute rowaaaaaaaaaaaaaaaaaaa'
#
#sen_column,spec_column = p_values.permutate_column(500,smile_binary_true,smile_binary_predict,mzvalue_binarycolumn)
#print 'done permute columnnnnnnnnnnnnnnnnnnnnn'
#p_values.save_lists(sen_column,spec_column,'sen_column_0.7massbank.txt','spec_column_0.7massbank.txt')
#print 'saved+++++++++++++++++++++++'
#
#smile_sen_pvalues_row,smile_spec_pvalues_row = p_values.get_p_values(sen_row, spec_row, smile_sensitivity, smile_specificity)
#p_values.save_lists(smile_sen_pvalues_row,smile_spec_pvalues_row,'smile_sen_pvalues_0.7row.txt','smile_spec_pvalues_0.7row.txt')
#print 'done ========================='
#
#smile_sen_pvalues_column,smile_spec_pvalues_column = p_values.get_p_values(sen_column, spec_column, smile_sensitivity, smile_specificity)
#
#p_values.save_lists(smile_sen_pvalues_column,smile_spec_pvalues_column,'smile_sen_pvalues_0.7column.txt','smile_spec_pvalues_0.7column.txt')
#print 'done p values======================'
#
#
#
#with open ('ranked_sen_spec_0.7massbank.csv','r') as f:
#  list_of_list=[]
#  for line in f:
#    templine = line.strip().split(',')
#    list_of_list.append(templine)
#    
#list_of_list[0].append('p_value_row_sen')
#list_of_list[0].append('p_value_column_sen')
#list_of_list[0].append('p_value_row_spec')
#list_of_list[0].append('p_value_column_spec')

#with open('smile_sen_pvalues_0.7row.txt','r') as f1:
#  smile_sen_pvalues_row = pickle.load(f1)
#  
#with open('smile_spec_pvalues_0.7row.txt','r') as f2:
#  smile_spec_pvalues_row = pickle.load(f2)
#  
#with open('smile_sen_pvalues_0.7column.txt','r') as f3:
#  smile_sen_pvalues_column = pickle.load(f3)
#  
#with open('smile_spec_pvalues_0.7column.txt','r') as f4:
#  smile_spec_pvalues_column = pickle.load(f4)
  
#with open('sen_row_massbank.txt','r') as f5:
#  sen_row = pickle.load(f5)
#  
#with open('spec_row_massbank.txt','r') as f6:
#  spec_row = pickle.load(f6)
#  
#with open('sen_column_massbank.txt','r') as f7:
#  sen_column = pickle.load(f7)
#  
#with open('spec_column_massbank.txt','r') as f8:
#  spec_column = pickle.load(f8)

#for i in range(len(list_of_list)):
#  if i ==0:
#    continue
#  else:
#    templist = list_of_list[i]
#    smile_string = templist[0]
#    if smile_sen_pvalues_row.get(smile_string) != None:
#      p_sen_row = float(smile_sen_pvalues_row.get(smile_string))
#    else:
#      p_sen_row = '-'
#    if smile_sen_pvalues_column.get(smile_string) != None:
#      p_sen_column = float(smile_sen_pvalues_column.get(smile_string))
#    else:
#      p_sen_column = '-'
#    if smile_spec_pvalues_row.get(smile_string) != None:
#      p_spec_row = float(smile_spec_pvalues_row.get(smile_string))
#    else:
#      p_spec_row = '-'
#    if smile_spec_pvalues_column.get(smile_string) != None:
#      p_spec_column = float(smile_spec_pvalues_column.get(smile_string))
#    else:
#      p_spec_column ='-'
#    templist.append(p_sen_row)
#    templist.append(p_sen_column)
#    templist.append(p_spec_row)
#    templist.append(p_spec_column)
#    
#headers = list_of_list.pop(0)
#
#df = pd.DataFrame(list_of_list, columns=headers)
#
#df.to_csv('ranked_sen_spec_p_value_0.7massbank2gnps.csv',index=False)


#%%

#import pickle
#


#p_values.plot_histogram_top_smile(5,smile_sen_pvalues_row,smile_spec_pvalues_row,sen_row,spec_row,'row')
#p_values.plot_histogram_top_smile(5,smile_sen_pvalues_column,smile_spec_pvalues_column,sen_column,spec_column,'column')

#p_values.plot_histogram_sen_spec(smile_sen_pvalues_row,smile_spec_pvalues_row)
#p_values.plot_histogram_sen_spec(smile_sen_pvalues_column,smile_spec_pvalues_column)

