from __future__ import division
from sklearn.metrics import confusion_matrix
import math
import random
import numpy as np
import pickle
from scipy import stats
import operator
import matplotlib.pyplot as plt
from chemspipy import ChemSpider
import requests

def get_confusion_matrix(smile_string,smile_binary_true, smile_binary_predict,smile_confusion):
    temp_true_vector = smile_binary_true.get(smile_string)
    temp_predict_vector = smile_binary_predict.get(smile_string)
    temp_confusion_matrix = confusion_matrix(temp_true_vector,temp_predict_vector)
#    print type(temp_confusion_matrix)
    smile_confusion[smile_string] = temp_confusion_matrix

#def sensitivity_specificity(smile_string,smile_confusion):
#    temp_confusion_matrix = smile_confusion.get(smile_string)
#    if temp_confusion_matrix.shape[0] == 2 and temp_confusion_matrix.shape[1] == 2:
#      sensitivity = temp_confusion_matrix[1,1]/(temp_confusion_matrix[1,1]+temp_confusion_matrix[1,0])
#      specificity = temp_confusion_matrix[0,0]/(temp_confusion_matrix[0,0]+temp_confusion_matrix[0,1])
#      
#      if sensitivity ==None or math.isnan(sensitivity):
#        new_sen = 0
#      else:
#        new_sen = sensitivity
#      if specificity ==None or math.isnan(specificity):
#        new_spec =0
#      else:
#        new_spec = specificity
#      return new_sen, new_spec
#    else:
#      return 0, 0
    
def sensitivity_specificity(smile_string,smile_confusion):
    temp_confusion_matrix = smile_confusion.get(smile_string)
    if temp_confusion_matrix.shape[0] == 2 and temp_confusion_matrix.shape[1] == 2:
      if temp_confusion_matrix[1,1] ==None or math.isnan(temp_confusion_matrix[1,1]):
        true_po = 0
      else:
        true_po = temp_confusion_matrix[1,1]
        
      if temp_confusion_matrix[1,0] ==None or math.isnan(temp_confusion_matrix[1,0]):
        false_neg =0
      else:
        false_neg = temp_confusion_matrix[1,0]
        
      if temp_confusion_matrix[0,0] ==None or math.isnan(temp_confusion_matrix[0,0]):
        true_neg =0
      else:
        true_neg = temp_confusion_matrix[0,0]
        
      if temp_confusion_matrix[0,1] ==None or math.isnan(temp_confusion_matrix[0,1]):
        false_po =0
      else:
        false_po = temp_confusion_matrix[0,1]
      sensitivity = true_po/float(true_po+false_neg)
      specificity = true_neg/float(true_neg+false_po)
      boo = True
      if sensitivity ==None or math.isnan(sensitivity):
        sensitivity = '-'
        boo = False
      if specificity ==None or math.isnan(specificity):
        specificity = '-'
        boo = False
      
      return sensitivity, specificity, boo
    else:
      sensitivity = 0
      specificity = 0
      boo = False
      return sensitivity, specificity, boo
    

#%%
#      
#def permutate_row(num_of_times,smile_binary_true,smile_binary_predict):
#  sensitivity_list={}
#  specificity_list={}
#  for time in range(num_of_times):
#    temp_smile_binary_true={}
#    temp_smile_confusion={}
#    temp_sen_values = []
#    temp_spec_values = []
#    for smile_string,binary_vector in smile_binary_true.iteritems():
##      print smile_string
##      print type(binary_vector)
#      random.shuffle(binary_vector)
##      print len(binary_vector)
#      temp_smile_binary_true[smile_string] = binary_vector
#      get_confusion_matrix(smile_string,temp_smile_binary_true, smile_binary_predict,temp_smile_confusion)
#      new_sen,new_spec = sensitivity_specificity(smile_string, temp_smile_confusion)
#      
#      temp_sen_values.append(new_sen)
#      temp_spec_values.append(new_spec)
#      
#    sensitivity_list [time] = np.mean(temp_sen_values)
#    specificity_list [time] = np.mean(temp_spec_values)
#  return sensitivity_list, specificity_list
#
#def permutate_column(num_of_times,smile_binary_true,smile_binary_predict,mzvalue_binary_vectors):
#  sensitivity_list={}
#  specificity_list={}
#  for time in range(num_of_times):
#    temp_smile_binary_true={}
#    temp_smile_confusion={}
#    temp_sen_values = []
#    temp_spec_values = []
##    for each column, the binary vector is shuffled 
#    for mz_value, binary_column in mzvalue_binary_vectors.iteritems():
#      random.shuffle(binary_column)
##      print type(binary_column)
##      from the shuffled vector, each smile_string is assigned a new number
#      for index in range(len(smile_binary_true.keys())):
#        smile_string_list = smile_binary_true.keys()
#        smile_string = smile_string_list[index]
#        if temp_smile_binary_true.get(smile_string) ==None:
#          temp_smile_binary_true[smile_string] = [binary_column[index]]
#        else:
#          temp_smile_binary_true[smile_string].append(binary_column[index])
#    for smile_string in smile_binary_true.keys():
#      get_confusion_matrix(smile_string,temp_smile_binary_true, smile_binary_predict,temp_smile_confusion)
#      new_sen,new_spec = sensitivity_specificity(smile_string, temp_smile_confusion)
#      
#      temp_sen_values.append(new_sen)
#      temp_spec_values.append(new_spec)
#      
#    sensitivity_list [time] = np.mean(temp_sen_values)
#    specificity_list [time] = np.mean(temp_spec_values)
#  return sensitivity_list, specificity_list
#
#def get_p_values(sensitivity_list,specificity_list,smile_sensitivity,smile_specificity):
#  sensitivity_mean = np.mean(sensitivity_list.values())
#  specificity_mean = np.mean(specificity_list.values())
#  
#  p_value_sen = stats.ttest_1samp(smile_sensitivity.values(),sensitivity_mean)
#  p_value_spec = stats.ttest_1samp(smile_specificity.values(),specificity_mean)
#  
#  return p_value_sen, p_value_spec

#%%

def permutate_row(num_of_times,smile_binary_true,smile_binary_predict):
  sensitivity_list={}
  specificity_list={}
  for time in range(num_of_times):
    temp_smile_binary_true={}
    temp_smile_confusion={}
    for smile_string,binary_vector in smile_binary_true.iteritems():
#      print smile_string
#      print type(binary_vector)
      random.shuffle(binary_vector)
#      print len(binary_vector)
      temp_smile_binary_true[smile_string] = binary_vector
      get_confusion_matrix(smile_string,temp_smile_binary_true, smile_binary_predict,temp_smile_confusion)
      new_sen,new_spec ,boo= sensitivity_specificity(smile_string, temp_smile_confusion)
      
      sensitivity_list.setdefault(smile_string,[]).append(new_sen)
      specificity_list.setdefault(smile_string,[]).append(new_spec)
        
  return sensitivity_list, specificity_list

def permutate_column(num_of_times,smile_binary_true,smile_binary_predict,mzvalue_binary_vectors):
  sensitivity_list={}
  specificity_list={}
  for time in range(num_of_times):
    temp_smile_binary_true={}
    temp_smile_confusion={}
#    for each column, the binary vector is shuffled 
    for mz_value, binary_column in mzvalue_binary_vectors.iteritems():
      random.shuffle(binary_column)
#      print type(binary_column)
#      from the shuffled vector, each smile_string is assigned a new number
      for index in range(len(smile_binary_true.keys())):
        smile_string_list = smile_binary_true.keys()
        smile_string = smile_string_list[index]
        if temp_smile_binary_true.get(smile_string) ==None:
          temp_smile_binary_true[smile_string] = [binary_column[index]]
        else:
          temp_smile_binary_true[smile_string].append(binary_column[index])
    for smile_string in smile_binary_true.keys():
      get_confusion_matrix(smile_string,temp_smile_binary_true, smile_binary_predict,temp_smile_confusion)
      new_sen,new_spec ,boo= sensitivity_specificity(smile_string, temp_smile_confusion)
      
      sensitivity_list.setdefault(smile_string,[]).append(new_sen)
      specificity_list.setdefault(smile_string,[]).append(new_spec)
      
  return sensitivity_list, specificity_list

# counting the proportion of sen/spec score that are larger or equal to the true scores
def get_p_values(sensitivity_list,specificity_list,smile_sensitivity,smile_specificity):
  smile_sen_pvalues={}
  smile_spec_pvalues={}
  for smile_string, sen_list in sensitivity_list.iteritems():
    true_sen = smile_sensitivity.get(smile_string)
    temp_num = sum(i >= true_sen for i in sen_list)
    p_value = temp_num/len(sen_list)
    smile_sen_pvalues[smile_string] = float(p_value)
  for smile_string, spec_list in specificity_list.iteritems():
    true_spec = smile_specificity.get(smile_string)
    temp_num = sum(i >= true_spec for i in sen_list)
    p_value = temp_num/len(spec_list)
    smile_spec_pvalues[smile_string] = float(p_value)
  return smile_sen_pvalues,smile_spec_pvalues

def save_lists(sensitivity_list, specificity_list, filename1, filename2):
  with open(filename1,'w') as f1:
    pickle.dump(sensitivity_list,f1)
  with open(filename2,'w') as f2:
    pickle.dump(specificity_list,f2)

def get_molecule_name(smile_string):
  cs = ChemSpider('c3f897f1-3a50-4288-8eb8-8deb9265be3f')
  for result in cs.search(smile_string):
    if result != None:
      common_name = result.common_name
      return common_name

def make_graph(index,datalist,row_columm,sen_spec):
  reshaped_data = np.array(datalist).reshape(-1,)
  plt.figure(figsize=(8,6))
  n_bins = 100
  plt.hist(reshaped_data, bins=n_bins, label='P_values')
  plt.legend(loc='right')
  filename = sen_spec +' P_values histogram molecule no. '+ str(index+1)+ row_columm
  plt.title(filename)
  plt.xlabel('P_values')
  plt.ylabel('Frequency of P_values')
  filename+='.png'
  plt.savefig(filename)
  plt.show()

# plot the histogram grpahs of the smiles with the lowest p_values
def plot_histogram_top_smile(num_of_top,smile_sen_pvalues,smile_spec_pvalues,sensitivity_list,specificity_list,row_column):
  sorted_sen = sorted(smile_sen_pvalues.iteritems(), key=operator.itemgetter(1))[:num_of_top]
  sorted_spec = sorted(smile_spec_pvalues.iteritems(), key=operator.itemgetter(1))[:num_of_top]
  for index1 in range(num_of_top):
    smile_string = sorted_sen[index1][0]
    print sorted_sen[index1][1]
    data_list = sensitivity_list.get(smile_string)
    print np.mean(data_list)
    print len(data_list), 'data list length'
#    make_graph(index1,data_list,row_column,'sensitivity')
  for index2 in range(num_of_top):
    smile_string = sorted_spec[index2][0]
    print sorted_spec[index2][1]
    data_list = specificity_list.get(smile_string)
    print np.mean(data_list)
    print len(data_list), ' data list length'
#    make_graph(index2,data_list,row_column,'specificity')

# plot graphs of all sensitivity and specificity
def plot_histogram_sen_spec(smile_sen_pvalues,smile_spec_pvalues,row_column):
  def plot_graph(datalist,title,row_column):
    reshaped_data = np.array(datalist).reshape(-1,)
    plt.figure(figsize=(8,6))
    n_bins = 50
    plt.hist(reshaped_data, bins=n_bins, label='P_values')
    plt.legend(loc='right')
    filename = row_column +'_'+ title +' P_values histogram of all molecules'
#    plt.xlim(1, 0)
    plt.axvline(0.05, color='b', linestyle='dashed', linewidth=2)
    plt.title(filename)
    plt.xlabel('P_values')
    plt.ylabel('Frequency of P_values')
    filename+='.png'
    plt.savefig(filename)
    plt.show()
  sen_list = smile_sen_pvalues.values()
  spec_list = smile_spec_pvalues.values()
  plot_graph(sen_list,'Sensitivity',row_column)
  plot_graph(spec_list,'Specificity',row_column)

