import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read the csv file and split the columns into smaller matrix and store them into a dictionary
def read_csv(filename):
  df=pd.read_csv(filename, sep=',',header=None)
  matrix=df.as_matrix()
  dictionary={}
  index=0
  while index < matrix.shape[1]:
    templist=[index,index+1,index+2]
    tempname=str(index)
    tempmatrix = matrix[:,templist]
    dictionary[tempname]=tempmatrix
    index+=3
  return dictionary

def make_array(dictionary):
  key_list=dictionary.keys()
  print key_list
  temparray=[]
  for index in range (len(key_list)):
    extract=dictionary[key_list[index]][:,0]
    tall=extract.shape[0]
    extract=extract.reshape(tall,1)
    if index==0:
      name=key_list[index]
      length=dictionary[name].shape[0]
      temparray=np.zeros((length,1))
      print temparray.shape
      temparray=np.hstack((temparray,extract))
    else:
      temparray=np.hstack((temparray,extract))
  array = temparray[:,1:]
  return array


#%%
def plot_csv(array):
  plt.style.use('seaborn-deep')
  data=array[1:,:].astype(np.float)
  labels=array[0,:].tolist()
  #bins = np.linspace(-10, 10, 30)
  fig=plt.figure(figsize=(16,12))
  bins=np.linspace(0.42,0.575,30)
  plt.hist(data,bins=bins, alpha=0.5, label=labels)
  #plt.hist(auc1,bins=bins, alpha=0.5, label=str(component_list[0]))
  #plt.hist(auc2, alpha=0.5, label=str(component_list[1]))
  #plt.hist(auc3, alpha=0.5, label=str(component_list[2]))
  plt.xticks(np.linspace(0.41,0.58,20))
  plt.title('KNN Histogram with frequency of AUC score for different dimension of latent space')
  plt.legend(loc='upper right')
  plt.xlabel('AUC score')
  plt.ylabel('Frequency')
  plt.show()
  fig.savefig('knn_AUC_histogram.png')

#%%
lll=read_csv('knnresults_filter.csv')

aaa=make_array(lll)

plot_csv(aaa)