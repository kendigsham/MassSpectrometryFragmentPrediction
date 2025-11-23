import numpy as np
import pandas as pd

# read the csv file and split the columns into smaller matrix and store them into a dictionary
def read_csv(filename):
  df=pd.read_csv(filename, sep=',',header=None)
  matrix=df.as_matrix()
  dictionary={}
    
  index=1
  while index < matrix.shape[1]:
    templist=[index,index+1,index+2]
    tempname=str(index)
    tempmatrix = matrix[:,templist]
    dictionary[tempname]=tempmatrix
    index+=3
  return dictionary

# check each row for the value of num_of molecules
def check_mol(matrix,num_of_mol):
  row_list=[0]
  for i in range (1,matrix.shape[0]):
    temprow = matrix[i,:]
    templist=temprow.tolist()
    lastindex=matrix.shape[1]-1
    number=int(templist[lastindex])
    if number > num_of_mol:
      row_list.append(i)
  final_matrix=matrix[row_list,:]
  return final_matrix

# filter those rows with molecules lower than a certain value
def filter_matrix(input_dictionary,num_of_mol):
  dictionary={}
  for key,value in input_dictionary.items():
    print value.shape
    tempmatrix=check_mol(value,num_of_mol)
    dictionary[key]=tempmatrix
  return dictionary

# combine all the array back into one
def combine_array(dictionary):
  key_list=dictionary.keys()
  temparray=[]
  index=0
  while index < len(key_list):
    if index==0:
      name=key_list[index]
      length=dictionary[name].shape[0]
      temparray=np.zeros((length,1))
      temparray=np.hstack((temparray,dictionary[key_list[index]]))
    else:
      temparray=np.hstack((temparray,dictionary[key_list[index]]))
    index+=1
  array = temparray[:,1:]
  return array

def write_file(filename,mol_num):
  ddd=read_csv(filename)
  kkk=filter_matrix(ddd,mol_num)
  array=combine_array(kkk)
  qqq=pd.DataFrame(array)
  qqq.to_csv('knnresults_filter.csv',header=False,index=False)


#%%
#print array
print array.dtype
#
qqq=pd.DataFrame(array)
qqq.to_csv('knnresults_filter.csv',header=False,index=False)


ddd=read_csv('knnresults.csv')
print ddd.keys()
kkk=filter_matrix(ddd,50)
print type(kkk)
print kkk.keys()
for key,value in kkk.items():
    print value.shape
    print value
array=combine_array(kkk)
print type(array), array.shape
#%%
#print array
print array.dtype
#
qqq=pd.DataFrame(array)
qqq.to_csv('knnresults_filter.csv',header=False,index=False)

  