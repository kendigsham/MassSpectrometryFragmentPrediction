import numpy as np

# using a mz_value to get the column index from the classmatrix
def get_nonzero_rowindex(classmatrix,mz_value_colindex,mz_value):
#  nonzero_rowindex = {}
#  print mz_value,'mz_value'
  colindex = mz_value_colindex.get(mz_value)
#  print colindex, 'colindex'
  tempcol = classmatrix.getcol(colindex)
  temp_list = tempcol.nonzero()[0]
#  print len(temp_list), 'nonzero_list'
  nonzero_rowindex = temp_list
  return nonzero_rowindex

# 
def get_nonzero_smile(nonzero_rowindex,rowindex_smile):
#  print len(nonzero_rowindex), 'nonzero_rowindex'
#  print len(rowindex_smile), 'rowindex_smile'
  
  negative_smile = list(set(rowindex_smile.values()))
  positive_smile = {}
  for index in nonzero_rowindex:
    smile_string=rowindex_smile.get(index)
#    print smile_string
    if smile_string !=None:
      if positive_smile.get(smile_string) ==None:
        negative_smile.remove(smile_string)
        positive_smile[smile_string] = 1
    else:
      print 'no such smile','.....................'
  return positive_smile.keys(), negative_smile

def make_latent_matrix(smile_latent_dict, positive_smile, negative_smile):
  smile_list = []
  for smile in positive_smile:
    temp_smile = smile_latent_dict.get(smile)
    if temp_smile != None:
      smile_list.append(temp_smile)
    else:
      print 'no such latent, positive'
  for smile in negative_smile:
    temp_smile2 = smile_latent_dict.get(smile)
    if temp_smile2 != None:
      smile_list.append(temp_smile2)
    else:
      print 'no such latent, negative'
  latent_matrix = np.vstack(smile_list)

  return latent_matrix

def make_class_vector(positive_smile, negative_smile):
  positive_vector = np.ones((len(positive_smile),1))
  negative_vector = np.zeros((len(negative_smile),1))
  class_vector = np.vstack((positive_vector, negative_vector))
  class_vector = class_vector.reshape(class_vector.shape[0],)

  return class_vector

def get_latent_class(classmatrix,mz_value_colindex, mz_value, rowindex_smile, smile_latent_dict):
  nonzero_rowindex = get_nonzero_rowindex(classmatrix, mz_value_colindex, mz_value)
  positive_smile, negative_smile = get_nonzero_smile(nonzero_rowindex, rowindex_smile)
#  print len(positive_smile), len(negative_smile)
  latent_matrix = make_latent_matrix(smile_latent_dict, positive_smile, negative_smile)
  make_latent_matrix(smile_latent_dict, positive_smile, negative_smile)
  class_vector = make_class_vector(positive_smile, negative_smile)
  
#  num_of_positive = len(positive_smile)
  
  return latent_matrix, class_vector
#  return num_of_positive

