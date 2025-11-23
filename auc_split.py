import sys
import cross_validate
import file_handler
import latent_n_class
from timeit import default_timer as timer
import scipy.sparse as sp
import pickle

def filter_mol(num_of_mol,mz_value,mz_numofmol):
  tempnum = mz_numofmol.get(mz_value)
  if tempnum != None:
    if tempnum >= num_of_mol:
      return True, tempnum
    else:
      return False, 0
  else:
      return False, 0

def load_list():
  with open('list_of_lists.txt','r') as ff:
    list_of_lists = pickle.load(ff)
  return list_of_lists
    
def read_dict(filename):
  temp_dict ={}
  with open (filename, 'r') as ff:
    for line in ff:
      line_list = line.strip().split(',')
      str1 = line_list[0]
      str2 = line_list[1]
      if filename == 'mz_value_colindex.txt' or filename == 'mz_numofmol.txt':
        temp_dict[str1] = int(str2)
      elif filename =='rowindex_smile.txt' or filename == 'tasknum_mzvalue.txt':
        temp_dict[int(str1)] = str2
      else:
        temp_dict[str1] = str2
  return temp_dict

def make_mz_list(chunk_index_list,tasknum_mzvalue):
  mz_list = []
  for i in chunk_index_list:
    mz_list.append(tasknum_mzvalue.get(i))
  return mz_list

def read_files():
  gnps_class = sp.load_npz('gnps_class.npz')

  mz_valuecolindex=read_dict('mz_value_colindex.txt')
  rowfile=read_dict('row_file.txt')
  filesmile=read_dict('file_smile.txt')
  rowindex=read_dict('rowindex_smile.txt')
  mzmol=read_dict('mz_numofmol.txt')
  tasknum_mzvalue = read_dict('tasknum_mzvalue.txt')
    
  return gnps_class, mz_valuecolindex, rowfile, filesmile, rowindex, mzmol, tasknum_mzvalue

def read_n_auc( smile_latent_csv, mol2filter, chunk_num):
  model_name = 'rbf'
#  new_done_name = './results/' +str(chunk_num) + '_result_done.txt'
#  resultfile2save = './results/' +str(chunk_num) + '_result.txt'
  new_done_name = str(chunk_num) + '_result_done.txt'
  resultfile2save = str(chunk_num) + '_result.txt'
  
  if file_handler.check_file(new_done_name):
    print file_handler.check_file(new_done_name)
    return
  
  classmatrix, mz_value_colindex, row_file, file_smile, rowindex_smile, mz_numofmol, tasknum_mzvalue = read_files()

#  print type(classmatrix), classmatrix.shape
#  print len(mz_value_colindex)
#  print len(row_file)
#  print len(file_smile)
#  print len(rowindex_smile)
#  print len(mz_numofmol)
#  print len(tasknum_mzvalue)

  smile_latent_dict = file_handler.load_latent_dict(smile_latent_csv)
  mz_value_done = file_handler.read_file(resultfile2save)
  list_of_lists = load_list()
  chunk_index_list = list_of_lists[chunk_num]
#  print chunk_index_list
  mz_list = make_mz_list(chunk_index_list,tasknum_mzvalue)
#  print len(mz_list)
  
  for index in range(len(mz_list)):
    mz_value = mz_list[index]
    temp_mz_value = mz_value_done.get(mz_value)
#    if temp_mz_value ==None:
#      boo , num_of_mol = filter_mol(mol2filter, mz_value, mz_numofmol)
#      if boo:
#        start = timer()
#        latent_matrix, class_vector , num_of_positive = latent_n_class.get_latent_class(classmatrix,mz_value_colindex,mz_value,rowindex_smile, smile_latent_dict)
#        mean_auc = cross_validate.get_auc_score(model_name,latent_matrix, class_vector)
#        print mean_auc
#        file_handler.writetofile(resultfile2save, mean_auc,mz_value,num_of_mol, num_of_positive)
#        end = timer()
#        print 'time needed ----', end-start
        
    if index == (len(mz_list)-1):
      file_handler.rename(resultfile2save,new_done_name)

#temp_number = sys.argv[1]
#number = int(temp_number) -1


read_n_auc('gnps_smiles_2_latent.csv', 5, 2)
