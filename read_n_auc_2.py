# import latent_dict
import cross_validate
import file_handler
import latent_n_class
from timeit import default_timer as timer
import json
import scipy.sparse as sp
import read_200_2

def filter_mol(num_of_mol,mz_value,mz_numofmol):
  tempnum = mz_numofmol.get(mz_value)
  if tempnum != None:
    if tempnum >= num_of_mol:
      return True, tempnum
    else:
      return False, 0
  else:
      return False, 0
    
def read_dict(filename):
  temp_dict ={}
  with open (filename, 'r') as ff:
    for line in ff:
      line_list = line.strip().split(',')
      str1 = line_list[0]
      str2 = line_list[1]
      if filename == 'mz_value_colindex.txt' or filename == 'mz_numofmol.txt':
        temp_dict[str1] = int(str2)
      elif filename =='rowindex_smile.txt':
        temp_dict[int(str1)] = str2
      else:
        temp_dict[str1] = str2
  return temp_dict

def read_files():
  gnps_class = sp.load_npz('gnps_class.npz')

  mz_valuecolindex=read_dict('mz_value_colindex.txt')
  rowfile=read_dict('row_file.txt')
  filesmile=read_dict('file_smile.txt')
  rowindex=read_dict('rowindex_smile.txt')
  mzmol=read_dict('mz_numofmol.txt')
    
  return gnps_class, mz_valuecolindex, rowfile, filesmile, rowindex, mzmol

def read_n_auc(ms_folder, smile_latent_csv, mzfile2save, resultfile2save, mol2filter, model_name):
  classmatrix, mz_value_colindex, row_file, file_smile, rowindex_smile, mz_numofmol = read_files()
#  classmatrix, mz_value_colindex, row_file, file_smile, rowindex_smile, mz_numofmol = read_200_2.get_class_vectors(ms_folder)
  print type(classmatrix), classmatrix.shape
  print len(mz_value_colindex)
  print len(row_file)
  print len(file_smile)
  print len(rowindex_smile)
  print len(mz_numofmol)
#  save_mz_numofmol(mz_numofmol,mzfile2save)
  smile_latent_dict = file_handler.load_latent_dict(smile_latent_csv)
  mz_value_done = file_handler.read_file(resultfile2save)
  mz_list = mz_value_colindex.keys()
  

  for mz_value in mz_list:
    temp_mz_value = mz_value_done.get(mz_value)
    if temp_mz_value ==None:
      boo , num_of_mol = filter_mol(mol2filter, mz_value, mz_numofmol)
      if boo:
        start = timer()
        latent_matrix, class_vector , num_of_positive = latent_n_class.get_latent_class(classmatrix,mz_value_colindex,mz_value,rowindex_smile, smile_latent_dict)
#        num_of_positive = latent_n_class.get_latent_class(classmatrix,mz_value_colindex,mz_value,rowindex_smile, smile_latent_dict)
        mean_auc = cross_validate.get_auc_score(model_name,latent_matrix, class_vector)
        print mean_auc
        file_handler.writetofile(resultfile2save, mean_auc,mz_value,num_of_mol, num_of_positive)

        end = timer()
        print (end - start), 'time for one mzasdasd'

read_n_auc('/home/kenny/linux/spectra_gnps', 'gnps_smiles_2_latent.csv', 'temp2mzfile.txt','testtemp2.txt', 5, 'rbf')
