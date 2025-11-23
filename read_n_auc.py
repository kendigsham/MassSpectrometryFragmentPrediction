# import latent_dict
import cross_validate
import read_200_2
import file_handler
import latent_n_class
from timeit import default_timer as timer
import json
import sys
#%%

#def filter_mol(num_of_mol,mz_value,mz_filename):
#  templist = mz_filename.get(mz_value)
#  if templist != None:
#    num_of_mol = len(templist)
#    if num_of_mol >= num_of_mol:
#      return True, num_of_mol
#  else:
#      return False, 0
    
def filter_mol(num_of_mol,mz_value,mz_filename):
  tempnum = mz_filename.get(mz_value)
  if tempnum != None:
    if tempnum >= num_of_mol:
      return True, tempnum
    else:
      return False, 0
  else:
      return False, 0
    
def save_mz_filename(mz_filename,filename2save):
  with open(filename2save,'wb') as fff:
    json.dump(mz_filename, fff)


#%%

#
#classmatrix, mz_value_colindex, row_file, file_smile, rowindex_smile, mz_filename = read_200.get_class_vectors('/home/kenny/linux/temp2')
#print type(classmatrix), classmatrix.shape
#print len(mz_value_colindex)
#print len(row_file)
#print len(file_smile)
#print len(rowindex_smile)
#print len(mz_filename)
##
#
#
#smile_list = file_smile.values()
#print len(smile_list), 'smile list length'
#
#smile_latent_dict = latent_dict.get_latent_dict(smile_list)
#file_handler.save_latent_dict('temp2_smiles_2_latent.csv',smile_latent_dict)
#print len(smile_latent_dict), 'length=========='
##print smile_latent_dict.keys()
#
#for i in smile_list:
#  print smile_latent_dict.get(i).shape
#
#for key ,val in smile_latent_dict.iteritems():
#  print type(key),key
#  print type(val)
#%%

def read_n_auc(ms_folder, smile_latent_csv, mzfile2save, resultfile2save, mol2filter, model_name):
  classmatrix, mz_value_colindex, row_file, file_smile, rowindex_smile, mz_filename = read_200_2.get_class_vectors(ms_folder)
  print type(classmatrix), classmatrix.shape
  print len(mz_value_colindex)
  print len(row_file)
  print len(file_smile)
  print len(rowindex_smile)
  print len(mz_filename)
#  save_mz_filename(mz_filename,mzfile2save)
  smile_latent_dict = file_handler.load_latent_dict(smile_latent_csv)
  mz_value_done = file_handler.read_file(resultfile2save)
  mz_list = mz_value_colindex.keys()
  start = timer()
  count =1 
  for mz_value in mz_list:
    temp_mz_value = mz_value_done.get(mz_value)
    if temp_mz_value ==None:
      boo , num_of_mol = filter_mol(mol2filter, mz_value, mz_filename)
      if boo:
        latent_matrix, class_vector , num_of_positive = latent_n_class.get_latent_class(classmatrix,mz_value_colindex,mz_value,rowindex_smile, smile_latent_dict)
        mean_auc = cross_validate.get_auc_score(model_name,latent_matrix, class_vector)
        print mean_auc
#        file_handler.writetofile(resultfile2save, mean_auc,mz_value,num_of_mol, num_of_positive)
    if count == 1:
      break
  end = timer()
  print end - start, 'time for one mz'
#%%

#args_list = sys.argv
#
#ms_folder = sys.argv[1]
#smile_latent_csv = sys.argv[2]
#mzfile2save = sys.argv[3]
#resultfile2save = sys.argv[4]
#mol2filter = int(sys.argv[5])
#model_name = sys.argv[6]
#
#print 'the arguments put in are: ', ms_folder, smile_latent_csv, mzfile2save, resultfile2save, mol2filter, model_name
#
#read_n_auc(ms_folder, smile_latent_csv, mzfile2save, resultfile2save, mol2filter, model_name)

#%%

read_n_auc('/home/kenny/linux/spectra_gnps', 'gnps_smiles_2_latent.csv', 'temp2mzfile.txt','testtemp2.txt', 2, 'rbf')
