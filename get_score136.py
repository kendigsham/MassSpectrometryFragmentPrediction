import numpy as np
import scipy.sparse as sp
import smiles
import latent_reps 
import json
import cross_validate as cv
import os.path

#%%
import operator
import numpy as np
import h5py
#to draw molecules etc
from rdkit import Chem
np.random.seed(seed=1)

import sys
sys.path.append('/home/kenny/linux/keras-molecules-master')

from molecules.model import MoleculeVAE, SimpleMoleculeVAE
from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, decode_smiles_from_indexes, load_dataset

data_file = '/home/kenny/linux/keras-molecules-master/pubchem_1m_char.h5'

model_file = '/home/kenny/linux/keras-molecules-master/pubchem_1m_val_loss_0.2770_val_acc_0.9798_MoleculeVAE.h5'
model = MoleculeVAE()

#%%

count = []

def get_dict():
  with open('dictionary_136.txt','r') as rrr:
    bin_smiles = json.load(rrr)
    
  bin_names = bin_smiles.keys()
  return bin_smiles , bin_names

def seperate_list(smiles_list,bin_list):
  yeslist2 = list(bin_list)[:]
  nonlist = smiles_list
  for binn in bin_list:
    try:
      nonlist.remove(binn)
    except KeyError:
      yeslist2.remove(binn)
      count.append(binn)
  return nonlist, yeslist2

def get_latent(smiles_list):
  latent_matrix = latent_reps.get_latent_rep(smiles_list,data_file,model_file,model)
  return latent_matrix

def make_class_matrix(yeslist, nonlist):
  yes_mtx = np.ones((len(yeslist),1))
  non_mtx = np.zeros((len(nonlist),1))
  class_mtx = np.vstack((yes_mtx, non_mtx))
  return class_mtx

def latent_matrix(yeslist, nonlist):
  yesmatrix = get_latent(yeslist)
  nonmatrix = get_latent(nonlist)
  full_matrix = np.vstack((yesmatrix,nonmatrix))
  return full_matrix

def writetofile(result_file,score,binn,num_of_mol,num_of_unique):
  line = str(binn)  + ',' +str(score) +','+ str(num_of_mol) +','+ str(num_of_unique)
  with open (result_file,'a') as www:
    www.write(line+'\n')

def read_file(result_file):
    bin_name = ['1']
    if os.path.isfile(result_file):
        mode = 'r'
        with open(result_file,mode) as fff:
          for line in fff:
            templine = line.split(',')
            bin_name.append(templine[0])

        return bin_name
    else:
        mode = 'w'
        first_line = 'bin_name,auc_score,num_of_molecule,num_of_unique\n'
        with open(result_file,mode) as www:
            www.write(first_line)
        return bin_name

#def write_data(result_file, result_line):
#    with open(result_file,'a') as fff:
#        fff.write(result_line)

def get_scores(model_name,result_file):
  smiles_list = smiles.get_smile_list()
  smiles_list = set(smiles_list)
  
  print len(smiles_list),'========='
  bin_smile_dict , bin_name_list = get_dict()
  
  bin_name= read_file(result_file)
  print bin_name
  for binn in bin_name_list:
    if binn not in bin_name:
      print binn
      yeslist = bin_smile_dict.get(binn)
      num_of_mol = len(yeslist)
      print num_of_mol,'...........'
      yeslist = set(yeslist)
      num_of_unique =len(set(yeslist))
      print num_of_unique,'/////////'
      nonlist , yeslist2= seperate_list(smiles_list, yeslist)
      class_mtx = make_class_matrix(yeslist2,nonlist)
      print type(class_mtx), class_mtx.shape
      latent_mtx = latent_matrix(yeslist2,nonlist)
      print type(latent_mtx), latent_mtx.shape
      score = cv.get_auc_score(model_name, latent_mtx, class_mtx)
      writetofile(result_file,score,binn,num_of_mol,num_of_unique)
    
get_scores('rbf','score136.txt')
