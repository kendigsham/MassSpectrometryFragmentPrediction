#import operator
import numpy as np
import h5py
##to draw molecules etc
from rdkit import Chem
##from rdkit.Chem import AllChem
##from rdkit.Chem import Draw
##from rdkit.Chem.Draw import IPythonConsole # Needed to show molecules drawing in notebook
##from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions # Only needed if modifying defaults
##
##from keras.models import Sequential, Model, load_model
#
## set the same random seed each time
#np.random.seed(seed=1)
#
import sys
sys.path.append('/home/kenny/linux/keras-molecules-master')
#
from molecules.model import MoleculeVAE, SimpleMoleculeVAE
from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, decode_smiles_from_indexes, load_dataset
#
data_file = '/home/kenny/linux/keras-molecules-master/pubchem_1m_char.h5'
#
model_file = '/home/kenny/linux/keras-molecules-master/pubchem_1m_val_loss_0.2770_val_acc_0.9798_MoleculeVAE.h5'
model = MoleculeVAE()
#%%



def load_charset(filename):
    h5f = h5py.File(filename, 'r')
    charset =  h5f['charset'][:]
    h5f.close()
    return charset

#%%

def to_one_hot_array(smile_str, charset, max_char=120):
    filtered = []
    for c in smile_str:
        if c in charset:
            filtered.append(c)
            if len(filtered) == max_char:
                break            

    charset_list = charset.tolist()
    one_hot_encoded_fn = lambda row: map(lambda x: one_hot_array(x, len(charset_list)),
                                                one_hot_index(row, charset_list))

    filtered_str = ''.join(filtered)
    filtered_str = filtered_str.ljust(max_char) # pad up to max_char
    filtered_arr = np.array(one_hot_encoded_fn(filtered_str))
    return filtered_arr

def get_input_arr(smiles_list, charset):
    input_arr = []
    for i in range(len(smiles_list)):
        smile = smiles_list[i]
        one_hot_encoded = to_one_hot_array(smile, charset)
        input_arr.append(one_hot_encoded)
    input_arr = np.array(input_arr)
    return input_arr

def get_input_array(smiles_list, charset):
    dictionary = {}
    for i in range(len(smiles_list)):
      smile = smiles_list[i]
      temp_one_hot = to_one_hot_array(smile, charset)
      dictionary[smile] = temp_one_hot
    return dictionary

def encode(model, input_array):
    x_latent = model.encoder.predict(input_array)
    return x_latent

def readfile():
    smilesnames = []
    with open('/home/kenny/linux/project/smilesnames.txt','r') as rr:
        for name in rr:
            smilesnames.append(name)
    return smilesnames

def get_smile_latent(model, input_array_dict):
  smile_latent_dict = {}
  for key,val in input_array_dict.iteritems():
    if smile_latent_dict.get(key) == None:
      # print type(val), val.shape
      tempval = val[np.newaxis,:,:]
      # print type(tempval), tempval.shape
      temp_latent = encode(model, tempval)
      # print type (temp_latent)
      smile_latent_dict [key] = temp_latent
  return smile_latent_dict

def get_smiles_string(smiles):
  canonicalised_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
  return canonicalised_smiles

def get_latent_dict(smileslist):
  # get charset from the 
  charset = load_charset(data_file)
  print type(charset)
  print charset.shape
  print charset

  model.load(charset, model_file)

  # reading from the file that contains all the smiles per line extracted from all
  # the massbank files
  # smilelist=readfile()

  canonicalised_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(str(x))) for x in smileslist]
  # to get a multi-dimensional array of to-hot-array
  input_array_dict = get_input_array(canonicalised_smiles, charset)
  # print type(input_array_dict)

  smile_latent_dict = get_smile_latent(model, input_array_dict)
  return smile_latent_dict




#%%

# for key,val in smile_latent_dict.iteritems():
#   print type(val), val.shape, val.dtype
