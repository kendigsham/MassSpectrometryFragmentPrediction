import operator
import numpy as np
import h5py
#to draw molecules etc
from rdkit import Chem
#from rdkit.Chem import AllChem
#from rdkit.Chem import Draw
#from rdkit.Chem.Draw import IPythonConsole # Needed to show molecules drawing in notebook
#from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions # Only needed if modifying defaults

from keras.models import Sequential, Model, load_model

# set the same random seed each time
np.random.seed(seed=1)

import sys
sys.path.append('/home/kenny/linux/keras-molecules-master')

from molecules.model import MoleculeVAE, SimpleMoleculeVAE
from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, decode_smiles_from_indexes, load_dataset

data_file = '/home/kenny/linux/keras-molecules-master/pubchem_1m_char.h5'

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

def encode(model, input_array):
    x_latent = model.encoder.predict(input_array)
    return x_latent

def readfile():
    smilesnames = []
    with open('smile355.txt','r') as rr:
        for name in rr:
            if len(name)>3:
              smilesnames.append(name)
    return smilesnames

def get_latent_reps(metadata, charset, model):    
    smiles = [m['smiles'] for m in metadata]
#    inchikeys = [m['InChIKey'] for m in metadata]
    canonicalised_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(str(x))) for x in smiles]
    input_array = get_input_arr(canonicalised_smiles, charset)
    latent_reps = encode(model, input_array)
    print type(latent_reps),latent_reps.shape
    return canonicalised_smiles, latent_reps
  
# calculate the variance for each mass-to-charge ratio by column
def getvariance(mtx):
    fragmentvar=dict()
    coomatrix=mtx.tocoo()
    #return all the indices of columns that are not empty
    colindex = coomatrix.col
    colindex=colindex.tolist()
    unique = set(colindex)
    unique_list=list(unique)
    for i in range(len(unique_list)):
#             get the whole column, it is still sparse matrix, convert to
#       array then further convert to list
        index=unique_list[i]
        tempcol = mtx.getcol(index)
#        the total number of rows or data
        number = tempcol.shape[0]
        sqr = tempcol.copy()
        sqr.data **=2
        variance = sqr.sum() / number - tempcol.mean()**2
        fragmentvar[str(i)]=variance
    return fragmentvar
        
# finding the maxixum 10 variance of the matrix
def findfragmentmax10(fragmentvar):
    max10=[]
    varlist=sorted(fragmentvar.items(),key=operator.itemgetter(1))
    varlist.reverse()
    for i in range(10):
        max10.append(varlist[i][0])
    return max10

#%%

# get charset from the 
charset = load_charset(data_file)
print type(charset)
print charset.shape

model.load(charset, model_file)

# reading from the file that contains all the smiles per line extracted from all
# the massbank files
smilelist=readfile()
print len(smilelist)
canonicalised_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(str(x))) for x in smilelist]
# to get a multi-dimensional array of to-hot-array
input_array = get_input_arr(canonicalised_smiles, charset)
print(input_array.shape,type(input_array))

# convert the array output input_array
latent_reps = encode(model, input_array)
print(latent_reps.shape,type(latent_reps))

np.save('latent_reps136.npy',latent_reps)



