#import numpy as np
import scipy.sparse as sp
import glob
from sklearn import preprocessing
from rdkit import Chem

row,col,data=[],[],[]
mz_value_colindex= {}
row_file = {}
file_smile ={}
rowindex_smile={}
mz_filename={}

col_index_count = 0


def getfilelist(filepath):
    filepath = filepath + '/*.ms'
    filelist= glob.glob(filepath)
#    filelist= glob.glob("/home/kenny/linux/spectra_massbank/*.ms")
#    filelist= glob.glob("/home/kenny/linux/temptemp/*.ms")
    return filelist

def checkfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def checkint(string):
    try:
        int(string)
        return True
    except ValueError:
        return False

# check and adjust the value of the m/z value, to the nearest 0.005
# get_original_class above and equal to 0.005 will be rounded down to 0.005
# everyhting below 0.005 will be rounded to 0.000
def adjust_value(bins):
  bins = bins*100
  bin_list=list(str(bins))
  string=[]
  for i in range(len(bin_list)):
    if bin_list[i]=='.':
      tempchar=bin_list[i+1]
      num =int(tempchar)
      if num <5:
        string=bin_list[:i-2]
        string.append('.')
        string.append(bin_list[i-2])
        string.append(bin_list[i-1])
        string.append('0')
      elif num >=5:
        string=bin_list[:i-2]
        string.append('.')
        string.append(bin_list[i-2])
        string.append(bin_list[i-1])
        string.append('5')        
  ss=map(str,string)
  mz_value = ''.join(ss)
  return mz_value


def findfragment (line):
    if(len(line)>2):
        words = line.split()
        if(checkint(words[0]) or checkfloat(words[0])):
            return float(words[0]), float(words[1])
        else:
            return 0,0

def findsmiles(line):
    words = line.split()
    if words[0] == '>smiles':
      return words[1]

def addfragment(bins,intensity,position):
    row.append(position)
    col.append(bins)
    data.append(intensity)

def get_col_index(mz_value):
    temp_value = mz_value
    col_index1 = 0
    if mz_value_colindex.get(temp_value) != None:
        colindex = mz_value_colindex.get(temp_value)
        col_index1 = colindex
    elif mz_value_colindex.get(temp_value) == None:
        global col_index_count
        col_index1 = col_index_count
        mz_value_colindex[temp_value] = col_index_count
        col_index_count +=1
    return col_index1

def openfile (filename,position):
    canonicalised_smiles=None
    temp_mz_filename = {}
    with open(filename,"r") as f:
        for line in f:
          if len(line)>2:
            line_trim=line.strip()
            if canonicalised_smiles ==None:
              smiles = findsmiles(line_trim)
              if smiles != None:
                canonicalised_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            elif canonicalised_smiles != None:
              bins, intensity = findfragment(line_trim)
              if bins !=0 and intensity !=0:
                mz_value=adjust_value(bins)
                if temp_mz_filename.get(mz_value) ==None:
                  temp_mz_filename[mz_value] = 1
                  if mz_filename.get(mz_value)==None:
                    mz_filename[mz_value] = [filename]
                  else:
                    mz_filename[mz_value].append(filename)
                temp_col_index = get_col_index(mz_value)
                addfragment(temp_col_index,intensity,position)
    file_smile[filename] = canonicalised_smiles
    rowindex_smile[position] = canonicalised_smiles

def makecscmatrix():
    cscmatrix = sp.csc_matrix((data,(row,col)))
    return cscmatrix

# making the matrix with 1 and 0, the input is the rescale1000 matrix
# trying to maintain the dimension of the binary so that it is the same as the 
# rescale matrix
def classmtx(mtx):
    binarizer = preprocessing.Binarizer(threshold = 0.0)
    classmtx = binarizer.transform(mtx)
    return classmtx

def get_class_vectors(filepath):
    # get the list of file from the directory
    listt = getfilelist(filepath)
    lengthlist=len(listt)
    for i in range(lengthlist):
        tempname=listt[i]
        openfile(tempname,i)
        row_file[str(i)] = tempname
#        make the matrix after get_original_class has been read in
    mtx=makecscmatrix()
#    print type(mtx), mtx.shape ,'========='
#    filename ='originalmtx.npz'
#    sp.save_npz(filename,mtx)
#     getting the class matrix
    classmatrix =classmtx(mtx)
#    print classmatrix.shape, type(classmatrix), '////////////'
#    filename3 = 'classmtx.npz'
#    sp.save_npz(filename3,classmatrix)
    return classmatrix, mz_value_colindex, row_file, file_smile, rowindex_smile, mz_filename
#    
#%%

classmatrix, mz_value_colindex, row_file, file_smile, rowindex_smile, mz_filename = get_class_vectors('/home/kenny/linux/spectra_gnps')
#
print len(mz_filename)
import pickle
with open('mz_filename_gnps.txt','w') as f1:
  pickle.dump(mz_filename,f1)


#classmatrix, mz_value_colindex, row_file, file_smile, rowindex_smile, mz_filename = get_class_vectors('/home/kenny/linux/temp2')
###    
#print len(row), len(col), len(data)
#print all(isinstance(n, int) for n in row)
#print all(isinstance(n, int) for n in col)
#print all(isinstance(n, float) for n in data)
#print 'mz_value', len(mz_value_colindex)
#print 'row_file', len(row_file)
#print 'file-smile', len(file_smile)
##
#print type(classmatrix), classmatrix.shape, 'classsssssss'
##%%
#import random
#for i in range (10):
#  num = random.randint(1, 80000)
#  lll = classmatrix.getcol(num)
#  print lll.nonzero()[0]