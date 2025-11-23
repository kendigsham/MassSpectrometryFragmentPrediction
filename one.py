import numpy as np
import scipy.sparse as sp
import glob
import operator
import math 
from sklearn.preprocessing import normalize  

row,col,data=[],[],[]
smilenposition=dict()

def getfilelist():
    filelist= glob.glob("/home/kenny/linux/spectra-massbank/*.ms")
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

def findfragment (line):
    if(len(line)>2):
        words = line.split()
        if(checkint(words[0]) or checkfloat(words[0])):
            #floor the value so they become bins
            return float(words[0]), float(words[1])
        else:
            return 0,0

def findsmiles(line):
    if (len(line) > 2):
        wordss = line.split()
        if (wordss[0] == ">smiles"):
            return wordss[1]
        else:
            return
    else:
        return

def addfragment(bins,intensity,position):
    row.append(position)
    col.append(bins)
    data.append(intensity)

def openfile (filename,position):
    with open(filename,"r") as f:
        for line in f:
            if (len(line)>2):
                smiles_string=findsmiles(line)
                bins, intensity = findfragment(line)
#                bins = math.floor(bins)
                if (smiles_string):
                    smilenposition[smiles_string]=position
                    print(smiles_string)
                elif(bins!=0 and intensity !=0):
                    addfragment(bins,intensity,position)
                else:
                    continue

def makecscmatrix():
    cscmatrix = sp.csc_matrix((data,(row,col)))
    return cscmatrix

def getvariance(mtx):
    fragmentvar=dict()
    coomatrix=mtx.tocoo()
    #return all the indices of columns that are not empty
    colindex = coomatrix.col
    colindex=colindex.tolist()
    unique = set(colindex)
    for i in unique:
#             get the whole column, it is still sparse matrix, convert to
#       array then further convert to list
        tempcol = mtx.getcol(i)
#        the total number of rows or data
        number = tempcol.shape[0]
        sqr = tempcol.copy()
        sqr.data **=2
        variance = sqr.sum() / number - tempcol.mean()**2
        fragmentvar[str(i)]=variance
    return fragmentvar
        

def findfragmentmax20(fragmentvar):
    max20=[]
    varlist=sorted(fragmentvar.items(),key=operator.itemgetter(1))
    varlist.reverse()
    for i in range(10):
        max20.append(varlist[i][0])
    return max20
        
def logcolormatrix(matrix,fragmentlist):
    colormatrix=matrix[:,fragmentlist]
    print(colormatrix.shape,type(colormatrix))
    return colormatrix
    
def writelist(max20):
    with open('bin_names.txt','w') as www:
        for item in max20:
            www.write("%s\n" % item)

def everything():
    # get the list of file from the directory
    list = getfilelist()
    lengthlist=len(list)
    print(lengthlist)
    for i in range(lengthlist):
        tempname=list[i]
        print(tempname)
        openfile(tempname,i)
#        make the matrix after everything has been read in
    mtx=makecscmatrix()
    sp.save_npz('original.npz',mtx)
#    print(mtx.shape,'!!!!!!!!!!!!!!!!!!!')

#    normalise the mtx, by column
    mtx2=normalize(mtx,norm='l2',axis=0)
    print(type(mtx2),mtx2.shape)
    print(mtx2)
    #    get the variance of the columns
    frag=getvariance(mtx)
    maxxx=findfragmentmax20(frag)
    maxxx.append(136)
    print maxxx
    writelist(maxxx)
    colormtx= logcolormatrix(mtx2,maxxx)
#    saving the matrix into a npz file
    sp.save_npz('colormatrix.npz',colormtx)
    
    

mmm=sp.load_npz('colormatrix.npz')
print(type(mmm),mmm.shape)