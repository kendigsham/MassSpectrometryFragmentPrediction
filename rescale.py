import numpy as np
import scipy.sparse as sp

#%%
# this is the matrix with all the bins and intensity , in CSC format
matrix1= sp.load_npz('rescale1000.npz')
matrix2= sp.load_npz('original.npz')

print matrix1.shape, type(matrix1), matrix1.getnnz
print matrix2.shape, type(matrix2), matrix2.getnnz



#%%
# get all the columns with more than or equal to 50 molecules
def col50(mtx):
    lll = []
    coomatrix=mtx.tocoo()
    #return all the indices of columns that are not empty
    colindex = coomatrix.col
    colindex=colindex.tolist()
    unique = set(colindex)
    for i in unique:
        num=mtx.getcol(i).count_nonzero()
        if num>=50:
            lll.append(i)
        else:
            continue
    return lll
    
matrix1= sp.load_npz('rescale1000.npz')
lll=col50(matrix1)

with open('col>50.txt','w') as www:
        for item in lll:
            www.write("%s\n" % item)

#%%
# 
def rescale(mtx):
    
    rrr = mtx.tocsr()
    maxcol = rrr.max(axis=1)
    maxlist = maxcol.toarray().tolist()

# changing to coo to find non zero column index
    coo = mtx.tocoo()
    print coo.shape
    
#    loop throught the list and do maths to each row
    numofrow=mtx.shape[0]  
#    change matrix to lil so items can be assigned
    lil_mtx=mtx.tolil()    
    for i in range(numofrow):
        maxnum=maxlist[i]
        
        rr = coo.getrow(i)
        aa=rr.tocoo()
#        get the column index with non zeros
        nonzeros= aa.col

        for j in nonzeros:
            lil_mtx[i,j]=lil_mtx[i,j]/maxnum*1000
    csc_mtx=lil_mtx.tocsc()
    return csc_mtx

mmm=rescale(matrix)
print type(mmm), mmm.shape

sp.save_npz('rescale1000.npz',mmm)

#%%



