import scipy.sparse as sp
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
    
def findfragmentmax10(fragmentvar):
    max10=[]
    varlist=sorted(fragmentvar.items(),key=operator.itemgetter(1))
    varlist.reverse()
    for i in range(10):
        max10.append(varlist[i][0])
    return max10

def getcol(index,matrix):
    column = matrix.getcol(index)
#     print type(column) , column.shape
    col = column.toarray()
#     print type(col), col.shape
    singlecol = col.T
# the return is a list
    return singlecol 
    
def getPCA(num_of_component,latent_reps):
    pca = PCA(n_components = num_of_component)
    transformed = pca.fit_transform(latent_reps)
    transformedPCA=transformed[:,0:num_of_component]
    #covariance = pca.get_covariance()
    #evr = pca.explained_variance_ratio_
    return transformedPCA
            
def plot():            
    matrix= sp.load_npz('rescale.npz')
    latent_reps=np.load('latent_reps292.npy')
    variance= getvariance(matrix)
    max10=findfragmentmax10(variance)
    latent_dim = latent_reps.shape[1]
    transformed=getPCA(2,latent_reps)
    for i in range(len(max10)):
        index=int(max10[i])
        singlecol=getcol(index,matrix)
        plt.figure(figsize=(10,8))
#        norm = matplotlib.colors.Normalize(vmin = np.min(singlecol), vmax = np.max(singlecol), clip = False)
        plt.scatter(transformed[:,0],transformed[:,1],c=singlecol,cmap='seismic',marker='o',s=40,alpha=0.4) 
        title = 'PCA - Latent Dim %d, bin = %d' % (latent_dim, int(max10[i]))
        plt.title(title)
        plt.colorbar()
        title+='.png'
        plt.savefig(title)
        plt.show()        

plot()