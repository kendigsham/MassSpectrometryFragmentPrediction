import sklearn.model_selection as skm
from sklearn import svm
from sklearn.decomposition import PCA
import scipy.sparse as sp
import numpy as np
import operator
import random
from scipy.stats import randint as sp_randint
from collections import Counter
import pandas as pd
import csv

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
    for i in range(3):
        max20.append(varlist[i][0])
    return max20
        
def logcolormatrix(matrix,fragmentlist):
    colormatrix=matrix[:,fragmentlist]
    print(colormatrix.shape,type(colormatrix))
    return colormatrix
    
    
def getPCA(num_of_component,latent_reps):
    pca = PCA(n_components = num_of_component)
    transformed = pca.fit_transform(latent_reps)
    transformedPCA=transformed[:,0:num_of_component]
    #covariance = pca.get_covariance()
    #evr = pca.explained_variance_ratio_
    return transformedPCA
    
#%%
    
# to write the dictionary into a text file
def access_dict(ddd):
    def access_inner(ddd,ffff):
        for i,j in ddd.iteritems():
            ffff.write ('%s\n'%(i+','+str(j)))
    ffff=open('best_parameters.txt','w')
    for k in ddd:
        ffff.write('%s\n' % k)
        access_inner(ddd[k],ffff)
    ffff.close()
    
#%%
    
def save_file(result_dict,filename):
    result=pd.DataFrame(result_dict)
    result.to_csv()
    
#%%
# prelimary testing with 5 component, and the 3 target vectors with the most variation
def testing(pca):
    latent_reps=np.load('/home/kenny/linux/latent_reps.npy')
    binary=sp.load_npz('binarymtx.npz')
    length_binary=binary.shape[1]
    component_list=np.arange(2,100,5)
    component_list=component_list.tolist()
    component_list.append(100)
    random_binary=random.sample(range(0, length_binary), 3)
    para_dict={}
    for number in range(len(component_list)):
        for best3 in random_binary:
            tempbinary1 = binary.getcol(best3)
            tempbinary=tempbinary1.toarray()
            tempbinary=tempbinary.reshape(tempbinary1.shape[0],)
            print type(tempbinary), tempbinary.shape
            if pca=='pca':
                transformed=getPCA(component_list[number],latent_reps)
            elif pca=='nopca':
                num=component_list[number]
                templist=random.sample(range(100), num)
                transformed=latent_reps[:,templist]
            print type(transformed), transformed.shape
            parameters = {'C':sp_randint(1, 100),'gamma':sp_randint(1, 100),'kernel':['rbf','linear']}
            model=svm.SVC(probability=True)
#            shuffle = skm.ShuffleSplit(n_splits=5,test_size=0.3)
            random_search=skm.RandomizedSearchCV(estimator=model,param_distributions=parameters,n_iter=60)
            random_search.fit(transformed,tempbinary)
            parameters=random_search.best_params_
            name=str(component_list[number])+','+str(best3)
#            the best parameters of this particular testing condition are being stored
#            into a dicitonary of dictionaries
            para_dict[name]=parameters
    access_dict(para_dict)
            
    
testing('nopca')