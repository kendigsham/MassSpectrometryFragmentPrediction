import scipy.sparse as sp
import numpy as np
from sklearn import svm
from sklearn import neighbors
import sklearn.model_selection as skm
import sklearn.metrics as skmetrics
from sklearn.decomposition import PCA
import pandas as pd
import sys 

#%%
#mtx = sp.load_npz('rescale1000.npz')
#print type(mtx), mtx.shape

#%%
# making the matrix with 1 and 0, the input is the rescale1000 matrix
def binarymtx(mtx):
    col50=[]
#   create a column of empty sparse matrix
    binarymtx =sp.csc_matrix(np.zeros((mtx.shape[0],1),dtype=int))
    with open('col50.txt', 'r') as rr:
        col50=rr.read().splitlines()
    for i in range(len(col50)):
        index = col50[i]
        newcol = sp.csc_matrix(np.zeros((2132,1)))
        tempcol=mtx.getcol(int(index))
#        changing the intensity class 1 / 0 , threshold is 0
        for i in range (tempcol.shape[0]):
            if tempcol[i,0] >0:
                newcol[i,0]=1
            else:
                newcol[i,0]=0
        binarymtx=sp.hstack([binarymtx,newcol])
#        remove the column of empty matrix
    binarymtx= binarymtx.tocsc()
    binarymtx=binarymtx[:,1:]
    return binarymtx

#%%

# the input is the latent_vectors(2d) and the class vector with each column being the class label
def linearsvm_model(pca2d, binaryvector):
#    using the shuffle split
    shuffle = skm.ShuffleSplit(n_splits=5,test_size=0.3)
    model = svm.LinearSVC()
    score_list=[]
#    this loop will be iterated 5 times, cuz the number of split is number from above
    for train, test in shuffle.split(pca2d,binaryvector):
#        print pca2d[train].shape, binaryvector[train].shape, pca2d[test].shape
        predict= model.fit(pca2d[train],binaryvector[train])
#        print predict.classes_
        predict=model.decision_function(pca2d[test])
#        only input the predicted probability of the positive classes
        auc_score = skmetrics.roc_auc_score(binaryvector[test],predict)
        score_list.append(auc_score)
    return score_list
        
#%%

# the input is the latent_vectors(2d) and the class vector with each column being the class label
def svm_model(pca2d, binaryvector):
#    using the shuffle split
    shuffle = skm.ShuffleSplit(n_splits=5,test_size=0.3)
    model = svm.SVC(kernel='rbf',probability=True)
    score_list=[]
#    this loop will be iterated 5 times, cuz the number of split is number from above
    for train, test in shuffle.split(pca2d,binaryvector):
#        print pca2d[train].shape, binaryvector[train].shape, pca2d[test].shape
        predict= model.fit(pca2d[train],binaryvector[train])
#        print predict.classes_
        predict=model.predict_proba(pca2d[test])
#        only input the predicted probability of the positive classes
        auc_score = skmetrics.roc_auc_score(binaryvector[test],predict[:,1])
        score_list.append(auc_score)
    return score_list
        

#%%

# the input is the latent_vectors(2d) and the class vector with each column being the class label
def knn_model(pca2d, binaryvector):
#    using the shuffle split
    shuffle = skm.ShuffleSplit(n_splits=5,test_size=0.3)
    model=neighbors.KNeighborsClassifier()
    score_list=[]
#    this loop will be iterated 5 times, cuz the number of split is number from above
    for train, test in shuffle.split(pca2d,binaryvector):
#        print pca2d[train].shape, binaryvector[train].shape, pca2d[test].shape
        predict= model.fit(pca2d[train],binaryvector[train])
#        print predict.classes_
        predict=model.predict_proba(pca2d[test])
#        only input the predicted probability of the positive classes
        auc_score = skmetrics.roc_auc_score(binaryvector[test],predict[:,1])
        score_list.append(auc_score)
    return score_list


#%%
def getPCA(num_of_component,latent_reps):
    pca = PCA(n_components = num_of_component)
    transformed = pca.fit_transform(latent_reps)
    transformedPCA=transformed[:,0:num_of_component]
    #covariance = pca.get_covariance()
    #evr = pca.explained_variance_ratio_
    return transformedPCA

#%%

binary=sp.load_npz('binarymtx.npz')

def score_list(binary, model_name , transformedPCA):
    score_list=[]
    for i in range (binary.shape[1]):
        ttt=binary.getcol(i)
        length=binary.shape[0]
        tempcol=ttt.toarray().reshape(length,)
        if model_name=='knn':
            temp_score=knn_model(transformedPCA, tempcol)
        elif model_name=='svm':
            temp_score=svm_model(transformedPCA, tempcol)
        elif model_name=='linearsvm':
            temp_score=linearsvm_model(transformedPCA, tempcol)
        if i ==0:
            score_list=temp_score
        else:
            score_array= np.array(temp_score).reshape(1,5)
            score_list = sp.vstack([score_list,score_array])
    return score_list
    
#score_list=score_list(binary,'linearsvm')
#sp.save_npz('score_listlinear.npz',score_list)

#%%

#score_list=sp.load_npz('score_listlinear.npz')
#print type(score_list), score_list.shape

def bin_name_list():
    col50=[]
    with open('col50.txt', 'r') as rr:
            col50=rr.read().splitlines()
    col50 = map(int, col50)
    return col50

# saving the score list with the bins names to a csv file
def savecsv(col50,score_list,filename):
    lll=np.array(col50).reshape(len(col50),1)
    score_name=sp.hstack([lll,score_list])
    sdf = pd.DataFrame(score_name.toarray(),columns=['bin','score1','score2','score3','score4','score5'])
    sdf.to_csv(filename, index=False)
    
#   filter all the bins with all the auc_scores higher than 0.5
def bin50(score_list,col50):
    score_list=score_list.tocsr()
    true05 = score_list>0.5
    bin_name=[]
    binlist=[]
    empty=True
    for i in range(score_list.shape[0]):
        print i
        temprow=true05.getrow(i)
        temp=temprow.toarray()
        if np.all(temp):
            bin_name.append(col50[i])
            if empty:
                binlist=score_list.getrow(i)
                empty=False
            else:
                row= score_list.getrow(i).toarray()
                binlist=sp.vstack([binlist,row])
        
    print type(bin_name), len(bin_name)
    print type(binlist), binlist.shape
    print bin_name
    bin_name=np.array(bin_name).reshape(len(bin_name),1)
    final_binlist=sp.hstack([bin_name,binlist])
    print final_binlist.shape, type(final_binlist)
    final_binlist=final_binlist.tocsc()
    return final_binlist
    
#final_binlist=bin50(score_list,col50)
#print final_binlist.shape, type(final_binlist)
#sp.save_npz('filtered_binlistlinear.npz',final_binlist)

#%%
#lll=sp.load_npz('filtered_binlistlinear.npz')
#
#print type(lll), lll.shape
#
#temp = lll.getcol(0)
#print type(temp), temp.shape
#
#for i in range(temp.shape[0]):
#    print temp[i,0]

#  writing the list of bin names tat have all the scores more than 0.5
def writelist(filtered_binlistlinear,filename):
    temp = filtered_binlistlinear.getcol(0)
    with open(filename,'w') as www:
        for item in range(temp.shape[0]):
            www.write("%s\n" % str(temp[item,0]))
            
#%%

def main(model_name):
    latent_reps=np.load('latent_reps.npy')
    binary=sp.load_npz('binarymtx.npz')
    binnamelist=bin_name_list()
    component_list=[2,10,100]
    for number in component_list:
        transformedPCA=getPCA(number,latent_reps)
        scorelist=score_list(binary,model_name,transformedPCA)
        filenamecsv=model_name+str(number)+'bin_score.csv'
        savecsv(binnamelist, scorelist, filenamecsv)
        finallist=bin50(scorelist,binnamelist)
        filename=model_name+str(number)+'filtered_bin_names.txt'
        writelist(finallist,filename)
        
#%%
        

main('knn')