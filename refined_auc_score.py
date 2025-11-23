import scipy.sparse as sp
import numpy as np
from sklearn import svm
from sklearn import neighbors
import sklearn.model_selection as skm
import sklearn.metrics as skmetrics
from sklearn.decomposition import PCA
import operator
import pandas as pd
from sklearn.cross_validation import cross_val_predict

#%%

# the input is the latent_vectors(2d) and the class vector with each column being the class label
def svm_model(pca2d, binaryvector):
#    using the shuffle split
    stratified = skm.StratifiedKFold(n_splits=2)
    model = svm.SVC(kernel='linear',probability=True)
    score_list=[]
#    this loop will be iterated 5 times, cuz the number of split is number from above
    for train, test in stratified.split(pca2d,binaryvector):
        try:
            predict= model.fit(pca2d[train],binaryvector[train])
        except ValueError:
            auc_score = 0.5
            print auc_score
            score_list.append(auc_score)
            continue
#        if the class labels have only one class, then the auc-score ill be set to 0.5
        if predict.classes_.shape[0] <2:
            auc_score = 0.5
            print auc_score
            score_list.append(auc_score)
        else:
#            tempsparse= sp.csr_matrix(pca2d[test])
#            predict=model.predict_proba(tempsparse)
            predict=model.predict_proba(pca2d[test])
    #        only input the predicted probability of the positive classes
            try:
#     sometimes the positive class is too few, its not possible get the auc score
#     so this is to catch that error and set auc score to 0.5
                auc_score = skmetrics.roc_auc_score(binaryvector[test],predict[:,1])
            except ValueError:
                auc_score = 0.5
                print auc_score
            score_list.append(auc_score)
    return score_list

#%%

## the input is the latent_vectors(2d) and the class vector with each column being the class label
def knn_model(pca2d, binaryvector):
#    using the shuffle split
#    shuffle = skm.ShuffleSplit(n_splits=5,test_size=0.3)
    stratified=skm.StratifiedKFold(n_splits=2)
    model=neighbors.KNeighborsClassifier(algorithm='brute')
    score_list=[]
#    this loop will be iterated 5 times, cuz the number of split is number from above
    for train, test in stratified.split(pca2d,binaryvector):
#        print pca2d[train].shape, binaryvector[train].shape, pca2d[test].shape
        predict= model.fit(pca2d[train],binaryvector[train])
#        if the class labels have only one class, then the auc-score ill be set to 0.5
        if predict.classes_.shape[0] <2:
            auc_score = 0.5
            print auc_score
            score_list.append(auc_score)
        else:
            tempsparse= sp.csr_matrix(pca2d[test])
            predict=model.predict_proba(tempsparse)
    #        only input the predicted probability of the positive classes
            try:
#     sometimes the positive class is too few, its not possible get the auc score
#     so this is to catch that error and set auc score to 0.5
                auc_score = skmetrics.roc_auc_score(binaryvector[test],predict[:,1])
            except ValueError:
                auc_score = 0.5
                print auc_score
            score_list.append(auc_score)
    return score_list
    
#%%

def countmolecule(mtx):
    mol_num={}
    coomatrix=mtx.tocoo()
    #return all the indices of columns that are not empty
    colindex = coomatrix.col
    colindex=colindex.tolist()
    unique = set(colindex)
    unique=list(unique)
    for i in unique:
        num=mtx.getcol(i).count_nonzero()
        mol_num[str(i)]= num
    return mol_num
  
#%%
# make score list with all the auc score of all the bins
def score_list(binary, model_name , transformedPCA):
    score_list=[]
    for i in range (binary.shape[1]):
        ttt=binary.getcol(i)
        length=binary.shape[0]
        tempcol=ttt.toarray().reshape(length,)
        if model_name=='knn':
            temp_score=knn_model(transformedPCA, tempcol)
        elif model_name=='svc':
            temp_score=svm_model(transformedPCA, tempcol)
        elif model_name=='linearsvm':
            temp_score=linearsvm_model(transformedPCA, tempcol)
        if i ==0:
            score_list=temp_score
        else:
            score_array= np.array(temp_score).reshape(1,2)
            score_list = sp.vstack([score_list,score_array])
    score_list=score_list.tocsr()
    return score_list
#%%

# parse in the scorelist matrix with 5 auc score for each bin, 
# and find the average of each bin and the maximum auc socre for each bin
def get_mean_auc(scorelist,max_matrix,mean_matrix):
    meanlist=[]
    maxlist=[]
    for i in range (scorelist.shape[0]):
#        each row is the scores for 1 bin
        temprow=scorelist.getrow(i)
        maxx = temprow.max()
        mean = temprow.mean()
        meanlist.append(mean)
        maxlist.append(maxx)
#        the mean and max score a all the bins are transformed into a column
    meancol= np.array(meanlist).reshape(scorelist.shape[0],1)
    maxcol= np.array(maxlist).reshape(scorelist.shape[0],1)
#    each row represent a bin
    meancol=sp.csc_matrix(meancol)
    maxcol=sp.csc_matrix(maxcol)
    max_matrix = sp.hstack([max_matrix,maxcol])
    mean_matrix = sp.hstack([mean_matrix,meancol])
    return max_matrix, mean_matrix
    
#%%
# get the average auc score of all the bins(rows) tried with different dimension of latent space (column)
def best_average_auc(matrix,molecule_count,bin_list,model_name):
    auc_score_list={}
    for i in range (matrix.shape[0]):
        temprow=matrix.getrow(i)
        mean=temprow.mean()
        auc_score_list[str(i)]=mean
    max3list=sorted(auc_score_list.items(),key=operator.itemgetter(1))
    max3list.reverse()
    print type(max3list), len(max3list),'==========='
    bin_name=[]
    auc_list=[]
    num_molecules=[]
    for i in range(len(max3list)):
        tempindex =int(max3list[i][0])
#        the bin_list should have the row order of the matrix input
        binname=bin_list[tempindex]
        bin_name.append(binname)
        num_molecule = molecule_count.get(str(binname))
        num_molecules.append(num_molecule)
        auc_list.append(max3list[i][1])
#    filename=model_name+'best_auc.txt'
#    with open (filename,'w') as fff:
#        for i in range(len(max3list)):
#            tempindex =int(max3list[i][0])
##            getting the bin name instead of the index name
#            bin_name =bin_list[tempindex]
#            num_molecule = molecule_count.get(bin_name)
#            fff.write('%s,%s,%s,num_of_molecule,%s\n' %(bin_name,max3list[i][0],max3list[i][1]),num_molecule)
    return bin_name,auc_list,num_molecules
    

def bin_name_list(rescale_matrix):
    coomatrix=rescale_matrix.tocoo()
    #return all the indices of columns that are not empty
    colindex = coomatrix.col
    colindex=colindex.tolist()
    unique = set(colindex)
    bin_list = list(unique)
    return bin_list
#%%

def main(model_name,binname):
    latent_reps =np.load('/home/kenny/linux/latent_reps.npy')
    filename =binname+'_rescale_matrix.npz'
    matrix =sp.load_npz(filename)
    molecule_count = countmolecule(matrix)
    bin_list =bin_name_list(filename)
    filename2=binname+'_binarymtx.npz'
    binary = sp.load_npz(filename2)
    max_matrix =sp.csc_matrix(np.zeros((binary.shape[1],1)))
    mean_matrix =sp.csc_matrix(np.zeros((binary.shape[1],1)))
#    transformedPCA=getPCA(num_of_component,latent_reps)
    scorelist =score_list(binary,model_name,latent_reps)
    print type(scorelist), scorelist.shape
    max_matrix,mean_matrix =get_mean_auc(scorelist,max_matrix,mean_matrix)
    max_matrix =max_matrix[:,1:]
    mean_matrix =mean_matrix[:,1:]
    print type(mean_matrix),mean_matrix.shape
#    max_name =model_name+'max_matrix.npz'
#    mean_name =model_name+'mean_matrix.npz'
#    sp.save_npz(max_name,max_matrix)
#    sp.save_npz(mean_name,mean_matrix)
    bin_name,auc_list,num_molecules=best_average_auc(mean_matrix,molecule_count,bin_list,model_name)
#    print len(bin_list),'============='
#    best_average_auc(mean_matrix,molecule_count,bin_list,model_name)
    return bin_name,auc_list,num_molecules    
    
#%%

def everything(model_name,binlist):
#    binlist=[2,10,40,60,80,100]
#    componentlist=[2]      
    dictionary={}
    for binnum in binlist:
        binname=str(binnum)
        auclist=str(binnum)+'auc_list'
        nummol=str(binnum)+'num_of_molecule'
        sub_bin_name,auc_list,num_molecules = main(model_name,binname)
#        main(model_name,component)
        dictionary[binname] = sub_bin_name
        dictionary[auclist] = auc_list
        dictionary[nummol] = num_molecules
    file_to_save = pd.DataFrame(dictionary)
    filename = 'subbin_results.csv'
    file_to_save.to_csv(filename,header=False,index=False)

lll=[54]

everything('svc',lll)