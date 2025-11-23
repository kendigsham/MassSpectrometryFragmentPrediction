import scipy.sparse as sp
import numpy as np
from sklearn import svm
import sklearn.model_selection as skm
import sklearn.metrics as skmetrics
import operator
import pandas as pd

# count molecules and filter the number of molecules that are stored into the list
# molecule counts lower than a threshold will not be added
def countmolecule(mtx,num_mol_filter):
    mol_num={}
    coomatrix=mtx.tocoo()
    #return all the indices of columns that are not empty
    colindex = coomatrix.col
    colindex=colindex.tolist()
    unique = set(colindex)
    unique=list(unique)
    for i in unique:
        num=mtx.getcol(i).count_nonzero()
        if num < num_mol_filter:
          mol_num[str(i)]= num
    return mol_num

# the input is the latent_vectors(2d) and the class vector with each column being the class label
def svm_model(pca2d, binaryvector,model_name):
#    using the shuffle split
    stratified = skm.StratifiedKFold(n_splits=5)
    if model_name=='linear':
      model = svm.SVC(kernel='linear',probability=True,class_weight='balanced',C=0.1)
      print 'model'
    elif model_name =='rbf':
      model = svm.SVC(kernel='rbf',probability=True,class_weight='balanced',C=0.1,gamma=10)
    score_list=[]
#    this loop will be iterated 5 times, cuz the number of split is number from above
    for train, test in stratified.split(pca2d,binaryvector):
        try:
            predict= model.fit(pca2d[train],binaryvector[train])
            
        except ValueError:
            auc_score = 0.5
            
            score_list.append(auc_score)
            continue
#        if the class labels have only one class, then the auc-score ill be set to 0.5
        if predict.classes_.shape[0] <2:
            auc_score = 0.5
            
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
                
            score_list.append(auc_score)
            
    return score_list

# make score list with all the auc score of all the bins
def score_list(binary, model_name , transformedPCA):
    print 'score'
    score_list=[]
    for i in range (binary.shape[1]):
        ttt=binary.getcol(i)
        length=binary.shape[0]
        tempcol=ttt.toarray().reshape(length,)
        if model_name=='linear':
            print 'linear'
            temp_score=svm_model(transformedPCA, tempcol,model_name)
        elif model_name=='rbf':
            temp_score=svm_model(transformedPCA, tempcol,model_name)
        if i ==0:
            score_list=temp_score
        else:
            length = len(temp_score)
            score_array= np.array(temp_score).reshape(1,length)
            score_list = sp.vstack([score_list,score_array])
    score_list=score_list.tocsr()
    return score_list
    
# get the list of bins name that are present in the binary matrix from the resale matrix
def bin_name_list(matrix,num_mol_filter,mol_display):
    display_list={}
    non_empty_list=[]
    coomatrix=matrix.tocoo()
    #return all the indices of columns that are not empty
    colindex = coomatrix.col
    colindex=colindex.tolist()
    unique = set(colindex)
    bin_name_list=list(unique)
    for i in bin_name_list:
      num=matrix.getcol(i).count_nonzero()
      if num >= num_mol_filter:
        non_empty_list.append(i)
      if num >= mol_display:
        display_list[str(i)]= num
    return non_empty_list, display_list
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
#    each column represent a bin
    meancol=sp.csc_matrix(meancol)
    maxcol=sp.csc_matrix(maxcol)
    max_matrix = sp.hstack([max_matrix,maxcol])
    mean_matrix = sp.hstack([mean_matrix,meancol])
    
    return max_matrix, mean_matrix
    
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
        num_molecule = molecule_count.get(str(binname))
#        the num 0f mols in the list should be those that are above the threshold set in countmolecule method
        if not num_molecule:
          continue
        else:
          bin_name.append(binname)
          num_molecules.append(num_molecule)
          auc_list.append(max3list[i][1])
    return bin_name,auc_list,num_molecules

def main(model_name,num_mol_filter,mol_display):
    latent_reps =np.load('latent_reps292.npy')
    print type(latent_reps),latent_reps.shape
    matrix =sp.load_npz('rescale.npz')
    print matrix.shape, type(matrix)
    bin_list, display_list =bin_name_list(matrix,num_mol_filter,mol_display)
#    binary =binarymtx(matrix,bin_list)
    binary = sp.load_npz('classes.npz')
    print binary.shape, type(binary)
    max_matrix =sp.csc_matrix(np.zeros((binary.shape[1],1)))
    mean_matrix =sp.csc_matrix(np.zeros((binary.shape[1],1)))
    scorelist =score_list(binary,model_name,latent_reps)
    print type(scorelist), scorelist.shape
    max_matrix,mean_matrix =get_mean_auc(scorelist,max_matrix,mean_matrix)
    max_matrix =max_matrix[:,1:]
    mean_matrix =mean_matrix[:,1:]
    print type(mean_matrix),mean_matrix.shape
    max_name =model_name+'max_matrix.npz'
    mean_name =model_name+'mean_matrix.npz'
    sp.save_npz(max_name,max_matrix)
    sp.save_npz(mean_name,mean_matrix)
    bin_name,auc_list,num_molecules=best_average_auc(mean_matrix,display_list,bin_list,model_name)
#    print len(bin_list),'============='
#    best_average_auc(mean_matrix,molecule_count,bin_list,model_name)
    return bin_name,auc_list,num_molecules    
    
def everything(model_name,num_mol_filter,mol_display): 
    print '-------'  
    dictionary={}
    binname='bin_name'
    auclist='auc_list'
    nummol='num_of_molecule'
    print'-------'
    bin_name,auc_list,num_molecules=main(model_name,num_mol_filter,mol_display)
    print'----------'
    dictionary[binname] = bin_name
    dictionary[auclist] = auc_list
    dictionary[nummol] = num_molecules
    file_to_save = pd.DataFrame(dictionary)
    filename = model_name +','+str(num_mol_filter)+','+str(mol_display) +',results.csv'
    file_to_save.to_csv(filename)
    
everything('linear',5,0)


