import scipy.sparse as sp
import numpy as np
from sklearn import svm
import sklearn.model_selection as skm
import sklearn.metrics as skmetrics
import os.path

# the input is the latent_vectors(2d) and the class vector with each column being the class label
def svm_model(latent292, classvector,model_name):
#    using the shuffle split
    stratified = skm.StratifiedKFold(n_splits=5)
    if model_name=='linear':
      model = svm.SVC(kernel='linear',probability=True,class_weight='balanced',C=0.1)
      print 'model'
    elif model_name =='rbf':
      model = svm.SVC(kernel='rbf',probability=True,class_weight='balanced',C=0.1,gamma=10)
    score_list=[]
#    this loop will be iterated 5 times, cuz the number of split is number from above
    for train, test in stratified.split(latent292,classvector):
        try:
            predict= model.fit(latent292[train],classvector[train])
            
        except ValueError:
            auc_score = 0.5
            
            score_list.append(auc_score)
            continue
#        if the class labels have only one class, then the auc-score ill be set to 0.5
        if predict.classes_.shape[0] <2:
            auc_score = 0.5
            
            score_list.append(auc_score)
        else:
#            tempsparse= sp.csr_matrix(latent292[test])
#            predict=model.predict_proba(tempsparse)
            predict=model.predict_proba(latent292[test])
    #        only input the predicted probability of the positive classes
            try:
#     sometimes the positive class is too few, its not possible get the auc score
#     so this is to catch that error and set auc score to 0.5
                auc_score = skmetrics.roc_auc_score(classvector[test],predict[:,1])
            except ValueError:
                auc_score = 0.5
                
            score_list.append(auc_score)
            
    return score_list

# get the list of bins name that are present in the binary matrix from the resale matrix
def bin_name_list(matrix,num_mol_filter):
    non_empty_list=[]
    num_of_mol ={}
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
        num_of_mol[str(i)] = num
    return non_empty_list, num_of_mol
  
# parse in the scorelist matrix with 5 auc score for each bin, 
# and find the average of each bin and the maximum auc socre for each bin
def get_mean_auc(score_list):
    mean_value = np.mean(score_list)
    return mean_value
    
def read_file(result_file):
    if os.path.isfile(result_file):
        mode = 'r'
        with open(result_file,mode) as fff:
            lines = fff.readlines()
        last_line = lines[-1].split(',')
        bin_name = last_line[0]
        return bin_name
    else:
        mode = 'w'
        first_line = 'bin_name,auc_score,num_of_molecule\n'
        with open(result_file,mode) as www:
            www.write(first_line)
        print 'file opened'
        return None


def write_data(result_file, result_line):
    with open(result_file,'a') as fff:
        fff.write(result_line)

def cal_all_mean(model_name, latent292, class_matrix, progress_num, non_empty_list,num_of_mol,result_file):
    print progress_num,'-----'
#    print non_empty_list
    if progress_num == 'bin_name' or progress_num == None:
        index = 0
    else:
        tempindex = non_empty_list.index(int(progress_num))
#        print tempindex,'========'
        index = tempindex + 1
#        print index
    while index < len(non_empty_list):
        progress = non_empty_list[index]
        print progress
        class_vector = class_matrix.getcol(index)
        length = class_vector.shape[0]
        tempcol = class_vector.toarray().reshape(length,)
        score_list = svm_model(latent292,tempcol,model_name)
        print len(score_list)
        mean_value = get_mean_auc(score_list)
        print mean_value
        result_line = str(progress) + ',' + str(mean_value) +',' + str(num_of_mol[str(progress)]) + '\n'
#        print str(num_of_mol[str(progress)])
        write_data(result_file, result_line)
        index +=1


def main(model_name,num_mol_filter,result_file):
    latent292 =np.load('latent_reps292.npy')
#    latent292 =np.load('latent_reps292_2113.npy')
    print type(latent292),latent292.shape

    matrix =sp.load_npz('rescale.npz')
#    matrix =sp.load_npz('136original.npz')
    print matrix.shape, type(matrix)

    non_empty_list, num_of_mol =bin_name_list(matrix,num_mol_filter)
    class_matrix = sp.load_npz('classes.npz')
#    class_matrix = sp.load_npz('136binarymtx.npz')
    print class_matrix.shape, type(class_matrix)

    progress_num = read_file(result_file)

    cal_all_mean(model_name, latent292, class_matrix,progress_num, non_empty_list, num_of_mol,result_file)
    
    
main ('rbf',5,'testting.txt')