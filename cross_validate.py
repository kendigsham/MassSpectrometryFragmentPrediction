
import numpy as np
from sklearn import svm

import sklearn.model_selection as skm
import sklearn.metrics as skmetrics


#%%

        
# the input is the latent_vectors(2d) and the class vector with each column being the class label
def svm_model(pca2d, classvector, model_name):
#    using the shuffle split
    stratified = skm.StratifiedKFold(n_splits=5)
    if model_name == 'rbf':
      model = svm.SVC(kernel=model_name,probability=True,C=0.1,gamma=10,class_weight='balanced')
    elif model_name == 'linear':
      model = svm.SVC(kernel=model_name,probability=True,C=0.1,class_weight='balanced')
      
    score_list=[]
#    this loop will be iterated 5 times, cuz the number of split is number from above
    for train, test in stratified.split(pca2d,classvector):
        try:
            predict= model.fit(pca2d[train],classvector[train])
        except ValueError:
            auc_score = 0.0
            # print auc_score
            score_list.append(auc_score)
            continue
#        if the class labels have only one class, then the auc-score ill be set to 0.5
        if predict.classes_.shape[0] <2:
            auc_score = 0.0
            # print auc_score
            score_list.append(auc_score)
        else:
#            tempsparse= sp.csr_matrix(pca2d[test])
#            predict=model.predict_proba(tempsparse)
            predict=model.predict_proba(pca2d[test])
    #        only input the predicted probability of the positive classes
            try:
#     sometimes the positive class is too few, its not possible get the auc score
#     so this is to catch that error and set auc score to 0.5
                auc_score = skmetrics.roc_auc_score(classvector[test],predict[:,1])
            except ValueError:
                auc_score = 0.0
            score_list.append(auc_score)
    return score_list


#%%

# make score list with all the auc score of all the bins
def score_list(classes, model_name , latent_rep):
    length=classes.shape[0]
    tempcol=classes.reshape(length,)
    temp_score=svm_model(latent_rep, tempcol,model_name)
    score_list=temp_score
    return score_list
    

#%%

def get_auc_score(model_name,latent_rep,classes):
    scorelist =score_list(classes,model_name,latent_rep)
#    print type(scorelist), len(scorelist), 'scorelist'
    mean_auc = np.mean(scorelist)
    return mean_auc
    


