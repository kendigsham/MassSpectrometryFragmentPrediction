import urllib2
import json
import pickle
import glob
import pprint
import requests
#%%
# mine
import scipy.sparse as sp
from sklearn import svm
from sklearn.model_selection import train_test_split

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import h5py

#%%
def readbin_name():
    with open('/home/kenny/linux/project/bin_names.txt','r') as rrr:
        number=rrr.read().splitlines()
    return number

# get the names 
number=readbin_name()
print(number),type(number)

colormatrix = sp.load_npz('/home/kenny/linux/project/colormatrix.npz')
print colormatrix.shape, type(colormatrix)

latent_reps = np.load('/home/kenny/linux/latent_reps.npy')
print type(latent_reps), latent_reps.shape

#%%

def getcol(index):
    column = colormatrix.getcol(index)
#     print type(column) , column.shape
    col = column.toarray()
#     print type(col), col.shape
    singlecol = col.T
# the return is a list
    return singlecol 
    
pca = PCA(n_components = 10)
# transformed = pca.fit_transform(X)
transformed = pca.fit_transform(latent_reps)
covariance = pca.get_covariance()

evr = pca.explained_variance_ratio_
print evr
print 'Explained variations -- first two PCs: %.2f' % (evr[0] + evr[1])
print 'Explained variations -- all components: %.2f' % np.sum(evr)

latent_dim = 100
for i in range(len(number)):
    
    singlecol=getcol(i)
    fig = plt.figure(figsize=(16,12))

    norm = matplotlib.colors.Normalize(vmin = np.min(singlecol), vmax = np.max(singlecol), clip = False)
    mydata = plt.scatter(transformed[:,0],transformed[:,1],c=singlecol,norm=norm,cmap='Blues',marker='o',s=40)
    
    title = 'PCA - Latent Dim %d, bin = %d' % (latent_dim, int(number[i]))
    plt.title(title)
    plt.colorbar()
    title+='.png'
    plt.savefig(title)
    plt.show()
    
def gettargetvector(singlecol):
    
#  changing the columns from the log values to 1 (presence) or 0 (absence)
    for i in range(len(singlecol[0])):
        if singlecol[0][i]<=0.1:
            singlecol[0][i]=0
        else:
            singlecol[0][i]=1
    singlecol = singlecol.T
    return singlecol
    
def check_targetvector():
    index_list=[]
#     for i in range(len(number)):
    for i in range(1):
        singlecol=getcol(i)
        rang = np.ptp(singlecol,axis=1)
        print rang
        target=gettargetvector(singlecol)
        print target
        print np.unique(target)
        unique=[]
#         if i not in target:
#             unique.append(i)
#         else:
#             continue
#         if len(unique) ==2:
#             index_list.append(i)
#         else:
#             continue
#     print index_list, len(index_list) 
check_targetvector()

# colorcol = getcol(0)
# r = np.ptp(colorcol,axis=1)
# print colorcol, colorcol.shape, r

def split_data(xdata,ydata):
#     splitting the data into training data and data for testing
    xtrain, xtest, ytrain, ytest=train_test_split(xdata,ydata,test_size=0.3,random_state=30)  
    return xtrain, xtest, ytrain, ytest

# train the data with support vector 
def svm_train(xtrain,ytrain):
    mysvm = svm.SVC(kernel='rbf',c=50,gamma=50)
    mysvm.fit(xtrain,ytrain)
    return mysvm
    
def accuracy_print(ac_list):
    accuracy = mysvm.score(xtest,ytest)
    ac_list.append(accuracy)
    
transformed2d=transformed[:,0:2]
# model_list = [svm.SVC(kernel='rbf') for i in range(len(number))]

# # loop through all the columns and find all the accuracy
# for count, model in enumerate(model_list):
singlecol = getcol(10)
print type(singlecol), singlecol.shape
target = gettargetvector(singlecol)
# the required shape is (1,2132)
xtrain, xtest, ytrain, ytest=split_data(transformed2d,target)
model.fit(xtrain,ytrain)
print model.score(xtest,ytest)

figure = plt.figure(figsize=(16,12))
h = .02  # step size in the mesh

# create a mesh to plot in, the first dimension is X, second diension is Y
# xmin, xmax = transformed2d[:, 0].min() -1, transformed2d[:, 0].max() +1
# ymin, ymax = transformed2d[:, 1].min() -1, transformed2d[:, 1].max() +1
# xx, yy = np.meshgrid(np.arange(xmin, xmax, h),np.arange(ymin, ymax, h))

xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, 500),np.linspace(-0.5, 0.5, 500))

Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# norm = matplotlib.colors.Normalize(vmin = np.min(colorcol), vmax = np.max(colorcol), clip = False)
# Plot also the training points
plt.scatter(transformed2d[:, 0], transformed2d[:, 1], c=singlecol,cmap=plt.cm.RdBu)
plt.xlabel('dimension 1')
plt.ylabel('dimension 2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()

