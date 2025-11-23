import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

mean_matrix=sp.load_npz('knnmean_matrix.npz')
component_list=[]
with open('knnbest_3_auc.txt','r') as rrr:
    for line in rrr:
        templine=line.split(',')
        component_list.append(int(templine[0]))
    
print component_list
print mean_matrix.shape, type(mean_matrix)

#columns={}
#names=[]

#for col in component_list:
#    tempcol=mean_matrix.getcol(col)
#    tempcol2=tempcol.toarray()
#    columns[str(col)]=tempcol2

tempcol =mean_matrix.getcol(component_list[0])
auc1 =tempcol.toarray()
print type(auc1), auc1.shape
tempcol =mean_matrix.getcol(component_list[1])
auc2 =tempcol.toarray()
print type(auc1), auc1.shape
tempcol =mean_matrix.getcol(component_list[2])
auc3 =tempcol.toarray()
print type(auc1), auc1.shape

data = np.hstack([auc1, auc2])
print type(data), data.shape

data = np.hstack([data,auc3])

print type(data), data.shape

plt.style.use('seaborn-deep')
#
#bins = np.linspace(-10, 10, 30)
fig=plt.figure(figsize=(16,12))
bins=np.linspace(0.42,0.62,30)
plt.hist(data,bins=bins, alpha=0.5, label=component_list)
#plt.hist(auc1,bins=bins, alpha=0.5, label=str(component_list[0]))
#plt.hist(auc2, alpha=0.5, label=str(component_list[1]))
#plt.hist(auc3, alpha=0.5, label=str(component_list[2]))
plt.xticks(np.linspace(0.40,0.6,20))
plt.title('Histogram with frequency of AUC score of best three bin')
plt.legend(loc='upper right')
plt.show()
plt.xlabel('AUC score')
plt.ylabel('Frequency')
fig.savefig('best_3_bins.png')

#%%

#plt.style.use('seaborn-deep')
#
#x = np.random.normal(1, 2, 5000)
#y = np.random.normal(-1, 3, 5000)
#
#print type(x), x.shape
#data = np.vstack([x, y]).T
#print type(data), data.shape
#bins = np.linspace(-10, 10, 30)
#
#
#plt.hist(data, bins, alpha=0.7, label=['x', 'y'])
#plt.legend(loc='upper right')
#plt.show()