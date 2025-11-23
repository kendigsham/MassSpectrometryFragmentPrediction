import matplotlib.pyplot as plt
from collections import Counter


def read_para_file(): 
    dictionary={}
    dictionary['kernel']=[]
    dictionary['C']=[]
    dictionary['gamma']=[]
    with open('best_parameters.txt','r') as rrr:
        for line in rrr:
            templine=line.split(',')
            if templine[0]=='kernel':
                dictionary['kernel'].append(templine[1].rstrip('\n'))
            if templine[0]=='C':
                dictionary['C'].append(templine[1].rstrip('\n'))
            if templine[0]=='gamma':
                dictionary['gamma'].append(templine[1].rstrip('\n'))
    return dictionary

#%%

def plot_kernel():
    fig=plt.figure(figsize=(16,12))
    c = Counter(dictionary['kernel'])
    linear = c['linear']
    rbf = c['rbf']
    bar_heights = (linear, rbf)
    x = (1, 2)
    fig, ax = plt.subplots()
    width = 0.4
    plt.title("Most frequent type of kernel")
    plt.xlabel("types of kernel")
    plt.ylabel("Frequency")
    ax.bar(x, bar_heights, width)
    ax.set_xlim((0, 3))
    ax.set_ylim((0, max(linear, rbf)*1.1))
    ax.set_xticks([i+width/3 for i in x])
    ax.set_xticklabels(['linear', 'rbf'])
    plt.show()
    fig.savefig('best_kernel.png')

#%%

import matplotlib.pyplot as plt
import numpy as np

dictionary=read_para_file()

def plot_best_c(dictionary):
    lll=dictionary['C']
    fig=plt.figure(figsize=(16,12))
    lll=np.array(lll).reshape(len(lll),1).astype(np.int)
    print type(lll), lll.shape
    kkk=lll.reshape(lll.shape[0],)
    print type(kkk), kkk.shape
    c=Counter(kkk)
    most_freq=c.most_common(3)
    extra_ticks=[]
    for i in range(3):
        extra_ticks.append(most_freq[i][0])
    bins=np.linspace(0,100,30, dtype = int)
    plt.hist(lll,bins=bins)
    plt.xticks(np.linspace(0,100,40, dtype = int))
    plt.title("Most frequent C value")
    plt.xlabel("C-value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    fig.savefig('best_c_value.png')
    
def plot_best_gamma(dictionary):
    lll=dictionary['gamma']
    fig=plt.figure(figsize=(16,12))
    lll=np.array(lll).reshape(len(lll),1).astype(np.int)
    print type(lll), lll.shape
    kkk=lll.reshape(lll.shape[0],)
    c=Counter(kkk)
    most_freq=c.most_common(3)
    extra_ticks=[]
    for i in range(3):
        extra_ticks.append(most_freq[i][0])
    bins=np.linspace(0,100,30, dtype = int)
    plt.hist(lll,bins=bins)
    plt.xticks(np.linspace(0,100,40, dtype = int))
    plt.title("Most frequent gamma value")
    plt.xlabel("gamma")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    fig.savefig('best_gamma_value.png')
    
plot_best_gamma(dictionary)
plot_best_c(dictionary)
plot_kernel()