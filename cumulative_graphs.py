import mz_value_list
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
csv_file = mz_value_list.read_csv('massbank_result.txt')

print type(csv_file)

#mz_value = csv_file['m/z_value'].tolist()
#print type(mz_value)
auc_score = csv_file['auc_score'].tolist()
print type(auc_score)

auc_score2 = np.array(auc_score).reshape(-1,)
print auc_score2.shape

n_bins = 50
plt.figure(figsize=(8,6))
bins=np.linspace(0,1,100)
# plot the cumulative histogram
plt.hist(auc_score2, bins=bins, histtype='step', cumulative=True, label='auc scores')
#plt.hist(auc_score2, bins=bins, label='auc scores',range=(bins.min(),bins.max()))

# tidy up the figure
plt.xticks(np.linspace(0,1,10))
#plt.yticks(np.linspace(0,1,10))
plt.legend(loc='right')
plt.title('Cumulative AUC score')
#plt.title('histogram of AUC score')
plt.xlabel('AUC score')
plt.ylabel('Number of m/z values')
plt.savefig('cumulative_massbank_auc_score')
plt.show()

#%%

sen_spec_csv = mz_value_list.read_csv('sortedlambda.txt')

sensitivity = sen_spec_csv['sensitivity'].tolist()
print type(sensitivity)
specificity = sen_spec_csv['specificity'].tolist()
print type(specificity)

sensitivity2 = np.array(sensitivity).reshape(-1,)
print sensitivity2.shape

specificity2 = np.array(specificity).reshape(-1,)
print specificity2.shape

n_bins = 50
plt.figure(figsize=(8,6))
bins=np.linspace(0,1,100)
# plot the cumulative histogram
#plt.hist(sensitivity2, bins=bins, label='senitivity',range=(bins.min(),bins.max()))
plt.hist(sensitivity2, bins=bins, histtype='step',cumulative=True, label='senitivity',range=(bins.min(),bins.max()))

#plt.hist(specificity2, bins=bins, label='specificity',range=(bins.min(),bins.max()))
plt.hist(specificity2, bins=bins, histtype='step',cumulative=True, label='specificity',range=(bins.min(),bins.max()))


# tidy up the figure
plt.xticks(np.linspace(0,1,10))
#plt.yticks(np.linspace(0,1,10))
plt.legend(loc='right')
plt.title('cumulative histogram of sensitivity and specificity score')
plt.xlabel('sensitivity and specificity score')
plt.ylabel('frequency')
plt.savefig('cumulative_sen_spec_massbank2gnps')
plt.show()
