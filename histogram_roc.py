import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_sen_v_spec(sen,spec):
  plt.figure(figsize=(10,8))
  plt.plot(sen,spec)
  
  plt.plot([0, 1], [0, 1], 'k--')
#  plt.xlim([0,1])
#  plt.ylim([0,1])
  plt.title('ROC curve for gnps2massbank',fontsize=16)
  plt.xlabel('False Positive Rate (1-Specificity)',fontsize=16)
  plt.ylabel('True Positive Rate (Sensitivity)',fontsize=16)
  title = 'ROC curve for fragment prediction gnps2massbank'+'.png'
#  plt.savefig(title)
  plt.show()
  
def histogram(row,column,filename):
  plt.figure(figsize=(10,8))
  aa = np.array(row).reshape(1,-1)
  bb = np.array(column).reshape(1,-1)
  data = np.vstack([aa,bb]).T
  
  plt.hist(data,label=['row-permutation','column-permutation'])
  plt.title(filename,fontsize=16)
#  plt.axvline(0.05, color='b', linestyle='dashed', linewidth=2)
  bins = np.linspace(0,1, 10)
  plt.legend(loc='upper center')
  plt.xticks(bins)
  plt.xlabel('p values',fontsize=16)
  plt.ylabel('Number of instances',fontsize=16)
  filename +='.png'
  print '--------------------'
  plt.savefig(filename)

  plt.show()

ppp = pd.DataFrame.from_csv('ranked_sen_spec_p_value_gnps2massbank.csv')


row_sen = ppp['p_value_row_sen'].tolist()

column_sen = ppp['p_value_column_sen'].tolist()

row_spec = ppp['p_value_row_spec'].tolist()

column_spec = ppp['p_value_column_spec'].tolist()

histogram(row_sen,column_sen,'Histogram sensitivity p values after row_column permutation (gnps2massbank)')

#histogram(row_spec,column_spec,'Histogram specificity p values after row/ column permutation')


print 'mean row sen', np.mean(row_sen)
print 'mean column sen', np.mean(column_sen)
print 'difference', np.mean(row_sen) - np.mean(column_sen)
print '=============='
print 'mean row spec', np.mean(row_spec)
print 'mean column spec' , np.mean(column_spec)
print 'difference', np.mean(row_spec) - np.mean(column_spec)


#sen = ppp['sensitivity'].tolist()
#
#spec = ppp['specificity'].tolist()
#
#
#oneminus = []
#
#for i in spec:
#  oneminus.append(float(1-i))
#  
#plot_sen_v_spec(oneminus,sen)