
import mz_value_list
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

dictt={}
with open('auc_massbank2gnps.txt','r') as f1:
  next(f1)
  for line in f1:
    templine = line.strip().split(',')
    dictt[templine[0]] = float(templine[1])

#print dictt

dataframe2 = mz_value_list.get_filtered_dataframe('massbank_result2.csv',20,10,0.7)

x_axis=[]
y_axis=[]



for index, row in dataframe2.iterrows():
  x_value = row['auc_score']
  x_axis.append(x_value)
  temp_mz_value = '%.3f'%row['m/z_value']
  y_value = dictt.get(temp_mz_value)
  if y_value == None:
    y_axis.append(0)
  else:
    y_axis.append(y_value)
  
print len(x_axis)
print len(y_axis)

xxx = np.array(x_axis)
yyy = np.array(y_axis)

slope, inter, r_value, p_value, std_err = stats.linregress(xxx,yyy)

print r_value**2
print p_value
#  
#filename ='scatter plot of AUC scores from dataset massbank and GNPS'
#plt.figure(figsize=(10,8))
#xx = np.array(x_axis)
#yy = np.array(y_axis)
#plt.scatter(xx,yy)
#plt.title(filename,fontsize=16)
#plt.plot([0.4, 1], [0.4, 1], 'k--')
#plt.axvline(0.7, color='b', linestyle='dashed', linewidth=2)
#plt.xlabel('AUC scores of massbank data',fontsize=16)
#plt.ylabel('AUC scores of GNPS data',fontsize=16)
#filename +='.png'
#plt.savefig(filename)
#plt.show()

#import numpy
#
#numpy.corrcoef(x_axis,y_axis)[0,1]