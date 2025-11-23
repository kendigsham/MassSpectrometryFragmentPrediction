import csv
import os
#import json

def save_latent_dict(filename,smile_latent_dict):
    with open(filename,'wb') as fff:
      csvfile = csv.writer(fff)
      for key ,val in smile_latent_dict.iteritems():
        temp_val = list(val.flat)
        temp_val.insert(0,key)
        csvfile.writerow(temp_val)
        
def convert_dict(line):
  smile = []
  latent_list = []
  smile = line[0]
  templist = line[1:]
  for item in templist:
    latent_list.append(float(item))
  return smile, latent_list

def load_latent_dict(filename):	
  smile_latent_dict ={}
  with open(filename,'rb') as fff:
    csvfile = csv.reader(fff,quoting=csv.QUOTE_MINIMAL)
    for line in csvfile:
      smile, latent_list = convert_dict(line)
#      print type(smile), type(latent_list[0])
#      print 'hihi'
      smile_latent_dict[smile] = latent_list
  return smile_latent_dict

def writetofile(result_file,score,binn,num_of_mol,num_of_positive):
  line = str(binn)  + ',' +str(score) +','+ str(num_of_mol) +','+ str(num_of_positive)
  with open (result_file,'a') as www:
    www.write(line+'\n')

def read_file(result_file):
    mz_value = {'None':1}
    if os.path.isfile(result_file):
        mode = 'r'
        with open(result_file,mode) as fff:
          for line in fff:
            templine = line.split(',')
            mz_value[templine[0]] = 1

        return mz_value
    else:
        mode = 'w'
        first_line = 'm/z_value,auc_score,num_of_molecule,num_of_positive\n'
        with open(result_file,mode) as www:
            www.write(first_line)
        return mz_value
  
def rename(oldname, newname):
  os.rename(oldname, newname)
  
def check_file(done_filename):
  if os.path.isfile(done_filename):
    return True
  else:
    return False