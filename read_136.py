import numpy as np
import scipy.sparse as sp
import glob
import json

global_dict={}


def getfilelist():
    filelist= glob.glob("/home/kenny/linux/spectra-massbank/*.ms")
#    filelist= glob.glob("/home/kenny/linux/temptemp/*.ms")
    return filelist

def checkfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def checkint(string):
    try:
        int(string)
        return True
    except ValueError:
        return False
      
def check136(number):
  if number >=136 and number <137:
    return True
  else:
    return False

# check and adjust the value of the m/z value, to the nearest 0.005
# everything above and equal to 0.005 will be rounded down to 0.005
# everyhting below 0.005 will be rounded to 0.000
def adjust_value(bins):
  bin100=bins*100
  bin_list=list(str(bin100))
  string=[]
  for i in range(len(bin_list)):
    if bin_list[i]=='.':
      tempchar=bin_list[i+1]
      num =int(tempchar)
      if num <5:
        string=bin_list[:i]
        string.append('0')
      elif num >=5:
        string=bin_list[:i]
        string.append('5')        
  ss=map(str,string)
  join_string= ''.join(ss)
  try:
    bin_value=int(join_string)
    return bin_value
  except ValueError:
    print join_string,'!!!!!!!!!!'


def findfragment (line):
    if(len(line)>2):
        words = line.split()
        if(checkint(words[0]) or checkfloat(words[0])):
            return float(words[0]), float(words[1])
        else:
            return 0,0

def findsmiles(line):
    words = line.split()
    if words[0] == '>smiles':
      return words[1]
    
def find_bin_dict(bin_value,smiles,local_bin_dict):
  bin_val = local_bin_dict.get(str(bin_value))
  if bin_val == None:
#    print bin_value
    local_bin_dict[str(bin_value)] = [smiles]
  
def openfile (filename):
    smiles=None
    local_bin_dict={}
    with open(filename,"r") as f:
        for line in f:
          if len(line)>2:
            line_trim=line.strip()
            if smiles ==None:
              smiles = findsmiles(line_trim)
            bins, intensity = findfragment(line_trim)
            if bins !=0 and intensity !=0:
                if check136(bins):
                  bin_value=adjust_value(bins)
                  find_bin_dict(bin_value, smiles,local_bin_dict)
    return local_bin_dict
        
def combine_dict(local_bin_dict):
  for key, value in local_bin_dict.iteritems():
    smile_list = global_dict.get(key)
    if smile_list == None:
      global_dict[key] = value
    else:
      for item in value:
        templist = global_dict[key]
        if type(templist) is str:
            temp = [templist]
            temp.append(item)
        else:
            global_dict[key].append(item)
          
def count_mol(num_mol_filter):
  final_dict={}
  for key,val in global_dict.iteritems():
    if len(val) >=5:
      final_dict[key] = val
  return final_dict

def everything(num_mol_filter):
    # get the list of file from the directory
    listt = getfilelist()
    lengthlist=len(listt)
#    print(lengthlist)
    for i in range(lengthlist):
        tempname=listt[i]
        local_bin_dict = openfile(tempname)
        if len(local_bin_dict) !=0:
#          print len(local_bin_dict)
          combine_dict(local_bin_dict)
    final_dict = count_mol(num_mol_filter)
#    for k,v in final_dict.iteritems():
#      print k,len(v), v[0],'////'
    with open ('dictionary_136.txt','w') as www:
      json.dump(final_dict,www)
#    
everything(5)
    