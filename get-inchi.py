import numpy as np
import scipy.sparse as sp
import glob
import json

inchi_list =[]
inchi =None

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
    bin_value=float(join_string)
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

def findinchi(line):
    words = line.split()
    if words[0] == '>InChIKey':
      global inchi
      inchi = words[1]
#      print inchi

  
def check_value(value, bin_value):
  if bin_value ==value:
    return True 
  else:
    return False

def openfile (filename,position, value):
    boo = False
    with open(filename,"r") as f:
        for line in f:
          if len(line)>2:
            line_trim=line.strip()
            global inchi
            if inchi == None:
              findinchi(line_trim)
              
            bins, intensity = findfragment(line_trim)
            if bins !=0 and intensity !=0:
#                if check136(bins):
              bin_value=adjust_value(bins)
              if bin_value == value:
#                print bin_value
                boo = True
    if boo == True:
      global inchi
      inchi_list.append(inchi)
    
      
    
#    the value of bin to be check is *1000 with no decimal places
def get_inchi_list(value):
    # get the list of file from the directory
    listt = getfilelist()
    lengthlist=len(listt)
    print(lengthlist)
    for i in range(lengthlist):
        tempname=listt[i]
        openfile(tempname, i, value)
        global inchi
        inchi =None
#        make the matrix after everything has been read in

#    print len(inchi_list)
#    print inchi_list
    return inchi_list

def everything(bin_num_list):
  dictionary={}
  for bin_name in bin_num_list:
    templist = get_inchi_list(bin_name)
    print len(templist), '======='
    dictionary[str(bin_name)] = templist
    global inchi_list
    inchi_list = []
  with open('inchi_keys.txt','w') as www:
    json.dump(dictionary,www)


bin_list=[136020,136110,136060]

everything(bin_list)



