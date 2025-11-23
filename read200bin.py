import numpy as np
import scipy.sparse as sp
import glob

row,col,data=[],[],[]


def getfilelist():
    filelist= glob.glob("/home/kenny/linux/spectra_massbank/*.ms")
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

def addfragment(bins,intensity,position):
    row.append(position)
    col.append(bins)
    data.append(intensity)

def openfile (filename,position):
    with open(filename,"r") as f:
        for line in f:
          if len(line)>2:
            line_trim=line.strip()
            bins, intensity = findfragment(line_trim)
            if bins !=0 and intensity !=0:
              bin_value=adjust_value(bins)
              addfragment(bin_value,intensity,position)

def makecscmatrix():
    cscmatrix = sp.csc_matrix((data,(row,col)))
    return cscmatrix
    
def nonemptycol(mtx,num_mol_filter):
    coomatrix=mtx.tocoo()
    #return all the indices of columns that are not empty
    colindex = coomatrix.col
    colindex=colindex.tolist()
    unique = set(colindex)
    non_empty=list(unique)
    non_empty_list=[]
    for i in non_empty:
      num=mtx.getcol(i).count_nonzero()
      if num >= num_mol_filter:
#          mol_num[str(i)]= num
        non_empty_list.append(i)
    return non_empty_list
        

# making the matrix with 1 and 0, the input is the rescale1000 matrix
# trying to maintain the dimension of the binary so that it is the same as the 
# rescale matrix
def binarymtx(mtx,non_empty_list):
#   create a column of empty sparse matrix
    binarymtx =sp.csc_matrix(np.zeros((mtx.shape[0],1),dtype=int))
    for index in range(len(non_empty_list)):
        i= non_empty_list[index]
        newcol = sp.lil_matrix(np.zeros((mtx.shape[0],1)))
        tempcol=mtx.getcol(i)
        ttt=tempcol.toarray().tolist()
#        changing the intensity class 1 / 0 , threshold is 0
        for i in range (len(ttt)):
            num=ttt[i][0]
            if num>0:
                newcol[i,0]=1
            else:
                newcol[i,0]=0
        binarymtx=sp.hstack([binarymtx,newcol])
#        remove the column of empty matrix
    binarymtx= binarymtx.tocsc()
    binarymtx=binarymtx[:,1:]
    return binarymtx

def everything(num_mol_filter):
    # get the list of file from the directory
    listt = getfilelist()
    lengthlist=len(listt)
    print(lengthlist)
    for i in range(lengthlist):
        tempname=listt[i]
        openfile(tempname,i)
#        make the matrix after everything has been read in
    mtx=makecscmatrix()
    print type(mtx), mtx.shape
    filename ='originalmtx.npz'
    sp.save_npz(filename,mtx)
    #    get the variance of the columns
    non_empty_col =nonemptycol(mtx,num_mol_filter)
#    writelist(non_empty_col,bin_num)
#     getting the class matrix
    binarymatrix =binarymtx(mtx,non_empty_col)
    print binarymatrix.shape, type(binarymatrix), '////////////'
    filename3 = 'binarymtx.npz'
    sp.save_npz(filename3,binarymatrix)
    


everything(5)
    
