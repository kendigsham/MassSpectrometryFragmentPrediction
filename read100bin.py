import numpy as np
import scipy.sparse as sp
import glob
import operator

row,col,data=[],[],[]

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

# provide the desire bin number and check if the candidate bin is within this range to store it
# breaking it down to 100 to store the bin
def checkbin(bin_num,candidate):
    maxx=bin_num+1
    if candidate>=bin_num and candidate < maxx:
        candidate100=int(candidate*100)
        bin_value = candidate100%100
        return True, bin_value
    else:
        return False, 0

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

def openfile (filename,position,bin_num):
    with open(filename,"r") as f:
        for line in f:
            if (len(line)>2):
                bins, intensity = findfragment(line)
#                print bins,intensity
#                bins = math.floor(bins)
                true, bin_value = checkbin(bin_num,bins)
                print '======='
                if (true):
                    addfragment(bin_value,intensity,position)
                    print bin_value, '******************'
                else:
#                    the intensity has to be zero to maintain the size of the matrix
                    addfragment(bin_value,0,position)

def makecscmatrix():
    cscmatrix = sp.csc_matrix((data,(row,col)))
    return cscmatrix

def rescale(mtx):
    rrr = mtx.tocsr()
    maxcol = rrr.max(axis=1)
    maxlist = maxcol.toarray().tolist()
# changing to coo to find non zero column index
    coo = mtx.tocoo()
    print coo.shape
#    loop throught the list and do maths to each row
    numofrow=mtx.shape[0]  
#    change matrix to lil so items can be assigned
    lil_mtx=mtx.tolil()    
    for i in range(numofrow):
        maxnum=maxlist[i]
        rr = coo.getrow(i)
        aa=rr.tocoo()
#        get the column index with non zeros
        nonzeros= aa.col
        for j in nonzeros:
            lil_mtx[i,j]=lil_mtx[i,j]/maxnum*1000
    csc_mtx=lil_mtx.tocsc()
    return csc_mtx

def getvariance(mtx):
    fragmentvar=dict()
    coomatrix=mtx.tocoo()
    #return all the indices of columns that are not empty
    colindex = coomatrix.col
    colindex=colindex.tolist()
    unique = set(colindex)
    for i in unique:
#             get the whole column, it is still sparse matrix, convert to
#       array then further convert to list
        tempcol = mtx.getcol(i)
#        the total number of rows or data
        number = tempcol.shape[0]
        sqr = tempcol.copy()
        sqr.data **=2
        variance = sqr.sum() / number - tempcol.mean()**2
        fragmentvar[str(i)]=variance
    return fragmentvar
    
def nonemptycol(mtx):
    coomatrix=mtx.tocoo()
    #return all the indices of columns that are not empty
    colindex = coomatrix.col
    colindex=colindex.tolist()
    unique = set(colindex)
    non_empty=list(unique)
    return non_empty
        

def findfragmentmax10(fragmentvar):
    max20=[]
    varlist=sorted(fragmentvar.items(),key=operator.itemgetter(1))
    varlist.reverse()
    for i in range(10):
        max20.append(varlist[i][0])
    return max20

def writelist(max20,bin_num):
    filename=str(bin_num)+'_100bin.txt'
    with open(filename,'w') as www:
        for item in max20:
            www.write("%s\n" % item)

# making the matrix with 1 and 0, the input is the rescale1000 matrix
# trying to maintain the dimension of the binary so that it is the same as the 
# rescale matrix
def binarymtx(mtx,non_empty_list):
#   create a column of empty sparse matrix
    binarymtx =sp.csc_matrix(np.zeros((mtx.shape[0],1),dtype=int))
    for index in range(len(non_empty_list)):
        i= non_empty_list[index]
        newcol = sp.lil_matrix(np.zeros((2132,1)))
        tempcol=mtx.getcol(i)
        ttt=tempcol.toarray().tolist()
#        changing the intensity class 1 / 0 , threshold is 0
        for i in range (len(ttt)):
            num=ttt[i][0]
            if num>0:
                newcol[i,0]=1
                print newcol[i,0]
            else:
                newcol[i,0]=0
        binarymtx=sp.hstack([binarymtx,newcol])
#        remove the column of empty matrix
    binarymtx= binarymtx.tocsc()
    binarymtx=binarymtx[:,1:]
    return binarymtx

def everything(bin_num):
    # get the list of file from the directory
    list = getfilelist()
    lengthlist=len(list)
    print(lengthlist)
    for i in range(lengthlist):
        tempname=list[i]
        print(tempname)
        openfile(tempname,i,bin_num)
#        make the matrix after everything has been read in
    mtx=makecscmatrix()
    print type(mtx), mtx.shape
    print mtx
    filename=str(bin_num)+'_original.npz'
    sp.save_npz(filename,mtx)
#    print(mtx.shape,'!!!!!!!!!!!!!!!!!!!')
    rescale_matrix=rescale(mtx)
    #    saving the matrix into a npz file
    filename2=str(bin_num)+'_rescale_matrix.npz'
    sp.save_npz(filename2,rescale_matrix)
    #    get the variance of the columns
    non_empty_col=nonemptycol(rescale_matrix)
#    writelist(non_empty_col,bin_num)
    # getting the class matrix
    binarymatrix=binarymtx(rescale_matrix,non_empty_col)
    filename3=str(bin_num)+'_binarymtx.npz'
    sp.save_npz(filename3,binarymatrix)

bin_list=[54]

for index in range(len(bin_list)):
    everything(bin_list[index])
    
