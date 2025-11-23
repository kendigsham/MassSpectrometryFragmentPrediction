import numpy as np
import scipy.sparse as sp
import glob


#smilesnames=[]

def getfilelist():
#    filelist= glob.glob('/home/kenny/linux/spectra_massbank/*.ms')
    filelist= glob.glob('/home/kenny/linux/spectra_gnps/*.ms')
    return filelist
    # for name in filelist:
    #     print(name)

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

def findfragment (line):
    if(len(line)>2):
        words = line.split()
        if(checkint(words[0]) or checkfloat(words[0])):
            return words[0], words[1]

def findsmiles(line):
    if(len(line)>2):
        wordss=line.split()
        if(wordss[0]==">smiles"):
            return wordss[1]
        else:
            return None
    else: 
        return None

def openfile (filename,smilesnames):
    smile = None
    with open(filename,"r") as f:
        for line in f:
            if smile == None:
                smile=findsmiles(line)
            if smile != None:
                smilesnames.append(smile)
                break
#
#def exportsmiles():
#    with open ('smilesnames','w') as w:
#        for name in smilesnames:
#            w.write("%s\n" % name)


def get_smile_list():
    smilesnames=[]
    listt = getfilelist()
    lengthlist=len(listt)
    for i in range(lengthlist):
        tempname=listt[i]
        openfile(tempname,smilesnames)
    return smilesnames

lll =get_smile_list()

print len(lll)

print len(set(lll))