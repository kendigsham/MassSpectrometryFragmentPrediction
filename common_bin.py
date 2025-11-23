import glob

def getfilelist():
    filelist= glob.glob("/home/kenny/linux/project/*filtered_bin_names.txt")
#    filelist= glob.glob("/home/kenny/linux/temptemp/*.ms")
    print type(filelist), len(filelist)
    return filelist
    
def readlist(filename):
    with open(filename,'r') as rr:
        binlist=rr.read().splitlines()
    return binlist
    
def commonlist(lists):
    common = set(lists[0])
    print common
    for s in lists[1:]:
        common.intersection_update(s)
        print common
    return common
    
def common_bin():
    filelist=getfilelist()
    print filelist
    lists=[]
    for i in filelist:
       templist=readlist(i)
       lists.append(templist)
       print templist
    common = commonlist(lists)
    print common

common_bin()