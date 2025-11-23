#! /usr/bin/python 

import sys 
import argparse
 
#parser = argparse.ArgumentParser(description = 'this is a demo')
#parser.add_argument('-i','--input',help='Input file name',required=True)
#args=parser.parse_args()

print 'hello world'

total = len(sys.argv)

cmdargs=str(sys.argv)

print 'arg list %s' % cmdargs
print 'scirpt name %s' % str(sys.argv[0])
print 'first argument %s' % str(sys.argv[1])
text= str(sys.argv[2])
print text

with open(text,'r') as rr:
    print rr.read().splitlines()
