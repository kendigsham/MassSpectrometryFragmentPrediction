import json
from chemspipy import ChemSpider
from PIL import Image
import requests
from StringIO import StringIO
import os

def get_image(inchi_key, filepath):
  cs = ChemSpider('c3f897f1-3a50-4288-8eb8-8deb9265be3f')
  for result in cs.search(inchi_key):
    if result != None:
      listt = list(str(result))

      kkk = listt[9:len(listt)-1]

      strr = ''.join(kkk)
      compound = cs.get_compound(int(strr))
      url = compound.image_url
    
      response = requests.get(url)
      img = Image.open(StringIO(response.content))
      
      common_name = result.common_name
      filename = filepath + '/' + common_name + '.png'

      img.save(filename)


def make_directory(bin_name):
  newpath = '/home/kenny/linux/project/' + bin_name 
  if not os.path.exists(newpath):
    os.makedirs(newpath)
    return newpath
  else:
    return newpath
  
def load_json(json_file):
  with open(json_file,'r') as rrr:
    dictionary = json.load(rrr)
    return dictionary
  
def everything(json_file):
  dic = load_json(json_file)
#  print dic
#  print dic
  for key,value in dic.iteritems():
    print len(value)
    path = make_directory(key)
    for inchi in value:
      get_image(inchi, path)
      
everything('inchi_keys.txt')