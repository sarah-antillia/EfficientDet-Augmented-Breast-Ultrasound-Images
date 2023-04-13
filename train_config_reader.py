import yaml
import os
import sys
import pprint
import traceback

import ruamel.yaml

def train_config_reader(config):

  with open(config) as fp:
    str_data = fp.read()
  data = ruamel.yaml.load(str_data)

  pprint.pprint(data)


def xtrain_config_reader(config):
    flags = None
    with open(config , 'r') as file:
        flags = yaml.load(file)

    pprint.pprint(flags)
   
if __name__=="__main__":
  try:
     config = sys.argv[1]
     train_config_reader(config)

  except:
    traceback.print_exc()

     