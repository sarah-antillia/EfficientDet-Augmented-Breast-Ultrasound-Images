# Copyright 2022 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# 2022/05/15 copyright (c) antillia.com
#
# ProjectCreator.py

import os
import sys
import glob
import traceback
from BatScriptCreator import BatScriptCreator
from ConfigCreator import ConfigCreator

# python ProjectCreator.py jp_roadsigns Japanese_RoadSigns 512x512 90

usage = "python ProjectCreator.py dataset_name project_name image_size num_classes"

if __name__ == "__main__":
  #config template dir
  template_dir  = "./config_templates/"
  dataset_name  = ""  
  project_name  = ""  # project_folder_name
  image_size    = "512x512"
  num_classes   = ""
  output_dir    = "./projects"
  #bat template dir
  btemplate_dir = "./bat_templates"

  try:
    if len(sys.argv) == 5:
     dataset_name = sys.argv[1]
     project_name = sys.argv[2]
     image_size   = sys.argv[3]
     num_classes  = sys.argv[4]
     number       = int(num_classes)

    else:
      raise Exception(usage)

    
    [w, h] = image_size.split("x")
    w = int(w)
    h = int(h)
    if w <= 0 or h <=0:
      raise Exception("Invaldi image size {}".format(image_size))

    if number <1 or number >1000:
      raise Exception("Invalid num_classes " + str(num_classes))
    if not os.path.exists(template_dir):
      raise Exception("Not found template_dir " + template_dir)

    if not os.path.exists(btemplate_dir):
      raise Exception("Not found btemplate_dir " + btemplate_dir)

    output_dir = os.path.join("./projects", project_name)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
 
    configs_output_dir = os.path.join(output_dir + "/configs")
    if not os.path.exists(configs_output_dir):
      os.makedirs(configs_output_dir)
   
    config_creator = ConfigCreator(template_dir)
    config_creator.run(dataset_name, project_name, configs_output_dir)
     
    bat_output_dir = output_dir
    if not os.path.exists(bat_output_dir):
      os.makedirs(bat_output_dir)

    bat_creator = BatScriptCreator(btemplate_dir)
    bat_creator.run(project_name, image_size, num_classes, bat_output_dir)
  
  except:
    traceback.print_exc()
