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

# 
# 2022/05/10 copyright (c) antillia.com
#
# ConfigCreator.py

import os
import sys
import glob
import traceback
from BatScriptCreator import BatScriptCreator

class ConfigCreator:

  def __init__(self, template_dir):
    self.template_dir = template_dir
    self.DATASET_NAME = "{DATASET_NAME}"
    self.PROJECT_NAME = "{PROJECT_NAME}"

  def run(self, dataset_name, project_name, output_dir):
    print("=== ProjectCreator run")
    pattern = self.template_dir + "/*.conf"
    configs = glob.glob(pattern)
    # 1: copy *.conf files in template_dir to output_dir
    for conf in configs:
      basename    = os.path.basename(conf)

      output_conf = os.path.join(output_dir, basename)
 
      tf = open(conf, "r")
      if tf == None:
        raise Exception("Failed to open conf file:{}".format(conf))
      lines = tf.readlines()
      tf.close()
      new_lines = []
      for line in lines:
        line = line.replace(self.DATASET_NAME, dataset_name)
        line = line.replace(self.PROJECT_NAME, project_name)
        new_lines.append(line)
      with open(output_conf, "w") as cf:
         cf.writelines(new_lines)
      print("=== Created {}".format(output_conf))
