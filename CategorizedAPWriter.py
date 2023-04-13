# 
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
# CategorizedAPWriter.py
#
# 2021/11/08 Write AP's for all categories to a result.csv file.

import os
import sys
import traceback
from LabelMapReader import LabelMapReader

class CategorizedAPWriter:

  def __init__(self, label_map_pbtxt, results_file):
    print("=== CategorizedAPWriter  {}".format(results_file))
    self.results_file = results_file
    reader = LabelMapReader()
    map, self.categories = reader.read(label_map_pbtxt)
    SEP = ","
    NL  = "\n"

    print("--- categories {}".format(self.categories))
    
    self.header = "epoch" + SEP
    for category in self.categories:
      self.header = self.header + category + SEP

    try:
      if not os.path.exists(self.results_file):
        with open(self.results_file, "w") as f:
          f.write(self.header + NL)
    except Exception as ex:
      traceback.print_exc()

    
  def write(self, e, results):
    SEP = ","
    NL  = "\n"
    
    try:
        with open(self.results_file, "a") as f:
          line = str(e) + SEP
          for category in self.categories:            
            key  = "AP_/" + category
            ap   = float(results[key])
            ap   = round(ap, 5)  #2021/11/20
            line = line +  str(ap) + SEP
             
          print("=== Categorized_AP {}".format(line))
          
          f.write(line  + NL)
          
    except Exception as ex:
        traceback.print_exc()
  
  