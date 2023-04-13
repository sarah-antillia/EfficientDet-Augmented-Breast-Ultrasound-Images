#******************************************************************************
#
#  Copyright (c) 2021 Antillia.com TOSHIYUKI ARAI. ALL RIGHTS RESERVED.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#******************************************************************************


# LabelMapReader.py
# 2021/11/05

#2021/11/12 toshiyuki.arai

"""
  Modified the following line in read method.
            # The following line will cause an error if line contained the line 'name: "Maximum_Width_in_Meters",', 
            # because "id" in "Width".
            #if "id" in line:
            if "id:" in line:
            

"""
# 2021/11/20  Modified read method to use if line.startswith("id:"):
# instead of if "id:" in line, to avoid parsing error when line
# contained 'name: Somethingid:Cat'
 
import os
import sys
import traceback

class LabelMapReader:

  def __init__(self):
    pass

  def read(self, label_map_file):
    id    = None
    name  = None
    items = {}
    classes = []
    with open(label_map_file, "r") as f:
        for line in f:
            line.replace(" ", "")
            line = line.strip()
            #print("--- line {}".format(line))
            if line.startswith("id:"):
                id = int(line.split(":")[1].replace(",", "").strip() )
            elif line.startswith("name:"):
                name = line.split(":")[1].replace(",", "").strip()
                name = name.replace("'", "").replace("\"", "")
            if id is not None and name is not None:
                classes.append(name)
                items[id]    = name
                
                id   = None
                name = None

    return items, classes



if __name__ == "__main__":
  label_map = "./label_map.pbtxt"

  try:
     reader = LabelMapReader()
     items, classes = reader.read(label_map)
     print("--- items   {}".format(items))
     print("--- classes {}".format(classes))


     for i in items:
       print("i {} name  {}".format(i, items[i]) )

     for i in range(len(classes)):
       try:
         class_name = classes[i]
         print("index {}  class {}".format(i, class_name))
       except:
         traceback.print_exc()


  except Exception as ex:
    traceback.print_exc()
