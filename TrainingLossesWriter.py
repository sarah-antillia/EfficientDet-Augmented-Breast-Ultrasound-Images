# Copyright 2020-2021 antillia.com Toshiyuki Arai
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
# TrainingLossesWriter.py
#

import os
import sys

class TrainingLossesWriter:

  def __init__(self, training_losses_file):
    self.training_losses_file = training_losses_file
    #if os.path.exists(self.training_losses_file):
    #  os.remove(self.training_losses_file)
    try:
      if not os.path.exists(self.training_losses_file):
        with open(self.training_losses_file, "w") as f:
          header = "epoch, box_loss, cls_loss, loss\n"
          f.write(header)
    except Exception as ex:
        traceback.print_exc()


  def write(self, e, results):
     SEP = ","
     NL  = "\n"

     try:

       with open(self.training_losses_file, "a") as f:
         box_loss = str( results['box_loss']  )
         cls_loss = str( results['cls_loss'])
         loss     = str( results['loss'])
         line = str(e) + SEP + box_loss + SEP + cls_loss + SEP + loss + NL

         f.write(line)
       
     except Exception as ex:
        traceback.print_exc()

  
