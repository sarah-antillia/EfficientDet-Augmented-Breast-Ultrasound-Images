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

#2021/02/10 
#mAParlyStopping.py

import os
import sys
from EarlyStopping import EarlyStopping 

class mAPEarlyStopping(EarlyStopping):
  def __init__(self, patience=5, verbose=0):
      super().__init__(patience, verbose)
      self._mAP = 0.0
      
  #2021/10/13
  #def validate(self, e, mAP):
  def validate(self, e, mAP, mAR):
       
      if self._mAP > mAP:
          #If mAP is not increasing
          self._step += 1
          print("=== mAPEarlyStopping epoch:{} step:{} prev mAP:{} > new mAP: {}".format(e, self._step,   
             round(self._mAP,4),
             round( mAP, 4) ))
          
          if self._step > self.patience:
              if self.verbose:
                  print('=== mAPEarlyStopping is validated')
              return True
      else:
          # self._mAP <= mAP
          print("=== mAPEarlyStopping epoch:{} step:{} prev mAP:{} <= new mAP:{}".format(e, self._step, 
           round(self._mAP, 4),
           round(mAP,4) ))

          self._step = 0
          self._mAP = mAP

      return False

