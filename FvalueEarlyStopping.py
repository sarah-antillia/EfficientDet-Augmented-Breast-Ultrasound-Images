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

# FvaluEarlyStopping.py
# 2021/10/13 

import os
import sys
from EarlyStopping import EarlyStopping 

class FvalueEarlyStopping(EarlyStopping):
  def __init__(self, patience=10, verbose=0):
      super().__init__(patience, verbose)
      self._f = 0.0

  def validate(self, e, ap, ar):
      f   = 2.0 * (float(ap)* float(ar))/(float(ap) + float(ar))
      if self._f > f:
          #If f is not increasing 
          self._step += 1
          print("=== FvalueEarlyStopping epoch:{}   -- patience:{}  step:{} prev f:{} > new f: {}".format(e, 
                    self.patience, self._step, self._f, f))
           
          if self._step > self.patience:
              if self.verbose:
                  print('=== FvalueEarlyStopping is validated')
              return True
      else:
          # self._f <= f
          print("=== FvalueEarlyStopping epoch:{}   -- patience:{}  step:{} prev f:{} <= new f:{}".format(e,
                    self.patience, self._step, self._f, f))
                    
          self._step = 0
          self._f = f

      return False

