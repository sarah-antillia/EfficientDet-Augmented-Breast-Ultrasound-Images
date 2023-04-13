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

# EarlyStopping.py
# 2021/10/13 

import os
import sys

class EarlyStopping():
  def __init__(self, patience=10, verbose=0):
      self._step    = 0
      self._f        = 0.0 
      self.patience = patience
      self.verbose  = verbose

  # Please override this method in a subclass derived from this class.
  def validate(self, e, ap, ar):
      pass
      return False

