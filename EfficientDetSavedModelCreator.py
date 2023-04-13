# Copyright 2020 Google Research. All Rights Reserved.
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
# ==============================================================================

# Copyright 2020-2021 antillia.com Toshiyuki Arai

# EfficientDetSavedModelCreator.py

r"""Tool to inspect a model."""
import os
# <added date="2021/0810"> arai
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# </added>
import sys

import time
from typing import Text, Tuple, List

from absl import app
from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

import hparams_config
import inference
import utils
import traceback
import datetime

from tensorflow.python.client import timeline  # pylint: disable=g-direct-tensorflow-import

from EfficientDetModelInspector import EfficientDetModelInspector

class EfficientDetSavedModelCreator(EfficientDetModelInspector):
  """A simple helper class for inspecting a model."""

  def __init__(self, config):
    super().__init__(config)
    print("=== EfficientDetSavedModelCreator")

    if os.path.exists(self.ckpt_path) == False:
         message = "---- Ckpt_model '" + self.ckpt_path + "' was not found -----"
         raise Exception (message)
    
    if os.path.exists(self.ckpt_path):
      if os.path.exists(self.saved_model_dir):
        now = datetime.datetime.now()
        now_str= "{0:%Y-%m-%d_%H_%M_%S}".format(now)
        print("now {}".format(now_str))
        os.rename(self.saved_model_dir, self.saved_model_dir + "_" + now_str)

        #message = "---- Saved_model '" + self.saved_model_dir + "' already exists! -----"
        #raise Exception (message)
    

def main(_):
  saved_model_config  = ""
  if len(sys.argv)==2:
    saved_model_config      = sys.argv[1]
  else:
    raise Exception("Usage: python EfficientDetSavedModelCreator.py saved_model_config")
  
  creator = EfficientDetSavedModelCreator(saved_model_config)
  creator.run_model()


##
#
if __name__ == '__main__':
  
  logging.set_verbosity(logging.WARNING)
  tf.enable_v2_tensorshape()
  tf.disable_eager_execution()
  app.run(main)
