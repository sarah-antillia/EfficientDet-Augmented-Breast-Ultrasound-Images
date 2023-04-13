# ==============================================================================
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

# 2021/09/22 atlan-antillia

# EfficientDetObjectDetector.py

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

import inference2 as inference
import utils
import traceback

from tensorflow.python.client import timeline  # pylint: disable=g-direct-tensorflow-import

from EfficientDetModelInspector import EfficientDetModelInspector
from DetectResultsWriter      import DetectResultsWriter

class EfficientDetObjectDetector(EfficientDetModelInspector):
  """A simple helper class for inspecting a model."""

  def __init__(self, config):
    super().__init__(config)
    print("=== EfficientDetObjectDetector")
    if self.runmode == "saved_model_infer":
      if os.path.exists(self.saved_model_dir) == False:
         message = "Specified runmode='saved_model_infer', but the saved_model '" + self.saved_model_dir + "' was not found -----"
         raise Exception (message)
    #print("--- output_image_dir {}".format(self.output_image_dir))
    if self.output_image_dir != None:
       if os.path.exists(self.output_image_dir) == False:
          os.makedirs(self.output_image_dir)
          print("--- Created output_image_dir {}".format(self.output_image_dir))


  # Override saved_model_inference in EfficientDetModelInspector
  def saved_model_inference(self, image_path_pattern, output_dir, **kwargs):
    """Perform inference for the given saved model."""
    print("=== EfficientDetObjectDetector.saved_model_inference -----------")
    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        batch_size   = self.batch_size,
        use_xla      = self.use_xla,
        model_params = self.model_config.as_dict(),
        **kwargs)
    driver.load(self.saved_model_dir)
    
    # Serving time batch size should be fixed.
    batch_size = self.batch_size or 1
    all_files = list(tf.io.gfile.glob(image_path_pattern))
    #print('all_files=', all_files)
    num_batches = (len(all_files) + batch_size - 1) // batch_size

    for i in range(num_batches):
      batch_files = all_files[i * batch_size:(i + 1) * batch_size]
      height, width = self.model_config.image_size
      images = [Image.open(f) for f in batch_files]
      filenames = [f for f in batch_files]
      #print("--- filenames {}".format(filenames))
      if len(set([m.size for m in images])) > 1:
        # Resize only if images in the same batch have different sizes.
        images = [m.resize(height, width) for m in images]
      raw_images = [np.array(m) for m in images]
      size_before_pad = len(raw_images)
      if size_before_pad < batch_size:
        padding_size = batch_size - size_before_pad
        raw_images += [np.zeros_like(raw_images[0])] * padding_size

      detections_bs = driver.serve_images(raw_images)
      for j in range(size_before_pad):

        (image, detected_objects, objects_stats)= driver.visualize(self.filters, 
                                                                    raw_images[j], 
                                                                    detections_bs[j], 
                                                                    **kwargs)

        img_id = str(i * batch_size + j)

        filename = all_files[int(img_id)]
        name = os.path.basename(filename)
        output_image_path = os.path.join(output_dir, self.str_filters + name )
        #output_image_path = os.path.join(output_dir, img_id + '.jpg')
        
        Image.fromarray(image).save(output_image_path)
        print('=== writing file to %s' % output_image_path)
        detect_results_writer = DetectResultsWriter(output_image_path)
        print("=== Writing detected_objects and objects_stats to csv files")
        detect_results_writer.write(detected_objects, objects_stats)


def main(_):
  detect_config      = ""
  if len(sys.argv)==2:
    detect_config    = sys.argv[1]
  else:
    raise Exception("Usage: python EfficientDetObjectDetector.py detect_config")
  if os.path.exists(detect_config) == False:
    raise Exception("Not found detect_config {}".format(detect_config))
      
  detector = EfficientDetObjectDetector(detect_config)
  detector.run_model()


##
#
if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.enable_v2_tensorshape()
  tf.disable_eager_execution()
  app.run(main)
