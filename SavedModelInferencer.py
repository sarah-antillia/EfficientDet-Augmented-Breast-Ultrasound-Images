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

# 2022/06/01 Copyright (C) antillia.com
# This code has been taken from model_inspect.py in google/automl/efficientdet.

r"""Tool to inspect a model."""
import os
import sys
sys.path.append("../../")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION']='false'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
from typing import Text, Tuple, List

from absl import app
from absl import flags
from absl import logging
import shutil

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

import hparams_config

#import inference
import inference2 as inference

import utils
from tensorflow.python.client import timeline  # pylint: disable=g-direct-tensorflow-import

from DetectResultsWriter  import DetectResultsWriter
from ModelInspector       import ModelInspector

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from PredictionCSV2COCOJsonConverter import PredictionCSV2COCOJsonConverter
from PredictionCOCOEvaluator import PredictionCOCOEvaluator

flags.DEFINE_string('ground_truth_json', '',
                    'File path to a coco_ground_truth_json to test_dataset.')

flags.DEFINE_string('classes_file', './classes.txt',
                    'File path for a clases.txt.')

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
FLAGS = flags.FLAGS


class SavedModelInferencer(ModelInspector):
  """A simple helper class for inspecting a model."""

  def __init__(self,
               model_name: Text,
               logdir: Text,
               tensorrt: Text = False,
               use_xla: bool = False,
               ckpt_path: Text = None,
               export_ckpt: Text = None,
               saved_model_dir: Text = None,
               tflite_path: Text = None,
               batch_size: int = 1,
               hparams: Text = '',
               **kwargs): 
    super().__init__(
               model_name,
               logdir,
               tensorrt,
               use_xla,
               ckpt_path,
               export_ckpt,
               saved_model_dir,
               tflite_path,
               batch_size,
               hparams,
               **kwargs)


  #2022/03/11 Modified to be able to write the detection results.
  #2022/03/14 Modified to write dection results into all_prediction_file csv file 
  def saved_model_inference(self, image_path_pattern, output_dir, **kwargs):
    """Perform inference for the given saved model."""
    print("=== ModelInspector.saved_model_inference -----------")
    
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

    all_predictions_csv = os.path.join(output_dir, "all_prediction.csv")
    NL = "\n"
    with open(all_predictions_csv, mode="w") as all_prediction_file:
      HEADER = "ImageID, Label, Confidence, XMin, YMin, XMax, YMax" + NL
      all_prediction_file.write(HEADER)

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
          (image, detected_objects, objects_stats)= driver.visualize(None, 
                                                                    raw_images[j], 
                                                                    detections_bs[j], 
                                                                    **kwargs)

          img_id = str(i * batch_size + j)

          filename = all_files[int(img_id)]
          name = os.path.basename(filename)
          output_image_path = os.path.join(output_dir, name )
        
          Image.fromarray(image).save(output_image_path)

          print('=== Writing an image with bboxes and labels to %s' % output_image_path)
          detect_results_writer = DetectResultsWriter(output_image_path)
          detect_results_writer.write_withname(name, detected_objects, objects_stats, all_prediction_file)
    
  
def main(_):
  if tf.io.gfile.exists(FLAGS.logdir) and FLAGS.delete_logdir:
    logging.info('Deleting log dir ...')
    tf.io.gfile.rmtree(FLAGS.logdir)
  if not os.path.exists(FLAGS.saved_model_dir):
    raise Exception("FATAL ERROR: Not found saved_model {}".format(FLAGS.saved_model_dir))
  inferencer = SavedModelInferencer(
      model_name=FLAGS.model_name,
      logdir=FLAGS.logdir,
      tensorrt=FLAGS.tensorrt,
      use_xla=FLAGS.use_xla,
      ckpt_path=FLAGS.ckpt_path,
      export_ckpt=FLAGS.export_ckpt,
      saved_model_dir=FLAGS.saved_model_dir,
      tflite_path=FLAGS.tflite_path,
      batch_size=FLAGS.batch_size,
      hparams=FLAGS.hparams,
      score_thresh=FLAGS.min_score_thresh,
      max_output_size=FLAGS.max_boxes_to_draw,
      nms_method=FLAGS.nms_method)
  inferencer.run_model(
      FLAGS.runmode,
      input_image=FLAGS.input_image,
      output_image_dir=FLAGS.output_image_dir,
      input_video=FLAGS.input_video,
      output_video=FLAGS.output_video,
      line_thickness=FLAGS.line_thickness,
      max_boxes_to_draw=FLAGS.max_boxes_to_draw,
      min_score_thresh=FLAGS.min_score_thresh,
      nms_method=FLAGS.nms_method,
      bm_runs=FLAGS.bm_runs,
      threads=FLAGS.threads,
      trace_filename=FLAGS.trace_filename)

  # 2022/06/01 Added the following lines to convert all_predictions.csv to prediction.json
  converter = PredictionCSV2COCOJsonConverter(FLAGS.classes_file, FLAGS.input_image)
  converter.run(FLAGS.output_image_dir, FLAGS.output_image_dir)

  # 2022/06/01 Added the follwing lines to compute coco map metrics from prediction json and ground-truth json
  evaluator = PredictionCOCOEvaluator(FLAGS.input_image)
  evaluator.run(FLAGS.output_image_dir, FLAGS.ground_truth_json, FLAGS.output_image_dir)

if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.enable_v2_tensorshape()
  tf.disable_eager_execution()
  app.run(main)
