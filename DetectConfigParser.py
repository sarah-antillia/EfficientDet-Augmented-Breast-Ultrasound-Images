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
# DetectConfigParser.py
#
import os
import sys
import glob
import json
from collections import OrderedDict
import pprint
import configparser 
import traceback

from DetectConfig import DetectConfig

class DetectConfigParser(DetectConfig):

  def __init__(self, detect_config):
    self.sfilters = ""
    self.detect_config = detect_config
    if not os.path.exists(self.detect_config):
      raise Exception("Not found " + self.detect_config)
      
    try:
      self.parse(self.detect_config)
      
    except Exception as ex:
      print(ex)
      
  def parse(self, detect_config):
    self.config = configparser.ConfigParser()
    self.config.read(detect_config)
    
    self.dump_all()

  def runmode(self):
    try:
      return self.config[self.CONFIGURATION][self.RUNMODE]
    except:
      return "savedmodel_inference"
      
  def model_name(self):
    try:
      return self.config[self.CONFIGURATION][self.MODEL_NAME]
    except:
      return "efficientdet-d0"

  def log_dir(self):
    try:
      return self.config[self.CONFIGURATION][self.LOG_DIR]
    except:
      return None

  def label_map_pbtxt(self):
    try:
      return self.config[self.CONFIGURATION][self.LABEL_MAP_PBTXT]
    except:
      return None

    
  def delete_logdir(self):
    try:
      return eval(self.config[self.CONFIGURATION][self.DELETE_LOGDIR])
    except:
      return False
    

  def batch_size(self):
    try:
      return int(self.config[self.CONFIGURATION][self.BATCH_SIZE])
    except:
      return 1

  def ckpt_dir(self):
    try:
      return self.config[self.CONFIGURATION][self.CKPT_DIR]
    except:
      return None
       
  def saved_model_dir(self):
    try:
      return self.config[self.CONFIGURATION][self.SAVED_MODEL_DIR]
    except:
      return None

  def use_xla(self):
    try:
      return eval(self.config[self.CONFIGURATION][self.USE_XLA])
    except:
      return False

  def tflite_path(self):
    try:
      return self.config[self.CONFIGURATION][self.TFLITE_PATH]
    except:
      return None
      
  def export_ckpt(self):
    try:
      return self.config[self.CONFIGURATION][self.EXPORT_CKPT]
    except:
      return None
  
  def hparams(self):
    try: 
      val = self.config[self.CONFIGURATION][self.HPARAMS]
      if val == 'None':
        return None
      return val
    except:
      return None

  def score_thresh(self):
    try:
      return float(self.config[self.CONFIGURATION][self.SCORE_THRESH])
    except:
      return 0.4

  def tensorrt(self):
    try:
      return eval(self.config[self.CONFIGURATION][self.TENSORRT])
    except:
      return False

  def output_image_dir(self):
    try:
      return self.config[self.CONFIGURATION][self.OUTPUT_IMAGE_DIR]
    except Exception as ex:
      return None
  
  def input_image(self):
    try:
      return self.config[self.CONFIGURATION][self.INPUT_IMAGE]
    except Exception as ex:
      return None
  
  def freeze(self):
    try:
      return eval(self.config[self.CONFIGURATION][self.FREEZE])
  
    except Excetion as ex:
      return False
    	
  def str_filters(self):
    return self.sfilters

  def filters(self):
    try:
      self.sfilters = ""
      str_filters = self.config[self.CONFIGURATION][self.FILTERS]
      if str_filters == "None":
        return None
      self.sfilters = str_filters

      self._filters = []
      if str_filters != None:
          tmp = str_filters.strip('[]').split(',')
          if len(tmp) > 0:
              for e in tmp:
                  e = e.lstrip()
                  e = e.rstrip()
                  self._filters.append(e)

                  """
                  if e in self.classes :
                    self.filters.append(e)
                  else:
                    print("Invalid label(class)name {}".format(e))
                  """              
      return self._filters

    except Exception as ex:
      return None

  # CONFIGURATION
  def output_image_dir(self):
    try:
      return self.config[self.CONFIGURATION][self.OUTPUT_IMAGE_DIR]
    except:
      return None

  def input_video(self):
    try:
      return self.config[self.CONFIGURATION][self.INPUT_VIDEO]
    except:
      return None

  def output_video(self):
    try:
      return self.config[self.CONFIGURATION][self.INPUT_VIDEO]
    except:
      return None

  # CONFIGURATION
  def line_thickness(self):
    try:
      return int(self.config[self.CONFIGURATION][self.LINE_THICKNESS])
    except:
      return 2

  def max_output_size(self):
    try:
      return int(self.config[self.CONFIGURATION][self.MAX_OUTPUT_SIZE])
    except:
      return 100

  def max_boxes_to_draw(self):
    try:
      return int(self.config[self.CONFIGURATION][self.MAX_BOXES_TO_DRAW])
    except:
      return 100
  
  def min_score_thresh(self):
    try:
      return float(self.config[self.CONFIGURATION][self.MIN_SCORE_THRESH])
    except:
      return 0.4

  def nms_method(self):
    try:
      return self.config[self.CONFIGURATION][self.NMS_METHOD]
    except:
      return "hard"
  
  def bm_runs(self):
    try:
      return int(self.config[self.CONFIGURATION][self.BM_RUNS])
    except:
      return 10

  def threads(self):
    try:
      return int(self.config[self.CONFIGURATION][self.THREADS])
    except:
      return 1

  def trace_filename(self):
    try:
      return self.config[self.CONFIGURATION][self.TRACE_FILENAME]
    except:
      return None

  def dump_all(self):
      
    print("model_name           {}".format(self.model_name() ))

    print("log_dir              {}".format(self.log_dir() ))

    print("label_map_pbtxt      {}".format(self.label_map_pbtxt() ))

    print("delete_logdir        {}".format(self.delete_logdir() ))

    print("batch_size           {}".format(self.batch_size() ))

    print("ckpt_dir             {}".format(self.ckpt_dir() ))
    
    print("saved_model_dir       {}".format(self.saved_model_dir() ))

    print("hparams              {}".format(self.hparams() ))
    
    print("output_image_dir     {}".format(self.output_image_dir() ))

    print("line_thickness       {}".format(self.line_thickness() ))
 
    print("max_boxes_to_draw    {}".format(self.max_boxes_to_draw() ))

    print("str_fiters            {}".format(self.str_filters() ))

    print("nms_method           {}".format(self.nms_method() ))

    print("filters              {}".format(self.filters() ))


if __name__ == "__main__":
  try:
    detect_config = "./projects/BloodCells/configs/detect.config"
    parser = DetectConfigParser(detect_config)
    
  except Exception as ex:
    traceback.print_exc()
    
