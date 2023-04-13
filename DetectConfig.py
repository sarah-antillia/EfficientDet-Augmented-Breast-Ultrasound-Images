#******************************************************************************
#
#  Copyright (c) 2020-2021 Antillia.com TOSHIYUKI ARAI. ALL RIGHTS RESERVED.
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

#DetectConfig.py

class DetectConfig: 
  ##########################

  CONFIGURATION     = "configuration"

  MODEL_NAME        = "model_name"
  #efficientdet-d0
  
  LABEL_MAP_PBTXT   = "label_map_pbtxt"  

  LOG_DIR           = "log_dir"
  TENSORRT          = "tensorrt"
  USE_XLA           = "use_xla"          
  CKPT_DIR          = "ckpt_dir" 

  FREEZE            = "freeze"
  EXPORT_CKPT       = "export_ckpt"      
  SAVED_MODEL_DIR   = "saved_model_dir"  
  TFLITE_PATH       = "tflite_path"      
  BATCH_SIZE        = "batch_size"
  HPARAMS           = "hparams"
  #'Comma separated k=v pairs of hyperparameters or a module'
  #  ' containing attributes to use as hyperparameters.')
           
  SCORE_THRESH      = "score_thresh"
  MAX_OUTPUT_SIZE   = "max_output_size"
  NMS_METHOD        = "nms_method"

  RUNMODE           = "runmode"
  INPUT_IMAGE       = "input_image" 
  OUTPUT_IMAGE_DIR  = "output_image_dir"
  INPUT_VIDEO       = "input_video"
  OUTPUT_VIDEO      = "output_video"
  #DETECT_RESULTS_DIR= "detect_results_dir"
    
  #OUTPUT_DIR         = "output_dir"
  
  ##########################

 
  LINE_THICKNESS    = "line_thickness"
  
  THRESHOLD         = "threshold"
  
  NMS_METHOD        = "nms_method"
  #'nms method, hard or gaussian.'

  MAX_BOXES_TO_DRAW = "max_boxes_to_draw"
  MIN_SCORE_THRESH  = "min_score_thresh" 
  BM_RUNS           = "bm_runs"          
  THREADS           = "threads"          
  TRACE_FILENAME    = "trace_filename"   
  
  FILTERS           = "filters"

  def __init__(self):
    pass
    
