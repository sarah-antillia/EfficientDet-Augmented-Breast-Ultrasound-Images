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
# TrainConfig.py
# 2020/12/17 atlan-antillia

import os
import sys
import glob
import json

import configparser
import traceback


class TrainConfig:
  #
  # class variables  
  PROJECT             = "project"
  NAME                = "name"
  OWNER               = "owner"

  
  HARDWARE            = "hardware"
  TPU                 = "tpu"
  TPU_ZONE            = "tpu_zone"
  GCP_PROJECT         = "gcp_project"
  STRATEGY            = "strategy"
  USE_XLA             = "use_xla"

  
  MODEL               = "model"
  NAME                = "name"
  MODEL_DIR           = "model_dir"
 
  CKPT                = "ckpt"
  BACKBONE_CHECKPOINT = "backbone_checkpoint"
  PROFILE             = "profile"
  MODE                = "mode"
  
  
  TRAINING            = "training"
  BATCH_SIZE          = "batch_size"
  EPOCHS              = "epochs"
  SAVE_CHECKPOINTS_STEPS     = "save_checkpoints_steps"
  RUN_EPOCH_IN_CHILD_PROCESS = "run_epoch_in_child_process"
  FILE_PATTERN        = "file_pattern"
  EXAMPLES_PER_EPOCH  = "examples_per_epoch"
  LABEL_MAP_PBTXT     = "label_map_pbtxt"
  
  HPARAMS             = "hparams"
  CORES               = "cores"
  USE_SPATIAL_PARTION = "use_spatial_partition"
  CORES_PER_REPLICA   = "cores_per_replica" 
  INPUT_PARTION_DIMS  = "input_partition_dims"
  TF_RANDOM_SEED      = "tf_random_seed"
  USE_FAKE_DATA       = "use_fake_data"
  TRAINING_LOSSES_FILE = "training_losses_file"

  NUM_EXAMPLES_PER_EPOCH = "num_examples_per_epoch" 

  VALIDATION          = "validation"
  #FILE_PATTERN       = "file_pattern"
  #BATCH_SIZE         = "batch_size"
  EVAL_SAMPLES        = "eval_samples"
  EVAL_DIR            = "eval_dir"
  ITERATIONS_PER_LOOP = "iterations_per_loop"
  VAL_JSON_FILE       = "val_json_file"
  EVAL_AFTER_TRAIN    = "eval_after_train"
  TESTDEV_DIR         = "testdev_dir"
  MIN_EVAL_INTERVAL   = "min_eval_interalval"
  EVAL_TIMEOUT        = "eval_timeout"
  COCO_METRICS_FILE   = "coco_metrics_file"
  COCO_AP_PER_CLASS_FILE = "coco_ap_per_class_file"

  
  EPOCH_CHANGE_NOTIFIER = "epoch_change_notifier"
  
  INVALID             = -1

  EARLY_STOPPING      = "early_stopping"
  METRIC              = "metric"
  PATIENCE            = "patience"
   
  EPOCH_CHANGE_MONITOR = "epoch_change_monitor"
  ENABLED              = "enabled"
  IPADDRESS            = "ipaddress"
  PORT                 = "port"

  
  #
  # Constructor
  def __init__(self):
     pass
     
  def parse_if_possible(self, val):
   val = str(val)
   if val == "None":
     return None
   if val == "False":
     return False
   if val == "True":
     return True
   return val
   
      
  
  # We assume that the format of the annotation file is a coco.json format
  # which may be under the "dataset/{dataset_name}/train, 
  # The file name may be something like "_annotations.coco.json"
  
  def find_annotation_file(self, dir):
    annotation_file = ""
    
    pattern = dir + "/train/*.json"
    print("=== pattern {}".format(pattern))
    
    json_files = glob.glob(pattern)
    print("=== find_annotation_file {}".format(json_files))
    
    if len(json_files) == 1:
      annotation_file = json_files[0]
    else:
      raise Exception("Not found a json annotation file {}".format(json_files))
      
    return annotation_file
    
    #annotation_file will be the name like as "../dataset/{dataset_name}/train/_annotations.coco.json"    #json


  # To get a list of class-names from the json_annotation file.
  def get_classes(self, json_file):
    print("=== get_classes")
    classes = []
    
    if json_file is not None:
        with open(json_file,'r') as f:
         js = json.loads(f.read())
         categories =js['categories']
         for values in categories:
           cname = values['name']
           id    = values['id']
           if id>0:
             classes.append(cname)
             
    print("classes {}".format(classes))
    return classes

