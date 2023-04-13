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

# EfficientDet 
# DownloadCkpt.py

import sys
import os
import time
import traceback
import numpy as np
import tarfile
import shutil
import tensorflow as tf


def download_checkpoint_file():

  try:
      #Download checkpoint file
      url = "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d1.tar.gz"
      folder = "efficientdet-d1"
      tar_file = "efficientdet-d1.tar.gz"

      if os.path.exists(folder) != True:
          print("Try download {}".format(url))

          tar_file = tf.keras.utils.get_file(tar_file, url)
          print("You have downloaded {}".format(tar_file))

          with tarfile.open(tar_file, "r:gz") as tar:
             def is_within_directory(directory, target):
                 
                 abs_directory = os.path.abspath(directory)
                 abs_target = os.path.abspath(target)
             
                 prefix = os.path.commonprefix([abs_directory, abs_target])
                 
                 return prefix == abs_directory
             
             def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
             
                 for member in tar.getmembers():
                     member_path = os.path.join(path, member.name)
                     if not is_within_directory(path, member_path):
                         raise Exception("Attempted Path Traversal in Tar File")
             
                 tar.extractall(path, members, numeric_owner=numeric_owner) 
                 
             
             safe_extract(tar)
      else:
          print("OK, you have the weight file {}!".format(tar_file))
       
  except Exception as ex:
    traceback.print_exc()


if __name__=="__main__":
         
  try:
      MODEL = "efficientdet-d0"
 
      ckpt_path = os.path.join(os.getcwd(), MODEL);

      download_checkpoint_file()

  except Exception as ex:
    traceback.print_exc()

