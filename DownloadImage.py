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
# DownloadImage.py

import sys
import os
import time
import traceback
import numpy as np
import tarfile
import shutil


def download_image_file(img_file):

  try:   
    path = os.path.join(os.getcwd(), "images")
    os.makedirs(path, exist_ok=True)
    local_image_path = os.path.join(path, img_file)
    if os.path.exists(local_image_path) != True:

         url = 'https://user-images.githubusercontent.com/11736571/77320690-099af300-6d37-11ea-9d86-24f14dc2d540.png'
         print("Downloading a file {}".format(url))

         img_file = tf.keras.utils.get_file(img_file, url)
         shutil.move(img_file, local_image_path)

         print("You have downloaded {}".format(local_image_path))
    else:
         print("Found a downloaded file {}".format(local_image_path))

    return local_image_path

  except Exception as ex:
    traceback.print_exc()


if __name__=="__main__":
         
  try:
      img_file="img.png"

      download_image_file(img_file)

  except Exception as ex:
    traceback.print_exc()

