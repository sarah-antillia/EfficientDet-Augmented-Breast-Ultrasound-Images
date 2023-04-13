#******************************************************************************
#
#  Copyright (c) 2021 Antillia.com TOSHIYUKI ARAI. ALL RIGHTS RESERVED.
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

"""

This is based on view_tfrecords_tf2.py for tensorflow 2 in https://github.com/EricThomson/tfrecord-view,
which is modified by antillia.com 

The tf2 version is based in https://github.com/jschw/tfrecord-view/blob/master/tfrecord_view_gui.py

view_records.py:
Consume and display data from a tfrecord file: pulls image and bounding boxes for display
so you can make sure things look reasonabloe, e.g., after augmentation.

Part of tensorflow-view repo: https://github.com/EricThomson/tfrecord-view

"""

# TFRecordInspector.py
#
# 2021/11/05 sarah-antillia
# 2021/11/08 Added the following method
#  def save_objects_count(self, objects_count):

# This is for tensorflow 2

import os
import sys
sys.path.append("../../")
from pprint import pprint
import cv2
import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
import warnings
import traceback
from LabelMapReader import LabelMapReader


class TFRecordInspector:

  def __init__(self, tfrecord_filepath, label_map_filepath, output_dir, with_annotation=True):
    self.tfrecord_filepath = tfrecord_filepath
    self.label_map_filepath = label_map_filepath
    self.output_dir         = output_dir
    self.with_annotation    = with_annotation
    
    self.class_labels = {}
    reader = LabelMapReader()
    self.class_labels, self.classes, = reader.read(self.label_map_filepath)
    print("---- class_labels {}".format(self.class_labels))


  def parse_record(self, data_record):
       
    example = None
    try:
      feature = {
                  'image/encoded': tf.io.FixedLenFeature([], tf.string),
                  'image/object/class/label': tf.io.VarLenFeature(tf.int64),
                  'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                  'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                  'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                  'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
                  'image/filename':         tf.io.FixedLenFeature([], tf.string)
                 }
      example = tf.io.parse_single_example(data_record, feature)
       
    except:
      # 2021/11/18
      # if 'image/filename' were missing.
      feature = {
                  'image/encoded': tf.io.FixedLenFeature([], tf.string),
                  'image/object/class/label': tf.io.VarLenFeature(tf.int64),
                  'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                  'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                  'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                  'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
                  #'image/filename':         tf.io.FixedLenFeature([], tf.string)
                 }
      example = tf.io.parse_single_example(data_record, feature)

      #traceback.print_exc()
      
    return example

  # 2021/11/08 
  def save_objects_count(self, objects_count):
    objects_count_dir = os.path.join(self.output_dir, "objects_count")
    if os.path.exists(objects_count_dir) == False:
      os.makedirs(objects_count_dir)
    objects_count_path = os.path.join(objects_count_dir, "objects_count.csv")
    
    NL  = "\n"
    SEP = ","
    
    with open(objects_count_path, mode='w') as s:
              
      objects_count  = sorted(objects_count.items(), key=lambda x:x[0])
      # This returns a list
      print(objects_count)
      
      keys  = ""
      values = ""      
      for (key,value) in objects_count:
        keys   = keys   + str(key)   + SEP
        values = values + str(value) + SEP

      s.write(keys   + NL)
      s.write(values + NL)
    print("=== Saved objects_count file {}".format(objects_count_path))
    
    
  def extract_images(self):
    dataset = tf.data.TFRecordDataset([self.tfrecord_filepath])
    
    record_iterator = iter(dataset)
    num_records = dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()
    print("----- Num records {}".format(num_records))
    
    objects_count = {}

    for im_ind in range(num_records):
        #Parse and process example
        parsed_example = self.parse_record(record_iterator.get_next())
        encoded_image = parsed_example['image/encoded']
        image_np = tf.image.decode_image(encoded_image, channels=3).numpy()
        filename = ""
        try:
          filename = parsed_example['image/filename']
          filename = "{}".format(filename)
          filename = filename.strip('b').strip("'")
          #print("=== filename {}".format(filename))
        except:
          #If 'image/filename' were missing. 
          filename = str(im_ind) + ".jpg"
          traceback.print_exc()

        labels =  tf.sparse.to_dense( parsed_example['image/object/class/label'], default_value=0).numpy()
        x1norm =  tf.sparse.to_dense( parsed_example['image/object/bbox/xmin'], default_value=0).numpy()
        x2norm =  tf.sparse.to_dense( parsed_example['image/object/bbox/xmax'], default_value=0).numpy()
        y1norm =  tf.sparse.to_dense( parsed_example['image/object/bbox/ymin'], default_value=0).numpy()
        y2norm =  tf.sparse.to_dense( parsed_example['image/object/bbox/ymax'], default_value=0).numpy()
        num_bboxes = len(labels)

        height, width = image_np[:, :, 1].shape
        image_copy    = image_np.copy()
        image_rgb     = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        cv_rgb = image_rgb[:, :, ::-1]
        pil_image = Image.fromarray(cv_rgb)
        #print(" --- class_labels {}".format(self.class_labels))
        draw = ImageDraw.Draw(pil_image)
        pil_font = ImageFont.truetype("arial.ttf", 18)

        if num_bboxes > 0:
            x1 = np.int64(x1norm * width)
            
            x2 = np.int64(x2norm * width)
            y1 = np.int64(y1norm * height)
            y2 = np.int64(y2norm * height)
            for bbox_ind in range(num_bboxes):
                    bbox = (x1[bbox_ind], y1[bbox_ind], x2[bbox_ind], y2[bbox_ind])
                    #print("--- labels {}".format(labels))
                    category = labels[bbox_ind] # -1
                    #print("--- bbox_ind {}".format(bbox_ind))
                    
                    #print("--- category {}".format(category))
                    label_name = self.class_labels[category]
                    print("--- category_id {}  label_name {}".format(category, label_name))
                    
                    # Store the number of objects into objects_count dict.
                    if label_name not in objects_count:
                      objects_count[label_name] = 1
                    else:
                      count = int(objects_count[label_name]) +1 
                      objects_count.update({label_name: count})
                    
                    if self.with_annotation:
                      label_position = (bbox[0] + 5, bbox[1] + 5)
                      
                      draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], 
                          outline=(255, 0, 0),width=2)
                                    
                      draw.text(xy = label_position, text = label_name, 
                         font = pil_font, fill = (255,0,0,0) )
                      
        output_image_file = os.path.join(output_images_dir, filename)
        print("=== Saved image file {}".format(output_image_file))
        pil_image.save(output_image_file)
        
    self.save_objects_count(objects_count)
        
  
#
# python TFRecordInspector.py ./tfrecord/sample.tfrecord ./tfrecord/label_map.pbtxt ./output   [True/False]
# python TFRecordInspector.py ./tfrecord/valid/valid.tfrecord ./label_map.pbtxt ./output/valid [True/False]
# python TFRecordInsep1ector.py ./projects/USA_RoadSigns/train/train.tfrecord ./projects/USA_RoadSigns/label_map.pbtxt ./inspector/train
#
if __name__ == '__main__':
  #tf.disable_eager_execution()
  tfrecord_file     = ""
  label_map_pbtxt   = ""
  output_images_dir = ""
  with_annotation   = True
  
  try:
    if len(sys.argv) <4:
      raise Exception("Usage: python TFRecordInsepector.py ./tfrecord/sample.tfrecord ./tfrecord/label_map.pbtxt ./output")
    if len(sys.argv) >=4:
      tfrecord_file     = sys.argv[1]
      label_map_pbtxt   = sys.argv[2]
      output_images_dir = sys.argv[3]
    if len(sys.argv) == 5:
      with_annotation =  eval(sys.argv[4])
    tfrecords = None
    print(tfrecord_file)
    if "*" in tfrecord_file:
      tfrecords = glob.glob(tfrecord_file)
    elif os.path.exists(tfrecord_file):
      tfrecords = [tfrecord_file]
    else:
      raise Exception(" Not found " + tfrecord_file)

    if not os.path.exists(label_map_pbtxt):
        raise Exception(" Not found " + label_map_pbtxt)

    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
    print("=== tfrecord_file     {}".format(tfrecord_file))
    print("=== label_map_pbtxt   {}".format(label_map_pbtxt))
    print("=== output_images_dir {}".format(output_images_dir))
    print("=== with_annotation   {}".format(with_annotation))
    for tfrecord in tfrecords:
      inspector = TFRecordInspector(tfrecord, 
                                    label_map_pbtxt, 
                                    output_images_dir, 
                                    with_annotation=with_annotation)
      inspector.extract_images()
      
  except Exception as ex:
    traceback.print_exc()
