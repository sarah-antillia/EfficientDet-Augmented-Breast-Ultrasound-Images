# Copyright 2022 antillia.com Toshiyuki Arai
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

# PredictionCSV2COCOJsonConverter.py
# 2022/03/14 to-arai
#  
import os
import sys
import json
import traceback
import glob

class PredictionCSV2COCOJsonConverter:

  def __init__(self, classes_file, image_pattern):
    print("===  PredictionCSV2COCOPredictionJson.__init__() classes_file:{}  image_pattern:{}".format(classes_file, image_pattern))

    self.classes = []
    with open(classes_file, "r") as f:
      all_class_names = f.readlines()
      for class_name in all_class_names:
        class_name = class_name.strip()
        if class_name.startswith("#") ==False:
          self.classes.append(class_name)
    print("==== classes {}".format(self.classes))
    #pattern = images_dir + "/*.jpg"
    self.image_list = glob.glob(image_pattern)
    if self.image_list == None or len(self.image_list) == 0:
      raise Exception("Not found image list in " + image_pattern) 

  def getImageIndex(self, label):
    index = -1
    for i, name in enumerate(self.image_list):
       if name.endswith(label):         
         index = i
         break
    return index

  def getCategoryIndex(self, label):
    index = -1
    for i, name in enumerate(self.classes):
       if name == label:
         index = i +1 #cagegory index
         break
    return index
    
  def run(self, all_predictions_dir, output_dir):
    print("=== PredictionCSV2COCOResultJson.run() start")
    print(" all_predictions_dir " + all_predictions_dir)
    predictions_csv = all_predictions_dir + "/all_prediction.csv"

    # Create prediction.json files from prediction.csv files  
    
    base_pred  = os.path.basename(predictions_csv)
    name       = base_pred.split(".")[0]
    print("--- base_pred_dir " + base_pred)
    print("--- prediction_csv {}".format(predictions_csv))
    predictions_list = self.convert(predictions_csv)
    predictions_json = {"predictions": predictions_list}
    prediction_coco_json = os.path.join(output_dir, name + ".json")
      
    with open(prediction_coco_json, 'w') as f:
      json.dump(predictions_json, f, ensure_ascii=False, indent=4)
      print("----- Write json {}".format(prediction_coco_json))


  def convert(self, prediction_csv):
    print("---- PredictionCSV2COCOResultJson.convert " + prediction_csv)
   
    predictions_list = []

    csv_f = open(prediction_csv, "r")
    all_lines  = csv_f.readlines()
    csv_f.close()
    all_predictions = []
    for n, l in enumerate(all_lines):
      if n >0:
        #Skip header line
        l = l.strip()
        all_predictions.append(l)
    #sprint("=== all predictions {}".format(all_predictions))
        
    predictions_list = []
    for i, prediction in enumerate(all_predictions):
      prediction = prediction.strip()
      print("--- {}  prediction {}".format(i, prediction))            
      imageId, label, confidence, minx, miny, maxx, maxy = prediction.split(",")
      print("--- imageId " + imageId)
      class_id = self.getCategoryIndex(label)
      category_id   = int(class_id) + 1   
      minx    = float(minx)
      miny    = float(miny)
      maxx    = float(maxx)
      maxy    = float(maxy)

      x      = round(minx, 2)
      y      = round(miny, 2)
           
      width  = round(maxx - minx, 2)
      height = round(maxy - miny, 2)
      image_id    = self.getImageIndex(imageId)
      if image_id == -1:
        raise Exception("Invalid imageId " + str(imageId))
      category_id = self.getCategoryIndex(label)
      if category_id == -1:
        raise Exception("Invalid categoryId " + label)
      score = float(confidence)
      pred = { "image_id": image_id, "category_id": category_id, 
               "bbox": [x,y,width,height], "score": score,}
      print("--- pred {}".format(pred))
      predictions_list.append(pred)
    return predictions_list  


# python ../../PredictionCSV2COCOJsonConverter.py ./predictions_csv_dir/ ./classes.txt ./images_dir /output_coco_prediction_dir

"""
prediction2cocojson.bat
python ../../PredictionCSV2COCOJsonConverter.py ^
  ./realistic_test_dataset_outputs ^
  ./classes.txt ^
  ./realistic_test_dataset ^
  ./coco_prediction

"""
if __name__ == "__main__":
  predictions_dir     = "./predictions_csv_dir/"
  classes_txt         = "./classes.txt"
  images_dir          = "./images_dir"
  output_annotation_dir = "./prediction_coco_json_dir"

  try:
    if len(sys.argv) == 5:
      predictions_dir = sys.argv[1]
      classes_txt     = sys.argv[2]
      images_dir      = sys.argv[3]
      output_annotation_dir = sys.argv[4]
    else:
      raise Exception("Invalid argument!")

    if os.path.exists(predictions_dir) == False:
      raise Exception("Not found cpt_predicitions_dir " + predictions_dir)
    if os.path.exists(classes_txt) == False:
      raise Exception("Not found classes_txt file " + classes_txt)
    if os.path.exists(images_dir) == False:
      raise Exception("Not found images_dir " + images_dir)

    if os.path.exists(output_annotation_dir) == False:
      os.makedirs(output_annotation_dir)

    #image_size = (1280, 720)
    converter = PredictionCSV2COCOJsonConverter(classes_txt, images_dir)
    converter.run(predictions_dir, output_annotation_dir)

  except:
    traceback.print_exc()
