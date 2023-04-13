# 
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
# EvaluationResultsWriter.py
#
# 2021/10/13 Modified to write f-value to a result.csv file.

import os
import sys
import traceback

class EvaluationResultsWriter:

  def __init__(self, evaluation_results_file):
    print("=== COCOMetricsWriter __init__")

    self.evaluation_results_file = evaluation_results_file
    self.header = "epoch, f, mAP, mAP@50IoU, mAP@75IoU, mAP_smallm, AP_medium, mAP_large,"
    self.header = self.header + " AR@1, AR@10, AR@100, AR@100small, AR@100medium, AR@100large\n"

    #if os.path.exists(self.evaluation_results_file):
    #  os.remove(self.evaluation_results_file)
 
    try:
      if not os.path.exists(self.evaluation_results_file):
        with open(self.evaluation_results_file, "w") as f:
          print("===  __init__ header {}".format(self.header))
          
          f.write(self.header)
    except Exception as ex:
        traceback.print_exc()

  def write(self, e, results):
    SEP = ","
    NL  = "\n"
    
    try:
      with open(self.evaluation_results_file, "a") as f:

        ap     = str( results['AP']  )
        ap_50  = str( results['AP50'])
        ap_75  = str( results['AP75'])
        ap_s   = str( results['APs'] )
        ap_m   = str( results['APm'] )
        ap_l   = str( results['APl'] )
        ar_1   = str( results['ARmax1'] )
        ar_10  = str( results['ARmax10'] )
        ar_100 = str( results['ARmax100'] )
        ar_s   = str( results['ARs'] )
        ar_m   = str( results['ARm'] )
        ar_l   = str( results['ARl'] )
        f_v    = str( 2.0 * (float(ap)* float(ar_1))/(float(ap) + float(ar_1)) )
        line = str(e) + SEP + f_v  + SEP + ap   + SEP + ap_50 + SEP + ap_75  + SEP + ap_s + SEP + ap_m + SEP + ap_l 
        line = line   + SEP + ar_1 + SEP + ar_10 + SEP + ar_100 + SEP + ar_s + SEP + ar_m + SEP + ar_l + NL 
        print("=== AP & AR {}".format(line))
        f.write(line)
    except Exception as ex:
        traceback.print_exc()
  
  