import multiprocessing
import os
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

#Tensorflow2.0
gpu_id = 0	
#tf.config.experimental.set_visible_devices(gpu_id)
gpus = tf.config.experimental.list_physical_devices('GPU')
print("---- gpus {}".format(gpus))
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*6)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

