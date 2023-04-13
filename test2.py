import tensorflow as tf
from tensorflow import *

gpu_id = 0
print(tf.__version__)
if tf.__version__ >= "2.1.0":


    print("--------------")

    physical_devices = tf.config.list_physical_devices('GPU')
    print("{}".format(physical_devices))

    tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)

    tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
elif tf.__version__ >= "2.0.0":
    print("=================")
    #TF2.0
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(physical_devices[gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
else:
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=str(gpu_id), # specify GPU number
            allow_growth=True
        )
    )
    set_session(tf.Session(config=config))
 