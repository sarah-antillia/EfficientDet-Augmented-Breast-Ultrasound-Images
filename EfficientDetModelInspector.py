# Copyright 2020 Google Research. All Rights Reserved.
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
# ==============================================================================

# Copyright 2020-2021 antillia.com Toshiyuki Arai

# EfficientDetModelInspector.py


r"""Tool to inspect a model."""
import os
# <added date="2021/0810"> arai
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# </added>
import sys

import time
from typing import Text, Tuple, List

from absl import app
from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

import hparams_config
#import inference
import inference2 as inference

import utils
import traceback
import pprint
from tensorflow.python.client import timeline  # pylint: disable=g-direct-tensorflow-import

from DetectConfigParser       import DetectConfigParser

# This is take from google/autom/efficientdet/model_inspector

class EfficientDetModelInspector(object):
  """A simple helper class for inspecting a model."""

  def __init__(self, config):   #, detect_file_or_dir):
    self.parser             = DetectConfigParser(config)

    self.model_name         = self.parser.model_name()
    self.logdir             = self.parser.log_dir()
    self.delete_logdir      = self.parser.delete_logdir()
    self.tensorrt           = self.parser.tensorrt()
    self.use_xla            = self.parser.use_xla()
    
    self.ckpt_path          = self.parser.ckpt_dir()
    #print("--- ckpt_path {}".format(self.ckpt_path))
    self.export_ckpt        = self.parser.export_ckpt()
    self.saved_model_dir    = self.parser.saved_model_dir()
    #print("---- saved_model_dir {}".format(self.saved_model_dir))
    
    self.tflite_path        = self.parser.tflite_path()
    self.batch_size         = self.parser.batch_size()
    self.hparams            = self.parser.hparams()
    self.score_thresh       = self.parser.min_score_thresh()
    self.max_output_size    = self.parser.max_boxes_to_draw()
    self.nms_method         = self.parser.nms_method()

    self.runmode            = self.parser.runmode()
    self.input_image        = self.parser.input_image()
    self.output_image_dir   = self.parser.output_image_dir()

    self.input_video        = self.parser.input_video()
    self.output_video       = self.parser.output_video()
    self.line_thickness     = self.parser.line_thickness()
    self.max_boxes_to_draw  = self.parser.max_boxes_to_draw()
    self.min_score_thresh   = self.parser.min_score_thresh()

    self.bm_runs            = self.parser.bm_runs()
    self.threads            = self.parser.threads()
    self.trace_filename     = self.parser.trace_filename()
    # 2021/09/23 arai
    self.filters            = self.parser.filters()
    self.str_filters        = self.parser.str_filters()

    #print(" -------------- {}".format(self.filters))
    if tf.io.gfile.exists(self.logdir) and self.delete_logdir:
      logging.info('Deleting log dir ...')
      tf.io.gfile.rmtree(self.logdir)

    model_config = hparams_config.get_detection_config(self.model_name)
    print("=== hparams {}".format(self.hparams))
    if self.hparams != None:
      model_config.override(self.hparams)  # Add custom overrides
    model_config.is_training_bn = False
    model_config.image_size = utils.parse_image_size(model_config.image_size)

    # If batch size is 0, then build a graph with dynamic batch size.
    #self.batch_size = batch_size or None
    self.labels_shape = [self.batch_size, model_config.num_classes]
    # A hack to make flag consistent with nms configs.
    if self.min_score_thresh != None:
      model_config.nms_configs.score_thresh = self.score_thresh
    if self.nms_method != None:
      model_config.nms_configs.method = self.nms_method
    if self.max_output_size != None:
      model_config.nms_configs.max_output_size = self.max_output_size

    height, width = model_config.image_size
    if model_config.data_format == 'channels_first':
      self.inputs_shape = [self.batch_size, 3, height, width]
    else:
      self.inputs_shape = [self.batch_size, height, width, 3]

    self.model_config = model_config
    print("--- runmode {}".format(self.runmode))

 
  def build_model(self, inputs: tf.Tensor) -> List[tf.Tensor]:
    """Build model with inputs and labels and print out model stats."""
    logging.info('start building model')
    cls_outputs, box_outputs = inference.build_model(
        self.model_name,
        inputs,
        **self.model_config)

    # Write to tfevent for tensorboard.
    train_writer = tf.summary.FileWriter(self.logdir)
    train_writer.add_graph(tf.get_default_graph())
    train_writer.flush()

    all_outputs = list(cls_outputs.values()) + list(box_outputs.values())
    return all_outputs

  def export_saved_model(self, **kwargs):
    """Export a saved model for inference."""
    tf.enable_resource_variables()
    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        batch_size=self.batch_size,
        use_xla=self.use_xla,
        model_params=self.model_config.as_dict(),
        **kwargs)
    driver.build()
    driver.export(self.saved_model_dir, self.tflite_path, self.tensorrt)

  def saved_model_inference(self, image_path_pattern, output_dir, **kwargs):
    """Perform inference for the given saved model."""
    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        batch_size=self.batch_size,
        use_xla=self.use_xla,
        model_params=self.model_config.as_dict(),
        **kwargs)
    driver.load(self.saved_model_dir)

    # Serving time batch size should be fixed.
    batch_size = self.batch_size or 1
    all_files = list(tf.io.gfile.glob(image_path_pattern))
    print('all_files=', all_files)
    num_batches = (len(all_files) + batch_size - 1) // batch_size

    for i in range(num_batches):
      batch_files = all_files[i * batch_size:(i + 1) * batch_size]
      height, width = self.model_config.image_size
      images = [Image.open(f) for f in batch_files]
      
      if len(set([m.size for m in images])) > 1:
        # Resize only if images in the same batch have different sizes.
        images = [m.resize(height, width) for m in images]
      raw_images = [np.array(m) for m in images]
      size_before_pad = len(raw_images)
      if size_before_pad < batch_size:
        padding_size = batch_size - size_before_pad
        raw_images += [np.zeros_like(raw_images[0])] * padding_size

      detections_bs = driver.serve_images(raw_images)
      for j in range(size_before_pad):
        img = driver.visualize(raw_images[j], detections_bs[j], **kwargs)
        img_id = str(i * batch_size + j)
        output_image_path = os.path.join(output_dir, img_id + '.jpg')
        Image.fromarray(img).save(output_image_path)
        print('writing file to %s' % output_image_path)

  def saved_model_benchmark(self,
                            image_path_pattern,
                            trace_filename=None,
                            **kwargs):
    """Perform inference for the given saved model."""
    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        batch_size=self.batch_size,
        use_xla=self.use_xla,
        model_params=self.model_config.as_dict(),
        **kwargs)
    driver.load(self.saved_model_dir)
    raw_images = []
    all_files = list(tf.io.gfile.glob(image_path_pattern))
    if len(all_files) < self.batch_size:
      all_files = all_files * (self.batch_size // len(all_files) + 1)
    raw_images = [np.array(Image.open(f)) for f in all_files[:self.batch_size]]
    driver.benchmark(raw_images, trace_filename)

  def saved_model_video(self, video_path: Text, output_video: Text, **kwargs):
    """Perform video inference for the given saved model."""
    import cv2  # pylint: disable=g-import-not-at-top

    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        batch_size=1,
        use_xla=self.use_xla,
        model_params=self.model_config.as_dict())
    driver.load(self.saved_model_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      print('Error opening input video: {}'.format(video_path))

    out_ptr = None
    if output_video:
      frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
      out_ptr = cv2.VideoWriter(output_video,
                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25,
                                (frame_width, frame_height))

    while cap.isOpened():
      # Capture frame-by-frame
      ret, frame = cap.read()
      if not ret:
        break

      raw_frames = [np.array(frame)]
      detections_bs = driver.serve_images(raw_frames)
      new_frame = driver.visualize(raw_frames[0], detections_bs[0], **kwargs)

      if out_ptr:
        # write frame into output file.
        out_ptr.write(new_frame)
      else:
        # show the frame online, mainly used for real-time speed test.
        cv2.imshow('Frame', new_frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  def inference_single_image(self, image_image_path, output_dir, **kwargs):
    driver = inference.InferenceDriver(self.model_name, self.ckpt_path,
                                       self.model_config.as_dict())
    #driver.inference(image_image_path, output_dir, **kwargs)
    driver.inference(self.filters, image_image_path, output_dir, **kwargs)

  def build_and_save_model(self):
    """build and save the model into self.logdir."""
    with tf.Graph().as_default(), tf.Session() as sess:
      # Build model with inputs and labels.
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      outputs = self.build_model(inputs)

      # Run the model
      inputs_val = np.random.rand(*self.inputs_shape).astype(float)
      labels_val = np.zeros(self.labels_shape).astype(np.int64)
      labels_val[:, 0] = 1

      if self.ckpt_path:
        # Load the true weights if available.
        inference.restore_ckpt(sess, self.ckpt_path,
                               self.model_config.moving_average_decay,
                               self.export_ckpt)
      else:
        sess.run(tf.global_variables_initializer())
        # Run a single train step.
        sess.run(outputs, feed_dict={inputs: inputs_val})

      all_saver = tf.train.Saver(save_relative_paths=True)
      all_saver.save(sess, os.path.join(self.logdir, self.model_name))

      tf_graph = os.path.join(self.logdir, self.model_name + '_train.pb')
      with tf.io.gfile.GFile(tf_graph, 'wb') as f:
        f.write(sess.graph_def.SerializeToString())

  def eval_ckpt(self):
    """build and save the model into self.logdir."""
    with tf.Graph().as_default(), tf.Session() as sess:
      # Build model with inputs and labels.
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      self.build_model(inputs)
      inference.restore_ckpt(sess, self.ckpt_path,
                             self.model_config.moving_average_decay,
                             self.export_ckpt)

  def freeze_model(self) -> Tuple[Text, Text]:
    """Freeze model and convert them into tflite and tf graph."""
    with tf.Graph().as_default(), tf.Session() as sess:
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      outputs = self.build_model(inputs)

      if self.ckpt_path:
        # Load the true weights if available.
        inference.restore_ckpt(sess, self.ckpt_path,
                               self.model_config.moving_average_decay,
                               self.export_ckpt)
      else:
        # Load random weights if not checkpoint is not available.
        self.build_and_save_model()
        checkpoint = tf.train.latest_checkpoint(self.logdir)
        logging.info('Loading checkpoint: %s', checkpoint)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)

      # export frozen graph.
      output_node_names = [node.op.name for node in outputs]
      graphdef = tf.graph_util.convert_variables_to_constants(
          sess, sess.graph_def, output_node_names)

      tf_graph = os.path.join(self.logdir, self.model_name + '_frozen.pb')
      tf.io.gfile.GFile(tf_graph, 'wb').write(graphdef.SerializeToString())

      # export savaed model.
      output_dict = {'class_predict_%d' % i: outputs[i] for i in range(5)}
      output_dict.update({'box_predict_%d' % i: outputs[5+i] for i in range(5)})
      signature_def_map = {
          'serving_default':
              tf.saved_model.predict_signature_def(
                  {'input': inputs},
                  output_dict,
              )
      }
      output_dir = os.path.join(self.logdir, 'savedmodel')
      b = tf.saved_model.Builder(output_dir)
      b.add_meta_graph_and_variables(
          sess,
          tags=['serve'],
          signature_def_map=signature_def_map,
          assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
          clear_devices=True)
      b.save()
      logging.info('Model saved at %s', output_dir)

    return graphdef
  
  def benchmark_model(self,
                      warmup_runs,
                      bm_runs,
                      num_threads,
                      trace_filename=None):
    """Benchmark model."""
    if self.tensorrt:
      print('Using tensorrt ', self.tensorrt)
      graphdef = self.freeze_model()

    if num_threads > 0:
      print('num_threads for benchmarking: {}'.format(num_threads))
      sess_config = tf.ConfigProto(
          intra_op_parallelism_threads=num_threads,
          inter_op_parallelism_threads=1)
    else:
      sess_config = tf.ConfigProto()

    sess_config.graph_options.rewrite_options.dependency_optimization = 2
    if self.use_xla:
      sess_config.graph_options.optimizer_options.global_jit_level = (
          tf.OptimizerOptions.ON_2)

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      output = self.build_model(inputs)

      img = np.random.uniform(size=self.inputs_shape)

      sess.run(tf.global_variables_initializer())
      if self.tensorrt:
        fetches = [inputs.name] + [i.name for i in output]
        goutput = self.convert_tr(graphdef, fetches)
        inputs, output = goutput[0], goutput[1:]

      if not self.use_xla:
        # Don't use tf.group because XLA removes the whole graph for tf.group.
        output = tf.group(*output)
      else:
        output = tf.add_n([tf.reduce_sum(x) for x in output])

      output_name = [output.name]
      input_name = inputs.name
      graphdef = tf.graph_util.convert_variables_to_constants(
          sess, sess.graph_def, output_name)

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
      tf.import_graph_def(graphdef, name='')

      for i in range(warmup_runs):
        start_time = time.time()
        sess.run(output_name, feed_dict={input_name: img})
        logging.info('Warm up: {} {:.4f}s'.format(i, time.time() - start_time))

      print('Start benchmark runs total={}'.format(bm_runs))
      start = time.perf_counter()
      for i in range(bm_runs):
        sess.run(output_name, feed_dict={input_name: img})
      end = time.perf_counter()
      inference_time = (end - start) / bm_runs
      print('Per batch inference time: ', inference_time)
      print('FPS: ', self.batch_size / inference_time)

      if trace_filename:
        run_options = tf.RunOptions()
        run_options.trace_level = tf.RunOptions.FULL_TRACE
        run_metadata = tf.RunMetadata()
        sess.run(
            output_name,
            feed_dict={input_name: img},
            options=run_options,
            run_metadata=run_metadata)
        logging.info('Dumping trace to %s', trace_filename)
        trace_dir = os.path.dirname(trace_filename)
        if not tf.io.gfile.exists(trace_dir):
          tf.io.gfile.makedirs(trace_dir)
        with tf.io.gfile.GFile(trace_filename, 'w') as trace_file:
          trace = timeline.Timeline(step_stats=run_metadata.step_stats)
          trace_file.write(trace.generate_chrome_trace_format(show_memory=True))

  def convert_tr(self, graph_def, fetches):
    """Convert to TensorRT."""
    from tensorflow.python.compiler.tensorrt import trt  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    converter = trt.TrtGraphConverter(
        nodes_denylist=[t.split(':')[0] for t in fetches],
        input_graph_def=graph_def,
        precision_mode=self.tensorrt)
    infer_graph = converter.convert()
    goutput = tf.import_graph_def(infer_graph, return_elements=fetches)
    return goutput
  

  def run_model(self):
    rumode = self.runmode
    """Run the model on devices."""
    if self.runmode == 'dry':
      self.build_and_save_model()
    elif self.runmode == 'freeze':
      self.freeze_model()
    elif self.runmode == 'ckpt':
      self.eval_ckpt()
    elif self.runmode == 'saved_model_benchmark':
      self.saved_model_benchmark(
          self.input_image,
          trace_filename=self.trace_filename)
    elif self.runmode in ('infer', 'saved_model', 'saved_model_infer', 'saved_model_video'):
      config_dict = {}
      if self.line_thickness != None:
        config_dict['line_thickness'] = self.line_thickness
      if self.max_boxes_to_draw != None:
        config_dict['max_boxes_to_draw'] = self.max_boxes_to_draw
      if self.min_score_thresh != None:
        config_dict['min_score_thresh'] = self.min_score_thresh

      if self.runmode == 'saved_model':
        self.export_saved_model(**config_dict)
      elif self.runmode == 'infer':
        self.inference_single_image(self.input_image,
                                    self.output_image_dir, **config_dict)
      elif self.runmode == 'saved_model_infer':
        self.saved_model_inference(self.input_image,
                                   self.output_image_dir, **config_dict)
      elif self.runmode == 'saved_model_video':
        self.saved_model_video(self.input_video, self.output_video,
                               **config_dict)
    elif self.runmode == 'bm':
      self.benchmark_model(
          warmup_runs=5,
          bm_runs=self.bm_runs,
          num_threads=self.threads,
          trace_filename=self.trace_filename)
    else:
      raise ValueError('Unkown runmode {}'.format(runmode))


def main(_):
  inspect_config      = ""
  if len(sys.argv)==2:
    inspect_config      = sys.argv[1]
  else:
    raise Exception("Usage: python EfficientDetModelInspector.py inspect_config")
  
  detector = EfficientDetModelInspector(inspect_config)
  detector.run_model()


##
#
if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.enable_v2_tensorshape()
  tf.disable_eager_execution()
  app.run(main)
