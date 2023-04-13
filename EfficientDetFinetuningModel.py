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

"""The main training script."""
import multiprocessing
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import sys
from absl import app
#from absl import FLAGS
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

import dataloader

#import det_model_fn
#2021/11/17
import det_model_fn2 as det_model_fn

import hparams_config
import utils

import pprint

from io import StringIO 
from LabelMapReader          import LabelMapReader

from TrainConfigParser       import TrainConfigParser
from mAPEarlyStopping        import mAPEarlyStopping
from FvalueEarlyStopping     import FvalueEarlyStopping

from COCOMetricsWriter       import COCOMetricsWriter
from EpochChangeNotifier     import EpochChangeNotifier
from TrainingLossesWriter    import TrainingLossesWriter
from CategorizedAPWriter     import CategorizedAPWriter

# 2021/11/22 Updated
# Based on the latest main.py

class EfficientDetFinetuningModel(object):

  def __init__(self, train_config):
    self.TRAIN          = 'train'
    self.EVAL           = 'eval'
    self.TRAIN_AND_EVAL = 'train_and_eval'
    
    self.parser         = TrainConfigParser(train_config)
    self.model_dir      = self.parser.model_dir()
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)

    eval_dir                    = self.parser.eval_dir()
    print("=== eval_dir {}",format(eval_dir))
    if os.path.exists(eval_dir) == False:
      os.makedirs(eval_dir)

    training_losses_file           = self.parser.training_losses_file()
    print("=== training_losses_file{}".format(training_losses_file))

    self.label_map_pbtxt           = self.parser.label_map_pbtxt()
    labelMapReader                 = LabelMapReader()
    self.label_map, classes        = labelMapReader.read(self.label_map_pbtxt)
    print("=== label_map {}".format(self.label_map))

    self.training_losses_writer    = TrainingLossesWriter(training_losses_file)

    coco_ap_per_class_file         = self.parser.coco_ap_per_class_file()    
    print("=== coco_ap_per_class_file  {}".format(coco_ap_per_class_file ))

    self.categorized_ap_writer     = CategorizedAPWriter(self.label_map_pbtxt, coco_ap_per_class_file)
    
    coco_metrics_file              = self.parser.coco_metrics_file()
    print("=== coco_metrics_file {}".format(coco_metrics_file))
    
    self.coco_metrics_writer       = COCOMetricsWriter(coco_metrics_file)
    self.early_stopping_metric     = self.parser.early_stopping_metric()
    patience                       = self.parser.early_stopping_patience()
    self.early_stopping            = None
  

    if  patience > 0:
      # 2021/10/13
      if self.early_stopping_metric   == "map":
        self.early_stopping = mAPEarlyStopping(patience=patience, verbose=1) 
      elif self.early_stopping_metric == "fvalue":
        self.early_stopping = FvalueEarlyStopping(patience=patience, verbose=1)

 
    if self.parser.strategy() == 'tpu':
      tf.disable_eager_execution()
      tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          self.parser.tpu(), 
          zone    = self.parser.tpu_zone(), 
          project = self.parser.gcp_project() )
      tpu_grpc_url = tpu_cluster_resolver.get_master()
      tf.Session.reset(tpu_grpc_url)
    else:
      tpu_cluster_resolver = None


  def train(self):
    # Check data path
    # 2021/11/27 mode->mode()
    if self.parser.mode() in ('train', 'train_and_eval'):
      if self.parser.train_file_pattern() is None:
        raise RuntimeError('Must specify --train_file_pattern for train.')
    if self.parser.mode() in ('eval', 'train_and_eval'):
      if self.parser.val_file_pattern() is None:
        raise RuntimeError('Must specify --val_file_pattern for eval.')

    # Parse and override hparams
    config = hparams_config.get_detection_config(self.parser.model_name() )
    if self.parser.hparams():
      config.override(self.parser.hparams())
    if self.parser.num_epochs():  # NOTE: remove this flag after updating all docs.
      config.num_epochs = self.parser.num_epochs()
    print("--- config {}".format(config))
    # Parse image size in case it is in string format.
    print("----- config.image_size()".format(config.image_size) )

    config.image_size = utils.parse_image_size(config.image_size)

    # The following is for spatial partitioning. `features` has one tensor while
    # `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
    # partition is performed on `features` and all partitionable tensors of
    # `labels`, see the partition logic below.
    # In the TPUEstimator context, the meaning of `shard` and `replica` is the
    # same; follwing the API, here has mixed use of both.
    if self.parser.use_spatial_partition():
      # Checks input_partition_dims agrees with num_cores_per_replica.
      if self.parser.num_cores_per_replica() != np.prod(self.parser.input_partition_dims() ):
        raise RuntimeError('--num_cores_per_replica must be a product of array'
                           'elements in --input_partition_dims.')

      labels_partition_dims = {
          'mean_num_positives': None,
          'source_ids':         None,
          'groundtruth_data':   None,
          'image_scales':       None,
          'image_masks':        None,
      }
      # The Input Partition Logic: We partition only the partition-able tensors.
      feat_sizes = utils.get_feat_sizes(
          config.get('image_size'), config.get('max_level'))
      for level in range(config.get('min_level'), config.get('max_level') + 1):

        def _can_partition(spatial_dim):
          partitionable_index = np.where(
              spatial_dim % np.array(self.parser.input_partition_dims() ) == 0)
          return len(partitionable_index[0]) == len(self.parser.input_partition_dims() )

        spatial_dim = feat_sizes[level]
        if _can_partition(spatial_dim['height']) and _can_partition(
            spatial_dim['width']):
          labels_partition_dims['box_targets_%d' %
                                level] = self.parser.input_partition_dims()
          labels_partition_dims['cls_targets_%d' %
                                level] = self.parser.input_partition_dims()
        else:
          labels_partition_dims['box_targets_%d' % level] = None
          labels_partition_dims['cls_targets_%d' % level] = None
      num_cores_per_replica = self.parser.num_cores_per_replica()
      input_partition_dims = [self.parser.input_partition_dims(), labels_partition_dims]
      num_shards = self.parser.num_cores() // num_cores_per_replica
    else:
      num_cores_per_replica = None
      input_partition_dims = None
      num_shards = self.parser.num_cores()

    params = dict(
        config.as_dict(),
        model_name=self.parser.model_name(),
        iterations_per_loop=self.parser.iterations_per_loop(),
        model_dir=self.parser.model_dir(),
        num_shards=num_shards,
        num_examples_per_epoch=self.parser.num_examples_per_epoch(),
        strategy=self.parser.strategy(),
        backbone_ckpt=self.parser.backbone_ckpt(),
        ckpt=self.parser.ckpt(),
        val_json_file=self.parser.val_json_file(),
        testdev_dir=self.parser.testdev_dir(),
        profile=self.parser.profile(),
        mode=self.parser.mode() )
    config_proto = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    if self.parser.strategy() != 'tpu':
      if self.parser.use_xla():
        config_proto.graph_options.optimizer_options.global_jit_level = (
            tf.OptimizerOptions.ON_1)
        config_proto.gpu_options.allow_growth = True

    model_dir = self.parser.model_dir()
    model_fn_instance = det_model_fn.get_model_fn(self.parser.model_name() )
    max_instances_per_image = config.max_instances_per_image
    if self.parser.eval_samples():
      eval_steps = int((self.parser.eval_samples() + self.parser.eval_batch_size() - 1) //
                       self.parser.eval_batch_size() )
    else:
      eval_steps = None
    total_examples = int(config.num_epochs * self.parser.num_examples_per_epoch())
    train_steps = total_examples // self.parser.train_batch_size()
    logging.info(params)

    if not tf.io.gfile.exists(model_dir):
      tf.io.gfile.makedirs(model_dir)

    config_file = os.path.join(model_dir, 'config.yaml')
    if not tf.io.gfile.exists(config_file):
      tf.io.gfile.GFile(config_file, 'w').write(str(config))

    train_input_fn = dataloader.InputReader(
        self.parser.train_file_pattern(),
        is_training=True,
        use_fake_data=self.parser.use_fake_data(),
        max_instances_per_image=max_instances_per_image)
    eval_input_fn = dataloader.InputReader(
        self.parser.val_file_pattern(),
        is_training=False,
        use_fake_data=self.parser.use_fake_data(),
        max_instances_per_image=max_instances_per_image)

    if self.parser.strategy() == 'tpu':
      tpu_config = tf.estimator.tpu.TPUConfig(
          self.parser.iterations_per_loop() if self.parser.strategy() == 'tpu' else 1,
          num_cores_per_replica=num_cores_per_replica,
          input_partition_dims=input_partition_dims,
          per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig
          .PER_HOST_V2)
      run_config = tf.estimator.tpu.RunConfig(
          cluster=tpu_cluster_resolver,
          model_dir=model_dir,
          log_step_count_steps=self.parser.iterations_per_loop(),
          session_config=config_proto,
          tpu_config=tpu_config,
          save_checkpoints_steps=self.parser.save_checkpoints_steps(),
          tf_random_seed=self.parser.tf_random_seed(),
      )
      # TPUEstimator can do both train and eval.
      train_est = tf.estimator.tpu.TPUEstimator(
          model_fn=model_fn_instance,
          train_batch_size = self.parser.train_batch_size(),
          eval_batch_size  = self.parser.eval_batch_size(),
          config=run_config,
          params=params)
      eval_est = train_est
    else:
      strategy = None
      if self.parser.strategy() == 'gpus':
        strategy = tf.distribute.MirroredStrategy()
      run_config = tf.estimator.RunConfig(
          model_dir=model_dir,
          train_distribute=strategy,
          log_step_count_steps=self.parser.iterations_per_loop(),
          session_config=config_proto,
          save_checkpoints_steps=self.parser.save_checkpoints_steps(),
          tf_random_seed=self.parser.tf_random_seed(),
      )

      def get_estimator(global_batch_size):
        params['num_shards'] = getattr(strategy, 'num_replicas_in_sync', 1)
        params['batch_size'] = global_batch_size // params['num_shards']
        params['eval_dir']   = self.parser.eval_dir()  #2021/11/14
        params['label_map']  = self.label_map          #2021/11/14
        params['disable_per_class_ap'] = self.parser.disable_per_class_ap() #2021/11/15
        print("------------------------disable_per_class_ap {}".format(params['disable_per_class_ap']))
        return tf.estimator.Estimator(
            model_fn=model_fn_instance, config=run_config, params=params)

      # train and eval need different estimator due to different batch size.
      train_est = get_estimator(self.parser.train_batch_size() )
      eval_est = get_estimator(self.parser.eval_batch_size() )

    # start train/eval flow.
    if self.parser.mode() == 'train':
      train_est.train(input_fn=train_input_fn, max_steps=train_steps)
      if self.parser.eval_after_train():
        eval_est.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    elif self.parser.mode() == 'eval':
      # Run evaluation when there's a new checkpoint
      for ckpt in tf.train.checkpoints_iterator(
          self.parser.model_dir(),
          min_interval_secs = self.parser.min_eval_interval(),
          timeout           = self.parser.eval_timeout() ):

        logging.info('Starting to evaluate.')
        try:
          eval_results = eval_est.evaluate(
              eval_input_fn, 
              steps = eval_steps, 
              name  = self.parser.eval_name() )
          # Terminate eval job when final checkpoint is reached.
          try:
            current_step = int(os.path.basename(ckpt).split('-')[1])
          except IndexError:
            logging.info('%s has no global step info: stop!', ckpt)
            break

          utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)
          if current_step >= train_steps:
            logging.info('Eval finished step %d/%d', current_step, train_steps)
            break

        except tf.errors.NotFoundError:
          # Checkpoint might be not already deleted by the time eval finished.
          # We simply skip ssuch case.
          logging.info('Checkpoint %s no longer exists, skipping.', ckpt)

    elif self.parser.mode() == 'train_and_eval':
      ckpt = tf.train.latest_checkpoint(self.parser.model_dir())
      try:
        step = int(os.path.basename(ckpt).split('-')[1])
        current_epoch = (
            step * self.parser.train_batch_size() // self.parser.num_examples_per_epoch() )
        logging.info('found ckpt at step %d (epoch %d)', step, current_epoch)
      except (IndexError, TypeError):
        logging.info('Folder %s has no ckpt with valid step.', self.parser.model_dir() )
        current_epoch = 0

      def run_train_and_eval(e):
        print('\n   =====> Starting training, epoch: %d.' % e)
        # 2021/11/15 
        os.environ['epoch'] = str(e)

        train_est.train(
            input_fn=train_input_fn,
            max_steps=e * self.parser.num_examples_per_epoch() // self.parser.train_batch_size() )
        print('\n   =====> Starting evaluation, epoch: %d.' % e)
        eval_results = eval_est.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        

        ckpt = tf.train.latest_checkpoint(self.parser.model_dir() )
        utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)
        map  = eval_results['AP']
        loss = eval_results['loss']

        #self.epoch_change_notifier.epoch_end(e, loss, map)
        
        self.coco_metrics_writer.write(e, eval_results)
        # 2021/11/10
        self.categorized_ap_writer.write(e, eval_results)

        self.training_losses_writer.write(e, eval_results)

        ckpt = tf.train.latest_checkpoint(self.parser.model_dir() )
        utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)

        earlystopping = False
        if self.early_stopping != None:
          ap = eval_results['AP']
          ar = eval_results['ARmax1']
          earlystopping = self.early_stopping.validate(e, ap, ar)
          
        return earlystopping


      epochs_per_cycle = 1  # higher number has less graph construction overhead.
      for e in range(current_epoch + 1, config.num_epochs + 1, epochs_per_cycle):
        if self.parser.run_epoch_in_child_process():
          p = multiprocessing.Process(target=run_train_and_eval, args=(e,))
          p.start()
          p.join()
          if p.exitcode != 0:
            return p.exitcode
        else:
          tf.reset_default_graph()
          earlystopping = run_train_and_eval(e)
          if earlystopping:
            print("=== Early_stopping validated at epoch {}".format(e))
            break
            
    else:
      logging.info('Invalid mode: %s', self.parser.mode())

def main(_):
  train_config = ""
  if len(sys.argv)==2:
    train_config = sys.argv[1]
  else:
    raise Exception("Usage: python EfficientDetFinetuningModel.py train_config")

  if os.path.exists(train_config) == False:
    raise Exception("Not found train_config {}".format(train_config)) 
  
  model = EfficientDetFinetuningModel(train_config)
  model.train()
       

if __name__ == '__main__':
  app.run(main)
