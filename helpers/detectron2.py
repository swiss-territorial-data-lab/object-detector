#!/usr/bin/env python
# coding: utf-8

import os
import time
import torch
import numpy as np
import logging

import datetime

from detectron2.engine.hooks import HookBase
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator
from detectron2.utils import comm
from detectron2.utils.logger import log_every_n_seconds

# cf. https://medium.com/@apofeniaco/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
# cf. https://towardsdatascience.com/face-detection-on-custom-dataset-with-detectron2-and-pytorch-using-python-23c17e99e162
# cf. http://cocodataset.org/#detection-eval


class LossEvalHook(HookBase):
  '''
  Evaluate the loss on the validation set during training
  1- Doing inference of dataset like an Evaluator does
  2- Get the loss metric like the trainer does
  '''

  def __init__(self, eval_period, model, data_loader):
      self._model = model
      self._period = eval_period
      self._data_loader = data_loader
  
  def _do_loss_eval(self):

    #print('Entering here...')

    # Copying inference_on_dataset from evaluator.py
    total = len(self._data_loader)
    num_warmup = min(5, total - 1)
        
    start_time = time.perf_counter()
    total_compute_time = 0
    losses = []
    for idx, inputs in enumerate(self._data_loader):            
        if idx == num_warmup:
            start_time = time.perf_counter()
            total_compute_time = 0
        start_compute_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_compute_time += time.perf_counter() - start_compute_time
        iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
        seconds_per_img = total_compute_time / iters_after_start
        if idx >= num_warmup * 2 or seconds_per_img > 5:
            total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
            eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
            log_every_n_seconds(
                logging.INFO,
                "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                    idx + 1, total, seconds_per_img, str(eta)
                ),
                n=5,
            )
        loss_batch = self._get_loss(inputs)
        losses.append(loss_batch)
    mean_loss = np.mean(losses)
    self.trainer.storage.put_scalar('validation_loss', mean_loss)
    comm.synchronize()

    return losses
          
  def _get_loss(self, data):

      #print('Entering there...')

      # How loss is calculated on train_loop 
      metrics_dict = self._model(data)
      metrics_dict = {
          k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
          for k, v in metrics_dict.items()
      }
      total_losses_reduced = sum(loss for loss in metrics_dict.values())
      return total_losses_reduced
      
      
  def after_step(self):

      #print('Entering overthere...')

      next_iter = self.trainer.iter + 1
      is_final = next_iter == self.trainer.max_iter
      if is_final or (self._period > 0 and next_iter % self._period == 0):
          self._do_loss_eval()
      self.trainer.storage.put_scalars(timetest=12)



class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    '''
    Adding an evaluator for the test set, because it is not included by default
    '''
      
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        
    os.makedirs("COCO_eval", exist_ok=True)
    
    return COCOEvaluator(dataset_name, cfg, False, output_folder)

  @classmethod
  def build_train_loader(cls, cfg):
    '''
    Build a custom dataloader to handel images with more than 3 channels
    cf. https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html
    '''

    # print('Working on it...')

    nbr_channels=4 # TODO: Define the nbr of channels in cfg or get it from images
    if False:  # nbr_channels>3:
      # TODO: modify the code from: https://detectron2.readthedocs.io/en/latest/_modules/detectron2/data/detection_utils.html#read_image
      mapper=True # Get the custom mapper
    else:
      mapper=DatasetMapper(cfg, is_train=False) # Default choice for mapper

    return build_detection_train_loader(cfg, mapper=mapper)

  
  def build_hooks(self):
    '''
    A Hook is a function called on each step.
    1- Add a custom Hook to the Trainer that gets called after EVAL_PERIOD steps
    2- When the Hook is called, do inference on the whole Evaluation dataset
    3- Every time inference is done, get the loss on the same way it is done when training, and store the mean value for all the dataset.
    '''
        
    hooks = super().build_hooks()
    
    hooks.insert(-1,
        LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0], DatasetMapper(self.cfg, True))
        )
    )

    print('test')
                
    return hooks

    

# HELPER FUNCTIONS

def _preprocess(preds):
  
  fields = preds['instances'].get_fields()

  out = {}

  # pred_boxes
  if 'pred_boxes' in fields.keys():
    out['pred_boxes'] = [box.cpu().numpy() for box in fields['pred_boxes']]
  # pred_classes
  if 'pred_classes' in fields.keys():
    out['pred_classes'] = fields['pred_classes'].cpu().numpy()
  # pred_masks
  if 'pred_masks' in fields.keys():
    out['pred_masks'] = fields['pred_masks'].cpu().numpy()
  # scores
  if 'scores' in fields.keys():
    out['scores'] = fields['scores'].cpu().numpy()

  return out


def dt2predictions_to_list(preds):

  instances = []
  
  tmp = _preprocess(preds)

  for idx in range(len(tmp['scores'])):
    instance = {}
    instance['score'] = tmp['scores'][idx]
    instance['pred_class'] = tmp['pred_classes'][idx]

    if 'pred_masks' in tmp.keys():
      instance['pred_mask'] = tmp['pred_masks'][idx]
    
    instance['pred_box'] = tmp['pred_boxes'][idx]
    
    instances.append(instance)

  return instances

