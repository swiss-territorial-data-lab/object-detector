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
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator
from detectron2.utils import comm
from detectron2.utils.logger import log_every_n_seconds

from rasterio import features
from shapely.affinity import affine_transform
from shapely.geometry import box
from rdp import rdp

# cf. https://medium.com/@apofeniaco/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
# cf. https://towardsdatascience.com/face-detection-on-custom-dataset-with-detectron2-and-pytorch-using-python-23c17e99e162
# cf. http://cocodataset.org/#detection-eval
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):

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

        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):

        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)



class CocoTrainer(DefaultTrainer):

  # https://github.com/facebookresearch/detectron2/blob/main/tools/train_net.py#L91
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
      
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        
    os.makedirs("COCO_eval", exist_ok=True)
    
    return COCOEvaluator(dataset_name, None, False, output_folder)

  
  def build_hooks(self):
        
    hooks = super().build_hooks()
    
    hooks.insert(-1,
        LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0], DatasetMapper(self.cfg, True))
        )
    )
                
    return hooks

    

# HELPER FUNCTIONS

def _preprocess(dets):
  
  fields = dets['instances'].get_fields()

  out = {}

  # pred_boxes
  if 'pred_boxes' in fields.keys():
    out['pred_boxes'] = [box.cpu().numpy() for box in fields['pred_boxes']]
  # det_classes
  if 'pred_classes' in fields.keys():
    out['pred_classes'] = fields['pred_classes'].cpu().numpy()
  # pred_masks
  if 'pred_masks' in fields.keys():
    out['pred_masks'] = fields['pred_masks'].cpu().numpy()
  # scores
  if 'scores' in fields.keys():
    out['scores'] = fields['scores'].cpu().numpy()

  return out


def detectron2dets_to_features(dets, im_path, transform, rdp_enabled, rdp_eps, year="None"):

  feats = []
  
  tmp = _preprocess(dets)

  for idx in range(len(tmp['scores'])):
    
    instance = {}
    instance['score'] = tmp['scores'][idx]
    instance['pred_class'] = tmp['pred_classes'][idx]

    if 'pred_masks' in tmp.keys():

      pred_mask_int = tmp['pred_masks'][idx].astype(np.uint8)
      _feats = [
        {
            'type': 'Feature', 
            'properties': {'score': instance['score'], 'det_class': instance['pred_class'], 'image': os.path.basename(im_path), 'year_det': year},
            'geometry': geom
        } for (geom, v) in features.shapes(pred_mask_int, mask=None, transform=transform) if v == 1.0
      ]

      for f in _feats:
        if rdp_enabled:
          coords = f['geometry']['coordinates']
          coords_after_rdp = [rdp(x, epsilon=rdp_eps) for x in coords]
          f['geometry']['coordinates'] = coords_after_rdp
          
        feats.append(f)

    else: # if pred_masks does not exist, pred_boxes should (it depends on Detectron2's MASK_ON config param)
      instance['pred_box'] = tmp['pred_boxes'][idx]

      geom = affine_transform(box(*instance['pred_box']), [transform.a, transform.b, transform.d, transform.e, transform.xoff, transform.yoff])
      _feats = [
          {
              'type': 'Feature', 
              'properties': {'score': instance['score'], 'det_class': instance['pred_class'], 'image': os.path.basename(im_path), 'year_det': year}, 
              'geometry': geom
          }
      ]

      feats += _feats

  return feats