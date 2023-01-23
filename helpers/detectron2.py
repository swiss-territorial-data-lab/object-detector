#!/usr/bin/env python
# coding: utf-8

import os, sys
import time
import torch
import gdal
import numpy as np
import logging
import copy
from typing import List, Optional, Union

import datetime

from detectron2.engine.hooks import HookBase
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import (build_detection_test_loader, build_detection_train_loader,
                            DatasetMapper, MetadataCatalog)
from detectron2.data import detection_utils as utils, transforms as T
from detectron2.evaluation import COCOEvaluator
from detectron2.utils import comm
from detectron2.utils.logger import log_every_n_seconds
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

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

      next_iter = self.trainer.iter + 1
      is_final = next_iter == self.trainer.max_iter
      if is_final or (self._period > 0 and next_iter % self._period == 0):
          self._do_loss_eval()
      self.trainer.storage.put_scalars(timetest=12)


class SpectralDatasetMapper(DatasetMapper):
  '''
  This class define a custom DatasetMapper to handle images with more than 3 bands.
  The code is the default DatasetMapper, except for reading the file. This part was modified in order 
  to keep all the bands.
  '''

  def __call__(self, dataset_dict):
    """
    Args:
        dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
    Returns:
        dict: a format that builtin models in detectron2 accept
    """
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

    # Modified part to read the image:
    dataset = gdal.Open(dataset_dict["file_name"])
    image = dataset.ReadAsArray()
    image = np.transpose(image, (1, 2, 0))

    if image.shape[2]<=3:
      print("The image does not have more than 3 bands.")
      sys.exit(1)

    del dataset

    utils.check_image_size(dataset_dict, image)

    # USER: Remove if you don't do semantic/panoptic segmentation.
    if "sem_seg_file_name" in dataset_dict:
        sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
    else:
        sem_seg_gt = None

    aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
    transforms = self.augmentations(aug_input)
    image, sem_seg_gt = aug_input.image, aug_input.sem_seg

    image_shape = image.shape[:2]  # h, w
    # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
    # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
    # Therefore it's important to use torch.Tensor.
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    if sem_seg_gt is not None:
        dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

    # USER: Remove if you don't use pre-computed proposals.
    # Most users would not need this feature.
    if self.proposal_topk is not None:
        utils.transform_proposals(
            dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
        )

    if not self.is_train:
        # USER: Modify this if you want to keep them for some reason.
        dataset_dict.pop("annotations", None)
        dataset_dict.pop("sem_seg_file_name", None)
        return dataset_dict

    if "annotations" in dataset_dict:
        self._transform_annotations(dataset_dict, transforms, image_shape)

    return dataset_dict

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
    Inspired from: https://github.com/Helmholtz-AI-Energy/TBBRDet
    '''

    if cfg.NUM_CHANNELS>3:
      # Get the custom mapper
      mapper=SpectralDatasetMapper(
                                  is_train=True,
                                  augmentations=[],
                                  use_instance_mask=True,
                                  image_format=None,
                                  ) 
    else:
      # Use default mapper
      mapper=DatasetMapper(cfg, is_train=True)

    return build_detection_train_loader(cfg, mapper=mapper)

  @classmethod
  def build_test_loader(cls, cfg, dataset_name):
    '''
    Build a custom dataloader to handel images with more than 3 channels
    cf. https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html
    Inspired from: https://github.com/Helmholtz-AI-Energy/TBBRDet
    '''

    if cfg.NUM_CHANNELS>3:
      # Get the custom mapper
      mapper=SpectralDatasetMapper(
                                  is_train=False,
                                  augmentations=[],
                                  use_instance_mask=True,
                                  image_format=None
                                  ) 
    else:
      # Use default mapper
      mapper=DatasetMapper(cfg, is_train=False)

    return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

  
  def build_hooks(self):
    '''
    A Hook is a function called on each step.
    1- Add a custom Hook to the Trainer that gets called after EVAL_PERIOD steps
    2- When the Hook is called, do inference on the whole Evaluation dataset
    3- Every time inference is done, get the loss on the same way it is done when training, and store the mean value for all the dataset.
    '''
        
    hooks = super().build_hooks()
    
    if self.cfg.NUM_CHANNELS>3:
      mapper=SpectralDatasetMapper(
                                  is_train=True,
                                  augmentations=[],
                                  use_instance_mask=True,
                                  image_format=None,
                                  )
    else:
      mapper=DatasetMapper(self.cfg, True)

    hooks.insert(-1,
        LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0], mapper=mapper)
        )
    )

                
    return hooks

class CocoPredictor(DefaultPredictor):
    """
	  Copy of the orginial script of DefaultPredictor. Modified so that it does not accept only the BGR format for images.

    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    3. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"] or cfg.NUM_CHANNELS>3, self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions

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

