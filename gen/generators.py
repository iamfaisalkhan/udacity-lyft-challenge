# Keras generators for loading image from data-frame. Thanks to the ideas present in
# https://www.kaggle.com/kmader/data-preprocessing-and-unet-segmentation-gpu
#
import os
import cv2
import pandas as pd
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from gen.datagen import random_transform

def preprocess_multi_label(lbl):
  # Identify lane marking pixels (label is 6)
  lane_marking_pixels = lbl[:,:,0] == 6

  # Set lane marking pixels to road (label is 7)
  lbl[lane_marking_pixels.nonzero()] = 7

  # Identify all vehicle pixels
  vehicle_pixels = (lbl[:,:,0] == 10).nonzero()
  # Isolate vehicle pixels associated with the hood (y-position > 496)
  hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
  hood_pixels = (vehicle_pixels[0][hood_indices], \
                   vehicle_pixels[1][hood_indices])
  # Set hood pixel labels to 0
  lbl[hood_pixels] = 0
  # Return the preprocessed label image

  new_lbl = np.zeros((lbl.shape[0], lbl.shape[1], 12))
  new_lbl[:, :, 0] = (lbl[:, :, 0] == 10).astype(np.uint8)
  new_lbl[:, :, 1] = (lbl[:, :, 0] == 7).astype(np.uint8)
  new_lbl[:, :, 2] = (lbl[:, :, 0] == 5).astype(np.uint8)
  new_lbl[:, :, 3] = (lbl[:, :, 0] == 4).astype(np.uint8)
  new_lbl[:, :, 4] = (lbl[:, :, 0] == 3).astype(np.uint8)
  new_lbl[:, :, 5] = (lbl[:, :, 0] == 2).astype(np.uint8)
  new_lbl[:, :, 6] = (lbl[:, :, 0] == 1).astype(np.uint8)
  new_lbl[:, :, 7] = (lbl[:, :, 0] == 8).astype(np.uint8)
  new_lbl[:, :, 8] = (lbl[:, :, 0] == 9).astype(np.uint8)
  new_lbl[:, :, 9] = (lbl[:, :, 0] == 11).astype(np.uint8)
  new_lbl[:, :, 10] = (lbl[:, :, 0] == 12).astype(np.uint8)
  new_lbl[:, :, 11] = (lbl[:, :, 0] == 0).astype(np.uint8)

  return new_lbl

def preprocess_label(lbl):
  # Identify lane marking pixels (label is 6)
  lane_marking_pixels = (lbl[:,:,0] == 6).nonzero()

  # Set lane marking pixels to road (label is 7)
  lbl[lane_marking_pixels] = 7

  # Identify all vehicle pixels
  vehicle_pixels = (lbl[:,:,0] == 10).nonzero()
  # Isolate vehicle pixels associated with the hood (y-position > 496)
  hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
  hood_pixels = (vehicle_pixels[0][hood_indices], \
                   vehicle_pixels[1][hood_indices])
  # Set hood pixel labels to 0
  lbl[hood_pixels] = 0
  # Return the preprocessed label image 

  new_lbl = np.zeros((lbl.shape[0], lbl.shape[1], 3))
  new_lbl[:, :, 0] = (lbl[:, :, 0] == 10).astype(np.uint8)
  new_lbl[:, :, 1] = (lbl[:, :, 0] == 7).astype(np.uint8)
  new_lbl[:, :, 2] = np.invert(np.logical_or(new_lbl[:, :, 0], new_lbl[:, :, 1])).astype(np.uint8)
    
  return new_lbl

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, seed = None, **dflow_args):
  base_dir = os.path.dirname(in_df[path_col].values[0])    
  print('## Ignore next message from keras, values are replaced anyways')
  df_gen = img_data_gen.flow_from_directory(base_dir, class_mode = 'sparse',seed = seed,**dflow_args)
  df_gen.filenames = in_df[path_col].values
  df_gen.classes = np.stack(in_df[y_col].values)
  df_gen.samples = in_df.shape[0]
  df_gen.n = in_df.shape[0]
  df_gen._set_index_array()
  df_gen.directory = '' # since we have the full path
  print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
  return df_gen

def  gen_func(in_df, 
              rgb_gen,
              lab_gen, 
              image_size = (480, 480), 
              target_size = (480, 480), 
              batch_size = 8, 
              seed = None, 
              vgg_preprocess = True,
              perturb = False):

  if seed is None:
      seed = np.random.choice(range(1000))
  
  train_rgb_gen = flow_from_dataframe(rgb_gen, 
                                      in_df, 
                                      path_col = 'image',
                                      y_col = 'id', 
                                      color_mode = 'rgb',
                                      target_size = image_size,
                                      batch_size = batch_size,
                                      seed = seed)
  
  train_lab_gen = flow_from_dataframe(lab_gen, 
                                      in_df, 
                                      path_col = 'label',
                                      y_col = 'id', 
                                      target_size = image_size,
                                      color_mode = 'rgb',
                                      batch_size = batch_size,
                                      seed = seed)

  for (x, _), (y, _) in zip(train_rgb_gen, train_lab_gen):
    m = x.shape[0]
    x_new = np.zeros((m, target_size[0], target_size[1], x.shape[-1]))
    y_new = np.zeros((m, target_size[0], target_size[1], x.shape[-1])).astype(np.uint8)
    i = 0
    for i in range(m):
      if (perturb):
        xx, yy = random_transform(x[i][184:, :, :], preprocess_label(y[i])[184:, :, :])
        yy[:, :, 2] = np.invert(np.logical_or(yy[:, :, 0], yy[:, :, 1]))
      else:
        xx, yy = x[i][184:, :, :], preprocess_label(y[i])[184:, :, :]

      x_new[i] = cv2.resize(preprocess_input(xx.astype(np.float32)), (544, 416))
      y_new[i] = cv2.resize(yy, (544, 416))
      i += 1
    yield x_new, y_new


def  gen_func_patch(in_df, 
                    rgb_gen, 
                    lab_gen, 
                    image_size = (480, 480),
                    target_size = (480, 480), 
                    batch_size = 8, 
                    seed = None):
  if seed is None:
      seed = np.random.choice(range(1000))
  
  train_rgb_gen = flow_from_dataframe(rgb_gen, 
                                      in_df, 
                                      path_col = 'image',
                                      y_col = 'id', 
                                      color_mode = 'rgb',
                                      target_size = image_size,
                                      batch_size = batch_size,
                                      seed = seed)
  
  train_lab_gen = flow_from_dataframe(lab_gen, 
                                      in_df, 
                                      path_col = 'label',
                                      y_col = 'id', 
                                      target_size = image_size,
                                      color_mode = 'rgb',
                                      batch_size = batch_size,
                                      seed = seed)

  for (x, _), (y, _) in zip(train_rgb_gen, train_lab_gen):
    m = x.shape[0]
    x_new = np.zeros((m, target_size[0], target_size[1], x.shape[-1]))
    y_new = np.zeros((m, target_size[0], target_size[1], x.shape[-1]))
    i = 0
    for i in range(m):
      xd, yd = y[i][:, :, 0].nonzero()
      if xd.shape[0] == 0:
        x0 = np.random.randint(0, image_size[0] - target_size[0])
        y0 = np.random.randint(0, image_size[1] - target_size[1])
        x1 = x0 + target_size[0]
        y1 = y1 + target_size[1]
      else:
        x0, x1 = xd.min(), xd.max()
        y0, y1 = yd.min(), yd.max()

      wr = target_size[1] - (y1-y0)
      hr = target_size[0] - (x1-x0)

      if hr > 0:
        b = hr - (496 - x1)
        x0, x1 = x0 - (hr - b), x1 + b
      if wr > 0:
        r = np.random.randint(0, min(wr, target_size[1]))
        y0, y1 = max(0, y0 - (wr - r)), y1 + r
      
      x_new[i] = cv2.resize(x[i][x0:x1, y0:y1, :], (target_size[1], target_size[0]))
      y_new[i] = cv2.resize(y[i][x0:x1, y0:y1, :], (target_size[1], target_size[0]))
      i += 1
    yield x_new, y_new
