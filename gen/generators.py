# Keras generators for loading image from data-frame. Thanks to the ideas present in
# https://www.kaggle.com/kmader/data-preprocessing-and-unet-segmentation-gpu
#
import os
import cv2
import pandas as pd
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator

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

  new_lbl = np.zeros_like(lbl)
  new_lbl[:, :, 0] = (lbl[:, :, 0] == 7).astype(np.uint8)
  new_lbl[:, :, 1] = (lbl[:, :, 0] == 10).astype(np.uint8)
  new_lbl[:, :, 2] = np.invert(np.logical_or(new_lbl[:, :, 0], new_lbl[:, :, 1])).astype(np.uint8)
    
  return new_lbl

img_gen_args = dict(samplewise_center=False, 
                    samplewise_std_normalization=False, 
                    horizontal_flip = True, 
                    vertical_flip = False,
                    )

                    # height_shift_range = 0.1)
                    # width_shift_range = 0.1,
                    # rotation_range = 3,
                    # shear_range = 0.01,
                    # fill_mode = 'nearest',
                    # zoom_range = 0.05)

rgb_gen = ImageDataGenerator(preprocessing_function = preprocess_input, **img_gen_args)
lab_gen = ImageDataGenerator(preprocessing_function = preprocess_label, **img_gen_args)
# lab_gen = ImageDataGenerator(preprocessing_function = preprocess_label)


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

def  train_and_lab_gen_func (in_df, image_size = (480, 480), target_size = (480, 480), batch_size = 8, seed = None):
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
      x_new[i] = cv2.resize(x[i], (target_size[1], target_size[0]))
      y_new[i, :, :, :] = cv2.resize(y[i], (target_size[1], target_size[0]))
      i += 1
    yield x_new, y_new
        