# Keras generators for loading image from data-frame. Thanks to the ideas present in
# https://www.kaggle.com/kmader/data-preprocessing-and-unet-segmentation-gpu
#
import os
import pandas as pd
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (480, 480) # slightly smaller than vgg16 normally expects

def prepprocess_label(lbl):
    new_lbl = np.zeros((*IMG_SIZE, 3))
    new_lbl[:, :, 0] = lbl[:, :, 0] == 7
    new_lbl[:400, :, 1] = lbl[:400, :, 0] == 10
    new_lbl[:, :, 2] = np.invert(np.logical_or(new_lbl[:, :, 0], new_lbl[:, :, 1]))
    
    return new_lbl

img_gen_args = dict(samplewise_center=False, 
                    samplewise_std_normalization=False, 
                    horizontal_flip = True, 
                    vertical_flip = False, 
                    height_shift_range = 0.1, 
                    width_shift_range = 0.1, 
                    rotation_range = 3, 
                    shear_range = 0.01,
                    fill_mode = 'nearest',
                    zoom_range = 0.05)

rgb_gen = ImageDataGenerator(preprocessing_function = preprocess_input, **img_gen_args)
lab_gen = ImageDataGenerator(preprocessing_function=prepprocess_label, **img_gen_args)

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

def  train_and_lab_gen_func (in_df, image_size = IMG_SIZE, batch_size = 8, seed = None):
  if seed is None:
      seed = np.random.choice(range(1000))
  
  train_rgb_gen = flow_from_dataframe(rgb_gen, 
                                      in_df, 
                                      path_col = 'image',
                                      y_col = 'id', 
                                      color_mode = 'rgb',
                                      target_size = IMG_SIZE,
                                      batch_size = batch_size,
                                      seed = seed)
  
  train_lab_gen = flow_from_dataframe(lab_gen, 
                                      in_df, 
                                      path_col = 'label',
                                      y_col = 'id', 
                                      target_size = IMG_SIZE,
                                      color_mode = 'rgb',
                                      batch_size = batch_size,
                                      seed = seed)
  
  for (x, _), (y, _) in zip(train_rgb_gen, train_lab_gen):
      yield x, y
        