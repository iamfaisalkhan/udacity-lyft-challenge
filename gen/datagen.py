import os
import cv2
import pandas as pd
import numpy as np

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input
from skimage.io import imread

def preprocess_multi_label(lbl):
  # Identify lane marking pixels (label is 6)
  lane_marking_pixels = (lbl[:, :, 0] == 6).nonzero()

  # Set lane marking pixels to road (label is 7)
  lbl[lane_marking_pixels] = 7

  # Identify all vehicle pixels
  vehicle_pixels = (lbl[:, :, 0] == 10).nonzero()
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

  # new_lbl[:, :, 2] = np.invert(np.logical_or(new_lbl[:, :, 0], new_lbl[:, :, 1])).astype(np.uint8)
  return new_lbl


def preprocess_label(lbl):
  # Identify lane marking pixels (label is 6)
  lane_marking_pixels = (lbl[:, :, 0] == 6).nonzero()

  # Set lane marking pixels to road (label is 7)
  lbl[lane_marking_pixels] = 7

  # Identify all vehicle pixels
  vehicle_pixels = (lbl[:, :, 0] == 10).nonzero()
  # Isolate vehicle pixels associated with the hood (y-position > 496)
  hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
  hood_pixels = (vehicle_pixels[0][hood_indices], \
                 vehicle_pixels[1][hood_indices])
  # Set hood pixel labels to 0
  lbl[hood_pixels] = 0
  # Return the preprocessed label image

  new_lbl = np.zeros_like(lbl)
  new_lbl[:, :, 0] = (lbl[:, :, 0] == 10).astype(np.uint8)
  new_lbl[:, :, 1] = (lbl[:, :, 0] == 7).astype(np.uint8)
  new_lbl[:, :, 2] = np.invert(np.logical_or(new_lbl[:, :, 0], new_lbl[:, :, 1])).astype(np.uint8)

  return new_lbl

def balanced_generator_from_df(df, batch_size, target_size):
  nbatches, n_skipped_per_epoch = divmod(df.shape[0], batch_size)

  # df = df.sample(frac=1)

  count =1
  epoch = 0

  while 1:
    epoch += 1
    i, j = 0, batch_size
    # Mini-batches within epoch.
    mini_batches_completed = 0
    for _ in range(nbatches):
      seed = np.random.choice(range(1000))

      sub = df.iloc[i:j]

      X = np.array([cv2.resize(preprocess_input(img_to_array(imread(f))), (target_size[1], target_size[0])) for f in sub.image])
      Y = np.array([cv2.resize(preprocess_label(img_to_array(imread(f))), (target_size[1], target_size[0])) for f in sub.label])

      yield X, Y

      i = j
      j += batch_size
      count += 1

def oversample_generator_from_df(df, batch_size, target_size, weights):
  nbatches, n_skipped_per_epoch = divmod(df.shape[0], batch_size)

  count =1
  epoch = 0


  while 1:
    epoch += 1
    i, j = 0, batch_size
    # Mini-batches within epoch.
    mini_batches_completed = 0
    for _ in range(nbatches):
      # seed = np.random.choice(range(1000))
      sub = df.sample(n=batch_size, replace=True, weights=weights)

      X = np.array([cv2.resize(preprocess_input(img_to_array(imread(f))), (target_size[1], target_size[0])) for f in sub.image])
      Y = np.array([cv2.resize(preprocess_label(img_to_array(imread(f))), (target_size[1], target_size[0])) for f in sub.label])

      yield X, Y
      i = j
      j += batch_size
      count += 1
#
