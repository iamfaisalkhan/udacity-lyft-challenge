import os
import cv2
import pandas as pd
import numpy as np

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input
from skimage.io import imread
from random import shuffle

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

  df = df.sample(frac=1)

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

      X = np.array([cv2.resize(imread(f)[200:, :, :], (target_size[1], target_size[0])) for f in sub.image], np.uint8)
      Y = np.array([cv2.resize(preprocess_label(imread(f))[200:, :, :], (target_size[1], target_size[0])) for f in sub.label], np.uint8)

      yield X, Y

      i = j
      j += batch_size
      count += 1

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    random_bright = .25+np.random.uniform()
    
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)

    return image1

def random_mask(X, Y, shape=(10, 10)):
  c, r, _= X.shape

  for i in range(3):
    x0 = np.random.randint(c-400, c-100)
    y0  = np.random.randint(0, r-shape[1])

    X[x0:x0+shape[0], y0:y0+shape[1], :] = 0
    Y[x0:x0+shape[0], y0:y0+shape[1], :] = 0

  return X, Y

def random_transform(X, Y, tran=10, rot=15, shear=5):
    rows,cols,chs = X.shape
    
    if np.random.uniform() > 0.5:
      cv2.flip(X, 0)
      cv2.flip(Y, 0)

    tx = tran * np.random.uniform() - tran/2
    ty = tran * np.random.uniform() - tran/2
    
    transMat = np.float32([[1, 0, tx], [0, 1, ty]])
    X = cv2.warpAffine(X, transMat, (cols,rows))
    Y = cv2.warpAffine(Y, transMat, (cols,rows))
    
    ang_rot = np.random.uniform(rot)-rot/2
    rot_M = cv2.getRotationMatrix2D((cols/2,rows/2), ang_rot, 1)
    X = cv2.warpAffine(X, rot_M,( cols,rows) )
    Y = cv2.warpAffine(Y, rot_M,( cols,rows) )

    pt1 = 5+shear*np.random.uniform()-shear/2
    pt2 = 20+shear*np.random.uniform()-shear/2
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    shear_M = cv2.getAffineTransform(pts1,pts2)
    X = cv2.warpAffine(X, shear_M, (cols,rows))
    Y = cv2.warpAffine(Y, shear_M, (cols,rows))

    X = augment_brightness_camera_images(X)

    # Random gauss noise
    # gauss = np.random.normal(0.0, 0.1**0.5, (rows, cols, chs))
    # gauss = gauss.reshape(rows, cols, chs)
    # X = X + gauss

    return X.astype(np.uint8), Y.astype(np.uint8)

def _cluster_train_set(df):
  weights = []
  ind = []
  m  = df.shape[0]
  for i, row in df.iterrows():
    lbl = preprocess_label(imread(row['label']))
    weights.append( 1 + lbl[:, :, 0].nonzero()[0].shape[0])
    ind.append(i)
    
  weights = np.array(weights)
  weights = [float(i)/sum(weights) for i in weights]
  c = sorted(zip(weights, ind))
    
  minor_cls_ind = [i[1] for i in reversed(c[-4000:]) ]
  major_cls_ind = [i[1] for i in c[0:1000]]
    
  return minor_cls_ind, major_cls_ind

def oversample_generator_from_df(df, batch_size, target_size, samples=4000, frac=0.8):

  count =1
  epoch = 0

  minor_clas_ind, major_cls_ind = _cluster_train_set(df)
  nbatches = samples // batch_size

  shuffle(minor_clas_ind)
  shuffle(major_cls_ind)

  minor_per_step = int(batch_size * frac)
  major_per_step = batch_size - minor_per_step

  X = np.zeros((batch_size, target_size[0], target_size[1], 3), np.uint8)
  Y = np.zeros((batch_size, target_size[0], target_size[1], 3), np.uint8)

  while 1:
    epoch += 1
    minor_ind = 0
    major_ind = 0

    shuffle(major_cls_ind)

    for _ in range(nbatches):
      cluster_minor = zip(df.loc[minor_clas_ind[minor_ind:minor_ind+minor_per_step]].image, 
                          df.loc[minor_clas_ind[minor_ind:minor_ind+minor_per_step]].label)

      cluster_major = zip(df.loc[major_cls_ind[major_ind:major_ind+major_per_step]].image,
                          df.loc[major_cls_ind[major_ind:major_ind+major_per_step]].label)

      i = 0
      for image, label in cluster_minor:
        x, y = random_transform(imread(image), preprocess_label(imread(label)))
        X[i] = cv2.resize(x[200:, :, :], (target_size[1], target_size[0]))
        y[:, :, 2] = np.invert(np.logical_or(y[:, :, 0], y[:, :, 1])).astype(np.uint8)
        Y[i] = cv2.resize(y[200:, :, :], (target_size[1], target_size[0]))
        i += 1

      for image, label in cluster_major:
        x, y = random_mask(imread(image), preprocess_label(imread(label)))
        X[i] = cv2.resize(x[200:, :, :], (target_size[1], target_size[0]))
        Y[i] = cv2.resize(y[200:, :, :], (target_size[1], target_size[0]))
        i += 1

      yield X, Y

      minor_ind += minor_per_step
      major_ind += major_per_step
      count += 1
#
