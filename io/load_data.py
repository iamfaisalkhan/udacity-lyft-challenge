
import os
from glob import glob
import pandas as pd
from sklearn.utils import shuffle


def load_data(data_dir = './data'):
  TRAIN_IMAGE_DIR = os.path.join(data_dir, 'Train/CameraRGB')
  TRAIN_LABEL_DIR = os.path.join(data_dir, 'Train/CameraSeg')

  clara_seg_data = pd.DataFrame(dict(image = glob(os.path.join(TRAIN_IMAGE_DIR, '*.*p*g'))))
  clara_seg_data['id'] = clara_seg_data['image'].map(lambda x: x.split('/')[-1].split('.')[0])
  clara_seg_data['label'] = clara_seg_data['image'].map(lambda x: os.path.join(TRAIN_LABEL_DIR, x.split('/')[-1]))

  split = int(clara_seg_data.shape[0] *.20)
  
  train_df, dev_df = clara_seg_data[0:-split], clara_seg_data[-split:]

  train_df = shuffle(train_df)
  dev_df = shuffle(dev_df)

  return train_df, dev_df



