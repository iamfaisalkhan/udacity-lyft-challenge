
import os
from glob import glob
import pandas as pd
from sklearn.utils import shuffle

def _load(data_dir):
  img_path = os.path.join(data_dir, 'CameraRGB')
  lbl_path = os.path.join(data_dir, 'CameraSeg')

  df = pd.DataFrame(dict(image = glob(os.path.join(img_path, '*.*p*g'))))
  df['id'] = df['image'].map(lambda x: x.split('/')[-1].split('.')[0])
  df['label'] = df['image'].map(lambda x: os.path.join(lbl_path, x.split('/')[-1]))

  return df

def load_data(data_dir = './data'):
  train = os.path.join(data_dir, 'Train')
  valid = os.path.join(data_dir, 'Valid')
  test = os.path.join(data_dir, 'Test')

  train_df = _load(train)
  valid_df = _load(valid)
  test_df = _load(test)

  return train_df, valid_df, test_df