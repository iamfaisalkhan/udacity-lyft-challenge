
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint

from keras.layers import *


def model_unetVGG16(nClasses=3, image_shape=(320, 416, 3), keep_prob=0.5):
  base_model = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
  
  for layer in base_model.layers:
    layer.trainable = False

  block1_pool = base_model.get_layer('block1_pool').output
  block2_pool = base_model.get_layer('block2_pool').output
  block3_pool = base_model.get_layer('block3_pool').output
  block4_pool = base_model.get_layer('block4_pool').output
  block5_pool = base_model.get_layer('block5_pool').output

  X = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same') (block5_pool)
  bn = BatchNormalization()(block4_pool)
  X = concatenate([X, bn])
  X = Conv2D(512, (3, 3), activation='relu', padding='same') (X)
  X = Conv2D(256, (3, 3), activation='relu', padding='same') (X)

  X = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same') (X)
  bn = BatchNormalization()(block3_pool)
  X = concatenate([X, bn])
  X = Conv2D(256, (3, 3), activation='relu', padding='same') (X)
  X = Conv2D(128, (3, 3), activation='relu', padding='same') (X)

  X = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same') (X)
  bn = BatchNormalization()(block2_pool)
  X = concatenate([X, bn])
  X = Conv2D(128, (3, 3), activation='relu', padding='same') (X)
  X = Conv2D(64, (3, 3), activation='relu', padding='same') (X)

  X = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (X)
  bn = BatchNormalization()(block1_pool)
  X = concatenate([X, bn])
  X = Conv2D(64, (3, 3), activation='relu', padding='same') (X)
  # X = Conv2D(32, (3, 3), activation='relu', padding='same')(X)
  #

  X = Conv2DTranspose(nClasses, (3, 3), strides=(2, 2), padding='same') (X)
  X =  Conv2D( nClasses , (1, 1) , padding='same')(X)

  X = (Activation('softmax', name='y_'))(X)
  
  model = Model(inputs=[base_model.input], outputs=[X])

  return model

if __name__ == '__main__':
  model = model_unetVGG16()
  model.summary()