
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint

from keras.layers import *

def Conv2D_BN(X, filters, K=3):
    X = Conv2D(filters, (K, K), dilation_rate=1, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    return X

def merge(X1, X2):
  X1 = UpSampling2D(size=(2, 2))(X1)
  return concatenate([X1, X2])


def model_unetVGG16_v1(nClasses=3, image_shape=(320, 416, 3), keep_prob=0.5):
  base_model = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)

  for layer in base_model.layers:
    layer.trainable = False

  block1_pool = base_model.get_layer('block1_pool').output
  block2_pool = base_model.get_layer('block2_pool').output
  block3_pool = base_model.get_layer('block3_pool').output
  block4_pool = base_model.get_layer('block4_pool').output
  block5_pool = base_model.get_layer('block5_pool').output

  bn = BatchNormalization()(block5_pool)
  X = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same') (bn)
  X = concatenate([X, block4_pool])
  X = Conv2D(512, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same') (X)

  X = Conv2D(256, (3, 3), activation='relu', padding='same') (X)

  X = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same') (X)
  bn = BatchNormalization()(block3_pool)
  X = concatenate([X, bn])
  X = Conv2D(256, (3, 3), activation='relu', padding='same') (X)
  X = Conv2D(128, (3, 3), activation='relu', padding='same') (X)

  X = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same') (X)
  bn = BatchNormalization()(block2_pool)
  X = concatenate([X, bn])
  X = Conv2D(128, (3, 3), activation='relu',padding='same') (X)
  X = Conv2D(64, (3, 3), activation='relu', padding='same') (X)

  X = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (X)
  bn = BatchNormalization()(block1_pool)
  X = concatenate([X, bn])
  X = Conv2D(64, (3, 3), activation='relu',  padding='same') (X)
  X = Conv2D(32, (3, 3), activation='relu', padding='same')(X)
  #

  X = Conv2DTranspose(nClasses, (3, 3), strides=(2, 2), padding='same') (X)
  X = Conv2D(nClasses, (1, 1), padding='same')(X)

  X = (Activation('softmax', name='y_'))(X)
  
  model = Model(inputs=[base_model.input], outputs=[X])

  return model


def model_unetVGG16_v2(nClasses=3, image_shape=(320, 416, 3)):
  base_model = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)

  for layer in base_model.layers:
    layer.trainable = False

  block1_conv2 = base_model.get_layer('block1_conv2').output
  block2_conv2 = base_model.get_layer('block2_conv2').output
  block3_conv3 = base_model.get_layer('block3_conv3').output
  block4_conv3 = base_model.get_layer('block4_conv3').output
  block4_pool = base_model.get_layer('block4_pool').output

  X = Conv2D(1024, (3, 3), activation='relu', padding='same') (block4_pool)
  X = Conv2D(1024, (3, 3), activation='relu', padding='same') (X)
  X + BatchNormalization()(X)

  X = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (X)
  X = concatenate([X, block4_conv3])
  X = Conv2D(512, (3, 3), activation='relu', padding='same') (X)
  X = Conv2D(512, (3, 3), activation='relu', padding='same') (X)
  X = BatchNormalization()(X)
  
  X = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (X)
  X = concatenate([X, block3_conv3])
  X = Conv2D(256, (3, 3), activation='relu', padding='same') (X)
  X = Conv2D(256, (3, 3), activation='relu', padding='same') (X)
  X = BatchNormalization()(X)

  X = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (X)
  X = concatenate([X, block2_conv2])
  X = Conv2D(128, (3, 3), activation='relu',padding='same') (X)
  X = Conv2D(128, (3, 3), activation='relu', padding='same') (X)
  X = BatchNormalization()(X)

  X = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (X)
  X = concatenate([X, block1_conv2])
  X = Conv2D(64, (3, 3), activation='relu', padding='same')(X)
  X = Conv2D(64, (3, 3), activation='relu', padding='same')(X)
  X = BatchNormalization()(X)

  # X = Conv2DTranspose(nClasses, (3, 3), strides=(2, 2), padding='same') (X)
  X = Conv2D(nClasses, (1, 1), padding='same')(X)

  X = (Activation('softmax', name='y_'))(X)
  
  model = Model(inputs=[base_model.input], outputs=[X])

  return model

def model_unet_v2(nClasses=3, image_shape=(320, 416, 3), keep_prob=0.5):
  base_input = Input(image_shape)

  X = Conv2D_BN(base_input, 64)
  X = Conv2D_BN(X, 64)
  X0 = X
  X = MaxPooling2D(pool_size=2)(X)

  X = Conv2D_BN(X, 128)
  X = Conv2D_BN(X, 128)
  X1 = X
  X = MaxPooling2D(pool_size=2)(X)

  X = Conv2D_BN(X, 256)
  X = Conv2D_BN(X, 256)
  X2  = X
  X = MaxPooling2D(pool_size=2)(X)

  X = Conv2D_BN(X, 512)
  X = Conv2D_BN(X, 512)
  X = Conv2D_BN(X, 512)

  X = merge(X, X2)
  X = Conv2D_BN(X, 256)
  X = Conv2D_BN(X, 256)

  X = merge(X, X1)
  X = Conv2D_BN(X, 128)
  X = Conv2D_BN(X, 128)

  X = merge(X, X0)
  X = Conv2D_BN(X, 64)
  X = Conv2D_BN(X, 64)

  X = Conv2D(nClasses, (1, 1), activation='softmax')(X)

  model = Model(inputs = [base_input], outputs = [X])

  return model

if __name__ == '__main__':
  model = model_unet_v2(3, (416, 544, 3))
  model.summary()