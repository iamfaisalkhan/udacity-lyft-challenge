
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dropout, Conv2D, Cropping2D, ZeroPadding2D, Input, Activation, BatchNormalization, UpSampling2D
from keras.callbacks import ModelCheckpoint

from models.utils import crop


def model_segnetVGG16(nClasses, image_shape=(480, 480, 3), keep_prob=0.5):
  base_model = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)

  for layer in base_model.layers:
    layer.trainable = False

  block5_pool = base_model.get_layer('block5_pool').output

  X = ZeroPadding2D( (1,1))(block5_pool)
  X = Conv2D(512, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)

  X = UpSampling2D( (2,2))(X)
  X = ZeroPadding2D( (1,1))(X)
  X = Conv2D( 256, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)

  X = UpSampling2D((2,2))(X)
  X = ZeroPadding2D((1,1))(X)
  X = Conv2D( 128 , (3, 3), padding='same')(X)
  X = BatchNormalization()(X)

  X = UpSampling2D((2,2))(X)
  X = ZeroPadding2D((1,1))(X)
  X = Conv2D( 64 , (3, 3), padding='same')(X)
  X = BatchNormalization()(X)

  X = Conv2D( nClasses , (3, 3) , padding='same')(X)
  X = Activation('softmax', name='y_')(X)

  model = Model(inputs=[base_model.input] , outputs=[X])
  
  return model

if __name__ == '__main__':
  model = model_segnetVGG16(nClasses = 3)
  model.summary()