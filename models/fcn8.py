
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dropout, Conv2D, Cropping2D, Conv2DTranspose, Add, Input, Reshape, Permute, Activation
from keras.callbacks import ModelCheckpoint

from models.utils import crop


def model_fcn8(nClasses, image_shape=(480, 480, 3)):
  base_model = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)

  for layer in base_model.layers:
    layer.trainable = False

  block3_pool = base_model.get_layer('block3_pool').output
  block4_pool = base_model.get_layer('block4_pool').output
  block5_pool = base_model.get_layer('block5_pool').output

  X = Conv2D(4096, (7, 7),  padding='same', activation='relu', name='block6_conv1')(block5_pool)
  X = Dropout(0.5, name='block6_dropout1')(X)
  X = Conv2D(4096, (1, 1), activation='relu', padding='same', name='block6_conv2')(X)
  X = Dropout(0.5, name='block6_dropout2')(X)
  X = Conv2D(nClasses, (1, 1), padding='same', kernel_initializer='he_normal', name='block6_conv3')(X)
  X = Conv2DTranspose(nClasses, kernel_size=(4,4), strides=(2,2) , padding='same', use_bias=False, name='block6_deconv1')(X)

  X2 = Conv2D(nClasses, (1, 1), kernel_initializer='he_normal', padding='same', name='block7_conv1')(block4_pool)
  X, X2 = crop(X, X2, base_model.input)
  X = Add(name='block7_Add')([X, X2])
  X = Conv2DTranspose(nClasses, kernel_size=(4,4), strides=(2, 2), padding='same')(X)

  X3 = Conv2D(nClasses, (1, 1), padding='same', name='block8_conv1')(block3_pool)
  X, X3 = crop(X, X3, base_model.input)
  X = Add(name='block8_Add')([X, X3])
  X = Conv2DTranspose(nClasses, kernel_size=(16, 16), strides=(8, 8), padding='same', name='block8_Deconv')(X)

  # X = (Reshape((-1, nClasses)))(X)
  X = (Activation('softmax'))(X)

  model = Model(inputs = [base_model.input], outputs=[X])

  return model


if __name__ == '__main__':
  model = model_fcn8(nClasses = 3, image_shape=(576, 800, 3))
  model.summary()