
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dropout, Conv2D, concatenate, BatchNormalization, MaxPooling2D, Conv2DTranspose, Input
from keras.callbacks import ModelCheckpoint


def model_unetVGG16(nClasses, image_shape=(480, 480, 3), keep_prob=0.5):
  
  img_input = Input(shape=image_shape)
  X = BatchNormalization()(img_input)

  base_model = VGG16(include_top=False, input_tensor=X, weights='imagenet', input_shape=image_shape)

  for layer in base_model.layers:
    layer.trainable = False



  block1_pool = base_model.get_layer('block1_pool').output
  block2_pool = base_model.get_layer('block2_pool').output
  block3_pool = base_model.get_layer('block3_pool').output
  block4_pool = base_model.get_layer('block4_pool').output
  block5_pool = base_model.get_layer('block5_pool').output


  # u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
  # u6 = concatenate([u6, c4])
  # c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
  # c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

  # u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
  # u7 = concatenate([u7, c3])
  # c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
  # c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

  # u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
  # u8 = concatenate([u8, c2])
  # c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
  # c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

  # u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
  # u9 = concatenate([u9, c1], axis=3)
  # c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
  # c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

  # outputs = Conv2D(nClasses, (1, 1), activation='softmax') (c9)

  model = Model(inputs=[base_model.input], outputs=[base_model.output])

  return model

if __name__ == '__main__':
  model = model_unetVGG16(3)
  model.summary()