
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dropout, Conv2D, Cropping2D, ZeroPadding2D, Input, Reshape, Activation, BatchNormalization, UpSampling2D
from keras.callbacks import ModelCheckpoint

from models.segnet_helpers import MaxPoolingWithArgmax2D, MaxUnpooling2D

def model_segnet(nClasses, image_shape=(480, 480, 3), kernel=3, pool_size=(2, 2)):
    inputs = Input(shape=image_shape)

    conv_1 = Conv2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Conv2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Conv2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Conv2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Conv2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Conv2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Conv2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Conv2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Conv2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Conv2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Conv2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Conv2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Conv2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Conv2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Conv2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Conv2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Conv2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Conv2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Conv2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Conv2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Conv2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Conv2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Conv2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Conv2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Conv2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Conv2D(nClasses, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    # conv_26 = Reshape((image_shape[0] * image_shape[1], nClasses), input_shape=(image_shape[0], image_shape[1], nClasses))(conv_26)

    outputs = Activation('softmax')(conv_26)
    print("Build decoder done..")

    segnet = Model(inputs=inputs, outputs=outputs, name="SegNet")

    return segnet

def model_segnetVGG16(nClasses, image_shape=(480, 480, 3), keep_prob=0.5):
  base_model = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)

  # for layer in base_model.layers:
  #   layer.trainable = False

  block5_pool = base_model.get_layer('block5_pool').output

  # Decoder follows
  X = UpSampling2D( (2,2))(block5_pool)
  X = Conv2D(512, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = Conv2D(512, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = BatchNormalization()(X)

  X = UpSampling2D( (2,2))(X)
  X = Conv2D(256, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = Conv2D(256, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = BatchNormalization()(X)

  X = UpSampling2D( (2,2))(X)
  X = Conv2D(128, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = Conv2D(128, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = BatchNormalization()(X)

  X = UpSampling2D( (2,2))(X)
  X = Conv2D(64, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = Conv2D(64, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = BatchNormalization()(X)

  X = UpSampling2D( (2,2))(X)
  X = Conv2D(32, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = Conv2D(32, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = BatchNormalization()(X)

  X = Conv2D( nClasses , (1, 1) , padding='valid')(X)
  X = BatchNormalization()(X)
  X = Activation('softmax', name='y_')(X)

  model = Model(inputs=[base_model.input] , outputs=[X])
  
  return model


def model_segnetVGG16_v2(nClasses, image_shape=(480, 480, 3)):
  base_model = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)

  for layer in base_model.layers:
    layer.trainable = False

  # block1_pool = base_model.get_layer('block1_pool').output
  # block2_pool = base_model.get_layer('block2_pool').output
  # block3_pool = base_model.get_layer('block3_pool').output
  # block4_pool = base_model.get_layer('block4_pool').output
  block5_pool = base_model.get_layer('block5_pool').output

  # Decoder follows
  X = UpSampling2D( (2,2))(block5_pool)
  X = Conv2D(512, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = Conv2D(512, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = BatchNormalization()(X)

  X = UpSampling2D( (2,2))(X)
  X = Conv2D(256, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = Conv2D(256, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = BatchNormalization()(X)

  X = UpSampling2D( (2,2))(X)
  X = Conv2D(128, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = Conv2D(128, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = BatchNormalization()(X)

  X = UpSampling2D( (2,2))(X)
  X = Conv2D(64, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = Conv2D(64, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = BatchNormalization()(X)

  X = UpSampling2D( (2,2))(X)
  X = Conv2D(32, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = Conv2D(32, (3, 3), padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = BatchNormalization()(X)

  X = Conv2D( nClasses , (1, 1) , padding='valid')(X)
  X = BatchNormalization()(X)
  X = Activation('softmax', name='y_')(X)

  model = Model(inputs=[base_model.input] , outputs=[X])
  
  return model


if __name__ == '__main__':
  model = model_segnet(nClasses = 3)
  model.summary()