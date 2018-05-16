import keras
import keras.backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

from models.fcn8 import model_fcn8
from generators import train_and_lab_gen_func
from load_data import load_data

class TFCheckpointCallback(keras.callbacks.Callback):
  def __init__(self, saver, sess):
    self.saver = saver
    self.sess = sess

  def on_epoch_end(self, epoch, logs=None):
    self.saver.save(self.sess, 'freeze/checkpoint.ckpt', global_step=epoch)


def train_nn():
  train_df, dev_df = load_data('./data')

  model = model_fcn8(3)
  sess = K.get_session()

  tf_graph = sess.graph
  tf_saver = tf.train.Saver()
  tfckptcb = TFCheckpointCallback(tf_saver, sess)

  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  weight_path = "{}_weights.best.hdf5".format('fcn8_vgg16')

  checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                               save_best_only=True, mode='min', save_weights_only=True)

  batch_size = 16
  train_and_lab_gen = train_and_lab_gen_func(train_df, batch_size=batch_size)
  valid_and_lab_gen = train_and_lab_gen_func(dev_df, batch_size=batch_size)
  callbacks_list = [checkpoint, tfckptcb]

  history = model.fit_generator(train_and_lab_gen,
                                steps_per_epoch=800 // batch_size,
                                validation_data=valid_and_lab_gen,
                                validation_steps=200 // batch_size,
                                epochs=10,
                                workers=4,
                                use_multiprocessing=True,
                                callbacks=callbacks_list
                                )


if __name__ == '__main__':
  train_nn()