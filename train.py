import os
import keras
import keras.backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import Adam

class TFCheckpointCallback(keras.callbacks.Callback):
  def __init__(self, saver, sess, path):
    self.saver = saver
    self.sess = sess
    self.path = path

  def on_epoch_end(self, epoch, logs=None):
    self.saver.save(self.sess, "{}/freeze/checkpoint.ckpt".format(self.path), global_step=epoch)


def train_nn(model,
            train_gen, 
            valid_gen, 
            training_size,
            validation_size,
            output_path = './output',
            batch_size = 16,
            epochs = 20,
            workers = 4,
            lr=1e-4,
            verbose = 2,
            gpus = 1):

  weight_path = "{}/{}.hdf5".format(output_path, 'model')
  freeze_path = "{}/freeze".format(output_path)
    
  sess = K.get_session()

  tf_graph = sess.graph
  tf_saver = tf.train.Saver()
  tfckptcb = TFCheckpointCallback(tf_saver, sess, output_path)

  if gpus > 1:
    model = multi_gpu_model(model, gpus)

  opt = Adam(lr=lr)
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

  # Call backs
  earlystop = EarlyStopping(monitor="val_loss", mode="min", patience=10)
  checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                               save_best_only=True, mode='min', save_weights_only=True)
  reducelr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)

  callbacks_list = [checkpoint, tfckptcb, earlystop, reducelr]
  history = model.fit_generator(train_gen,
                                steps_per_epoch=training_size//(batch_size * gpus),
                                validation_data=valid_gen,
                                validation_steps=validation_size // (batch_size * gpus),
                                epochs=epochs,
                                workers=workers,
                                use_multiprocessing=True,
                                callbacks=callbacks_list
                                )

  tf.train.write_graph(tf_graph.as_graph_def(),
                       freeze_path, 'graph.pbtxt', as_text=True)
  tf.train.write_graph(tf_graph.as_graph_def(),
                       freeze_path, 'graph.pb', as_text=False)

  return model

if __name__ == '__main__':
  train_nn()