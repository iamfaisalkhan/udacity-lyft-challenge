import os
import keras
import keras.backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss
    
def jaccard_distance_loss(y_true, y_pred, smooth=100):
  """
  Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
          = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

  The jaccard distance loss is usefull for unbalanced datasets. This has been
  shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
  gradient.

  Ref: https://en.wikipedia.org/wiki/Jaccard_index

  @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
  @author: wassname
  """
  intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
  sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
  jac = (intersection + smooth) / (sum_ - intersection + smooth)
  return (1 - jac) * smooth


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
            verbose = 2,
            lr=1e-4,
            gpus = 1):

  weight_path = "{}/{}.hdf5".format(output_path, 'model')
  freeze_path = "{}/freeze".format(output_path)
    
  sess = K.get_session()

  tf_graph = sess.graph
  tf_saver = tf.train.Saver()
  tfckptcb = TFCheckpointCallback(tf_saver, sess, output_path)

  # Call backs
  earlystop = EarlyStopping(monitor="val_loss", mode="min", patience=30)
  checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                               save_best_only=True, mode='min', save_weights_only=True)
  reducelr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
  tensorboar = TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True, write_images=True)

  callbacks_list = [checkpoint, tfckptcb, earlystop, reducelr, tensorboar]
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

  return history

if __name__ == '__main__':
  train_nn()