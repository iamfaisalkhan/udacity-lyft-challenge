import sys, skvideo.io, json, base64, cv2
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
from keras.models import Model, load_model
from keras.applications.vgg16 import  preprocess_input
import tensorflow as tf
import time
import multiprocessing as mp

file = sys.argv[-1]

# Define encoder function
def encode(array):
  pil_img = Image.fromarray(array)
  buff = BytesIO()
  pil_img.save(buff, format="PNG")
  return base64.b64encode(buff.getvalue()).decode("utf-8")

def load_graph(graph_file, use_xla=False):
    jit_level = 0
    config = tf.ConfigProto()
    if use_xla:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        return sess.graph

video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

MODEL_PATH = '/data/fcn8VGG16LowRes_opt.pb'

graph = load_graph(MODEL_PATH, True)


X = graph.get_tensor_by_name('input_1:0')
Yhat = graph.get_tensor_by_name('y_/truediv:0')
pred = tf.argmax(Yhat, axis=-1, output_type=tf.int32)
final = tf.image.resize_images(tf.expand_dims(pred, -1), [600, 800])

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

BATCH_SIZE=80

X_arr = np.zeros((BATCH_SIZE, 384, 384, 3), dtype=np.float64)

m = video.shape[0]
with tf.Session(graph=graph, config=config) as session:
  for i in range(0, m, BATCH_SIZE):
    cnt = 0
    for j in range(i, min(i+BATCH_SIZE, m)):
        X_arr[cnt, :, :, :] = preprocess_input(cv2.resize(video[j], (384, 384)).astype(np.float32))
        cnt += 1

    result = session.run(final, feed_dict={X : X_arr[0:cnt]})
    for x in range(cnt):
      binary_car_result = (result[x, :, :, 0] == 0).astype(np.uint8)
      binary_road_result = (result[x, :, :, 0] == 1).astype(np.uint8)
      answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
      frame += 1

# Print output in proper json format
print (json.dumps(answer_key))
