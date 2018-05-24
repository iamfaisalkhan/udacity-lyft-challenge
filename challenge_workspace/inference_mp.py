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

pool = mp.Pool(processes=3)
q = mp.Queue()

def process_results(result, fs, size):
  answer = encode(result)
  return (f, p, answer)

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
m = video.shape[0]

answer_key = {}
for i in range(m):
  answer_key[i] = ['', '']

# Frame numbering starts at 1
frame = 1


MODEL_PATH = './saved_models/fcn8_extended_training/model_saved.h5.pb'

graph = load_graph(MODEL_PATH, True)

X = graph.get_tensor_by_name('input_1:0')
Yhat = graph.get_tensor_by_name('y_/truediv:0')
pred = tf.argmax(Yhat, axis=-1, output_type=tf.int32)
final = tf.image.resize_images(tf.expand_dims(pred, -1), [600, 800])

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2

BATCH_SIZE=32

X_arr = np.zeros((BATCH_SIZE, 480, 480, 3), dtype=np.float64)

cnt = 0
for j in range(i, min(i+BATCH_SIZE, m)):
  X_arr[cnt, :, :, :] = preprocess_input(cv2.resize(video[j], (480, 480)).astype(np.float32))
  cnt += 1

with tf.Session(graph=graph) as session:
  for i in range(0, m, BATCH_SIZE):
    cnt = min(i+BATCH_SIZE, m)
    # for j in range(i, min(i+BATCH_SIZE, m)):
    #     X_arr[cnt, :, :, :] = preprocess_input(cv2.resize(video[j], (480, 480)).astype(np.float32))
    #     cnt += 1

    result = session.run(final, feed_dict={X : X_arr[0:cnt]})
    # _ = pool.apply_async(process_results, result)

    # for x in range(cnt):
    #   binary_car_result = (result[x, :, :, 0] == 0).astype(np.uint8)
    #   binary_road_result = (result[x, :, :, 0] == 1).astype(np.uint8)
    #   answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
    #   frame += 1

# Print output in proper json format
print (json.dumps(answer_key))

