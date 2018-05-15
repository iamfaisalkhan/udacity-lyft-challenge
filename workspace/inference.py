import sys, skvideo.io, json, base64, cv2
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
from keras.models import Model, load_model
from keras.applications.vgg16 import  preprocess_input
import tensorflow as tf
import time

file = sys.argv[-1]

# Define encoder function
def encode(array):
	pil_img = Image.fromarray(array)
	buff = BytesIO()
	pil_img.save(buff, format="PNG")
	return base64.b64encode(buff.getvalue()).decode("utf-8")

def load_graph(frozen_graph_filename):
  with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)

  return graph


video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

t0 = time.time()

MODEL_PATH = '../saved_models/fcn8_tf_1.7.pb'

graph = load_graph(MODEL_PATH)

X = graph.get_tensor_by_name('import/input_1:0')
Yhat = graph.get_tensor_by_name('import/activation_1/truediv:0')

print ("Loaded model in ", (time.time() - t0), " seconds")

with tf.Session(graph=graph) as session:
  for rgb_frame in video:
    rgb_frame = cv2.resize(rgb_frame, (480, 480))
    rgb_frame = preprocess_input(rgb_frame.astype(np.float64))
    result = session.run(Yhat, feed_dict={X : np.expand_dims(rgb_frame, 0)})[0]
    result = result.argmax(axis=2)
    binary_car_result = cv2.resize((result == 1).astype(np.uint8), (800, 600))
    binary_road_result = cv2.resize((result == 0).astype(np.uint8), (800, 600))
    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
        
    frame+=1

# Print output in proper json format
print (json.dumps(answer_key))
