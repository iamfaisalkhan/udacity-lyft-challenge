import sys, skvideo.io, json, base64, cv2
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
from keras.models import Model, load_model
from keras.applications.vgg16 import  preprocess_input

sys.path.append('./')

import keras.backend as K

file = sys.argv[-1]

# Define encoder function
def encode(array):
	# pil_img = Image.fromarray(array)
	# buff = BytesIO()
	# pil_img.save(buff, format="PNG")
	# return base64.b64encode(buff.getvalue()).decode("utf-8")
  retval, buffer = cv2.imencode('.png', array)
  return base64.b64encode(buffer).decode("utf-8")

video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

# MODEL_PATH = './saved_models/unet/unet_v2/model_saved.h5'
MODEL_PATH = './saved_models/fcn8/fcn8_v6/fcn8_v6.h5'
# MODEL_PATH = './saved_models/fcn8/fcn8LowRes/fcn8VGG16LowRes.h5'

K.set_learning_phase(0)
model = load_model(MODEL_PATH)
# model = load_model(MODEL_PATH)
#
BATCH_SIZE=32
IMG_SIZE=(384, 384)

X_arr = np.zeros((BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float64)
m = video.shape[0]

for i in range(0, m, BATCH_SIZE):
  cnt = 0
  for j in range(i, min(i+BATCH_SIZE, m)):
    X_arr[cnt] = preprocess_input(cv2.resize(video[j], (IMG_SIZE[1], IMG_SIZE[0])).astype(np.float64))
    cnt += 1

  result = model.predict(X_arr[0:cnt])

  for x in range(cnt):
    output = result[x].argmax(axis=2)
    binary_car_result = cv2.resize((output == 0).astype(np.uint8), (800, 600))
    binary_car_result[496:, :] = 0
    binary_road_result = cv2.resize((output == 1).astype(np.uint8), (800, 600))
    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
    frame += 1

# Print output in proper json format
print (json.dumps(answer_key))
