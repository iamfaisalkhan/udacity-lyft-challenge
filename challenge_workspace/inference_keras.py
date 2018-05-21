import sys, skvideo.io, json, base64, cv2
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
from keras.models import Model, load_model
from keras.applications.vgg16 import  preprocess_input
import keras.backend as K

file = sys.argv[-1]

# Define encoder function
def encode(array):
	pil_img = Image.fromarray(array)
	buff = BytesIO()
	pil_img.save(buff, format="PNG")
	return base64.b64encode(buff.getvalue()).decode("utf-8")

video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1


MODEL_PATH = '/data/fcn_weighted_model.h5'


K.set_learning_phase(0)
model = load_model(MODEL_PATH)

BATCH_SIZE=32

X_arr = np.zeros((BATCH_SIZE, 480, 480, 3), dtype=np.float64)

m = video.shape[0]
for i in range(0, m, BATCH_SIZE):
  cnt = 0
  for j in range(i, min(i+BATCH_SIZE, m)):
    video[j] = cv2.cvtColor(video[j], cv2.COLOR_RGB2BGR)
    X_arr[cnt, :, :, :] = preprocess_input(cv2.resize(video[j], (480, 480)).astype(np.float64))
    cnt += 1

  result = model.predict(X_arr)

  for x in range(cnt):
    output = result[x].argmax(axis=2)
    binary_car_result = cv2.resize((output == 0).astype(np.uint8), (800, 600))
    binary_road_result = cv2.resize((output == 1).astype(np.uint8), (800, 600))
    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
    frame += 1

# Print output in proper json format
print (json.dumps(answer_key))