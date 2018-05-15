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


MODEL_PATH = './saved_models/fcn8_vgg16.model.hdf5'


K.set_learning_phase(0)
model = load_model(MODEL_PATH)

for rgb_frame in video:
        rgb_frame = cv2.resize(rgb_frame, (480, 480))
        rgb_frame = preprocess_input(rgb_frame.astype(np.float64))
        result = model.predict(np.expand_dims(rgb_frame, 0))[0]
        result = result.argmax(axis=2)
        binary_car_result = cv2.resize((result == 1).astype(np.uint8), (800, 600))
        binary_road_result = cv2.resize((result == 0).astype(np.uint8), (800, 600))
        answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
        
        frame+=1

# Print output in proper json format
print (json.dumps(answer_key))
