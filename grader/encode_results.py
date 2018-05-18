import sys
import glob
import numpy as np
import skvideo.io
import json, base64
from PIL import Image
from io import BytesIO, StringIO
from skimage.io import imread

def encode(array):
  pil_img = Image.fromarray(array)
  buff = BytesIO()
  pil_img.save(buff, format="PNG")
  return base64.b64encode(buff.getvalue()).decode("utf-8")

if len(sys.argv) < 1:
  print ("Missing argument: python %s path", sys.argv[0])

path = sys.argv[1]
# files = glob.glob("%s/CameraRGB/*.png"%path)
# files.sort()

# tmp = imread(files[0])
# m = len(files)

# data = np.empty((m, *tmp.shape))

# for ind, file in enumerate(files):
#   data[ind] = imread(file)

# skvideo.io.vwrite("video.mp4", data)

files = glob.glob("%s/CameraSeg/*.png"%path)
files.sort()

truth_key = {}
frame = 1

for ind, file in enumerate(files):
  img = imread(file)[:, :, 0]
  car = np.zeros_like(img)
  road = np.zeros_like(img)

  car = (img == 10).astype(np.uint8)
  road = (img == 7).astype(np.uint8)
  
  truth_key[frame] = [encode(car), encode(road)] 
  frame += 1

print (json.dumps(truth_key))


