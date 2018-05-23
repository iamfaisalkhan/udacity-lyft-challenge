import pycuda.driver as cuda
import pycuda.autoinit
import argparse

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

import tensorrt as trt

from tensorrt.parsers import uffparser

import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# This is a helper function, provided by TensorRT devs, to run inference
def infer(context, input_img, batch_size):
    # load engine
    engine = context.get_engine()
    assert(engine.get_nb_bindings() == 2)
    # create output array to receive data
    dims = engine.get_binding_dimensions(1).to_DimsCHW()
    elt_count = dims.C() * dims.H() * dims.W() * batch_size
    # convert input data to Float32
    input_img = input_img.astype(np.float32)
    # Allocate pagelocked memory
    output = cuda.pagelocked_empty(elt_count, dtype=np.float32)
    # alocate device memory
    d_input = cuda.mem_alloc(batch_size * input_img.size * input_img.dtype.itemsize)
    d_output = cuda.mem_alloc(batch_size * output.size * output.dtype.itemsize)

    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    # execute model
    context.enqueue(batch_size, bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # return predictions
    return output

# load model
uff_model = open('VGG16.uff', 'rb').read()
# create model parser
parser = uffparser.create_uff_parser()
parser.register_input("input_1", (3, 224, 224), 0)
parser.register_output("predictions/Softmax")
# create inference engine and context (aka session)
trt_logger = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
engine = trt.utils.uff_to_trt_engine(logger=trt_logger,
                                     stream=uff_model,
                                     parser=parser,
                                     max_batch_size=1, # 1 sample at a time
                                     max_workspace_size= 1 << 30, # 1 GB GPU memory workspace
                                     datatype=trt.infer.DataType.FLOAT) # that's very cool, you can set precision
context = engine.create_execution_context()

# load and preprocess image
test_image = image.load_img('hotdog.jpg', target_size=(224, 224, 3))
test_image = image.img_to_array(test_image)
processed_im = preprocess_input(np.expand_dims(test_image, 0))[0, :, :, :]
# prepare image for TRT3 engine
processed_im = np.transpose(processed_im, axes=(2, 0, 1))
processed_im = processed_im.copy(order='C')
# infer probs
prediction_proba = infer(context, processed_im, 1)
# decode labels
decode_predictions(np.expand_dims(prediction_proba, 0))
