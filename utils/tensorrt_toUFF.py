# make TF to use only 1 of your devices :-)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Import libs
import uff
import tensorflow as tf
import keras.backend as K
from keras.applications.vgg16 import VGG16


model = VGG16(include_top=True, weights='imagenet')

# Get model input and output names
model_input = model.input.name.strip(':0')
model_output = model.output.name.strip(':0')
print(model_input, model_output)

graph = tf.get_default_graph().as_graph_def()
# expect something like:
# node {
#   name: "input_1"
#   op: "Placeholder"
#   attr {
#     key: "dtype"
#     value {
#       type: DT_FLOAT
#     }
#   }
#   attr {
# etc...
# Get session
sess = K.get_session()
# freeze graph and remove nodes used for training 
frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graph, [model_output])
frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
# Create UFF model and dump it on disk 
uff_model = uff.from_tensorflow(frozen_graph, [model_output])
dump = open('VGG16.uff', 'wb')
dump.write(uff_model)
dump.close()


