"""
https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
"""

import keras
from keras.models import load_model
import keras.backend as K
import tensorflow as tf

from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util
from tensorflow.tools.graph_transforms import TransformGraph

from tensorflow.python.tools import freeze_graph
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile


MODEL_NAME = 'lyft_fcn8'

input_graph_path = 'freeze/graph.pbtxt'
checkpoint_path = './freeze/checkpoint.ckpt-19'
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'
output_optimized_graph_name = 'frozen_optimized_'+MODEL_NAME+'.pb'

# freeze_graph.freeze_graph(input_graph_path, input_saver="",
#                           input_binary=False, input_checkpoint=checkpoint_path, 
#                           output_node_names="y_/truediv", restore_op_name="save/restore_all",
#                           filename_tensor_name="save/Const:0", 
#                           output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")


input_graph_def = graph_pb2.GraphDef()
with gfile.Open(output_frozen_graph_name, "rb") as f:
  data = f.read()
  input_graph_def.ParseFromString(data)

print ([print(n.name) for n in input_graph_def.node])

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
      input_graph_def,
      input_node_names=['input_1:0'],
      output_node_names=['y_/truediv'],
      placeholder_type_enum=dtypes.float32.as_datatype_enum)


f = gfile.FastGFile(output_optimized_graph_name, "w")
f.write(output_graph_def.SerializeToString())


# def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
#   """
#   Freezes the state of a session into a pruned computation graph.

#   Creates a new computation graph where variable nodes are replaced by
#   constants taking their current value in the session. The new graph will be
#   pruned so subgraphs that are not necessary to compute the requested
#   outputs are removed.
#   @param session The TensorFlow session to be frozen.
#   @param keep_var_names A list of variable names that should not be frozen,
#                         or None to freeze all the variables in the graph.
#   @param output_names Names of the relevant graph outputs.
#   @param clear_devices Remove the device directives from the graph for better portability.
#   @return The frozen graph definition.
#   """
#   graph = session.graph
#   with graph.as_default():
#     freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
#     output_names = output_names or []
#     output_names += [v.op.name for v in tf.global_variables()]
#     input_graph_def = graph.as_graph_def()
#     if clear_devices:
#       for node in input_graph_def.node:
#         node.device = ""

#     transforms = ["remove_device", "remove_control_dependencies"]
#     transformed_graph_def = TransformGraph(input_graph_def, [], output_names, transforms)
#     frozen_graph = graph_util.convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)

#     return frozen_graph


# model = load_model('saved_models/fcn8_vgg16.model.hdf5')
# K.set_learning_phase(0)
# frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
# tf.train.write_graph(frozen_graph, "output", "fcn8.pb", as_text=False)