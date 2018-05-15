#!/usr/bin/env python

""" optimizer.py: Optimize keras model. Based on https://github.com/amir-abdi/keras_to_tensorflow"""

__author__ = "Faisal Khan"

import argparse
from pathlib import Path


def optimize(model_file, output_path, output_model_file, output_graph_def=False):
  import tensorflow as tf
  from keras.models import load_model
  from keras import backend as K
  from tensorflow.python.framework import graph_util, graph_io
  from tensorflow.tools.graph_transforms import TransformGraph

  K.set_learning_phase(0)

  model = None
  try:
    print (model_file)
    model = load_model(model_file)
  except ValueError as err:
    print('Model probably saved with only weights')
    raise err

  # Output paths etc..
  pred_node_names = [model.output.op.name]
  Path(output_path).mkdir(parents=True, exist_ok=True)

  sess = K.get_session()

  if output_graph_def:
    tf.train.write_graph(sess.graph.as_graph_def(), output_path, 'mode.ascii', as_text=True)
    print("Saved graph definition as ascii", str(Path(output_path)/'model.ascii'))

  # transforms = ["quantize_weights", "quantize_nodes"]
  # transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
  constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
  graph_io.write_graph(constant_graph, output_path, output_model_file, as_text=False)
  print ("Saved the frozen graph at : ", str(Path(output_path)/output_model_file))

def main():
  parser = argparse.ArgumentParser(description='Optimize keras model for inference')
  required = parser.add_argument_group('Required Arguments')
  optional = parser.add_argument_group('Optional Arguments')
  required.add_argument('-i', action='store', dest='input_model', type=str, required=True)
  optional.add_argument('-output', action='store', dest='output_path', type=str, default='./output')
  args = parser.parse_args()

  model_file = args.input_model


  optimize(model_file, './output', '%s.pb'%Path(model_file).name, True)

if __name__ == '__main__':
  main()