#!/usr/bin/env python

""" optimizer.py: Optimize keras model. Based on https://github.com/amir-abdi/keras_to_tensorflow"""

__author__ = "Faisal Khan"

import argparse
import sys
from pathlib import Path

import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from tensorflow.python.framework import graph_util, graph_io
from tensorflow.tools.graph_transforms import TransformGraph

sys.path.append('./')
from models.segnet_custom_layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


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
        ops = sess.graph.get_operations()
        n_ops = len(ops)
        return sess.graph, ops

def optimize(model_file, output_path, output_model_file, output_graph_def=False):
  model = None
  try:
    print (model_file)
    model = load_model(model_file,
                   {'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
                    'MaxUnpooling2D' : MaxUnpooling2D
                   })
  except ValueError as err:
    print('Model probably saved with only weights')
    raise err

  # Output paths etc..
  pred_node_names = [model.output.op.name]
  print ("Output node name ", pred_node_names)
  Path(output_path).mkdir(parents=True, exist_ok=True)

  sess = K.get_session()
  K.set_learning_phase(0)

  if output_graph_def:
    tf.train.write_graph(sess.graph.as_graph_def(), output_path, 'model.txt', as_text=True)
    print("Saved graph definition as ascii", str(Path(output_path)/'model.txt'))

  ops = sess.graph.get_operations()
  n_ops = len(ops)
  print ("Graph-In operations = ", n_ops)

  # transforms = ["quantize_weights", "quantize_nodes"]
  # transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
  constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
  graph_io.write_graph(constant_graph, output_path, output_model_file, as_text=False)
  graph_file = "{}/{}".format(str(Path(output_path)), output_model_file)
  print ("Saved the frozen graph ", graph_file)

  return graph_file

def main():
  parser = argparse.ArgumentParser(description='Optimize keras model for inference')
  required = parser.add_argument_group('Required Arguments')
  optional = parser.add_argument_group('Optional Arguments')

  required.add_argument('-i', action='store', dest='input_model', type=str, required=True)
  optional.add_argument('-o', action='store', dest='output_path', type=str, default='./output')
  optional.add_argument('--ops', action='store_true', dest='show_ops')

  args = parser.parse_args()

  model_file = args.input_model
  output_path = args.output_path
  show_ops = args.show_ops

  p = optimize(model_file, output_path, '%s.pb'%Path(model_file).name, True)

  if (show_ops):
    _, ops = load_graph(p)
    print ("Grap-Out operations = ", len(ops))


if __name__ == '__main__':
  main()