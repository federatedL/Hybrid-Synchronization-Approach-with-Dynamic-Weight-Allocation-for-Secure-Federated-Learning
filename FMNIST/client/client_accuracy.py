import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import subprocess
import pickle
import argparse
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import load_model

X_test = None
y_test = None

def read_test_data():
  with open('test_data_X.pkl', 'rb') as data_X:
        X_test = pickle.load(data_X)

  with open('test_data_y.pkl', 'rb') as data_y:
        y_test = pickle.load(data_y)

  return X_test, y_test

def write_result(client_no, result):
  #combined_acc = {'loss': results[0], 'acc': results[1]}
  with open('client_acc_{}.pkl'.format(client_no), 'wb') as client_acc:
      pickle.dump(result, client_acc)

  return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('client_no', type=str, help="The serial of client passed from server")
    args = parser.parse_args()
    client_no = args.client_no

    X_test, y_test = read_test_data()
    model = load_model('iris.keras')
    test_accuracy = tf.keras.metrics.Accuracy()
    ds_test_batch  = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(10)
    for (x, y) in ds_test_batch:
        logits = model(x, training=False)
        prediction = tf.math.argmax(logits, axis=1, output_type=tf.int64)
        test_accuracy(prediction, y)

    result = test_accuracy.result().numpy().item()
    write_result(client_no, result)