# -*- coding: utf-8 -*-
"""
Udacity assignment 2
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/2_fullyconnected.ipynb

@author: HWAG
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

#cd C:\Users\HWAG\Documents\Python Scripts\Udacity-DL\Udacity_code
pickle_file = '../assign1_notMNIST_data/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
  
image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

  
"""
###################gradient descent training###########################
"""
# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = len(train_dataset)

graph = tf.Graph()
with graph.as_default():

  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
  tf_train_labels = tf.constant(train_labels[:train_subset])
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  
  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
  
  def accuracy(predictions, labels):
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


num_steps = 3001
with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(
        predictions, train_labels[:train_subset, :]))
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  
"""
Initialized
Loss at step 0: 18.940586
Training accuracy: 12.1%
Validation accuracy: 15.6%
Loss at step 100: 2.490209
Training accuracy: 71.5%
Validation accuracy: 71.2%
Loss at step 200: 2.073720
Training accuracy: 73.9%
Validation accuracy: 73.6%
Loss at step 300: 1.843755
Training accuracy: 74.9%
Validation accuracy: 74.7%
Loss at step 400: 1.686432
Training accuracy: 75.5%
Validation accuracy: 75.2%
Loss at step 500: 1.569115
Training accuracy: 75.8%
Validation accuracy: 75.7%
Loss at step 600: 1.477350
Training accuracy: 76.1%
Validation accuracy: 75.9%
Loss at step 700: 1.403038
Training accuracy: 76.3%
Validation accuracy: 76.0%
Loss at step 800: 1.341227
Training accuracy: 76.6%
Validation accuracy: 76.2%
Loss at step 900: 1.288720
Training accuracy: 76.8%
Validation accuracy: 76.4%
Loss at step 1000: 1.243357
Training accuracy: 77.0%
Validation accuracy: 76.5%
Loss at step 1100: 1.203624
Training accuracy: 77.2%
Validation accuracy: 76.6%
Loss at step 1200: 1.168435
Training accuracy: 77.3%
Validation accuracy: 76.8%
Loss at step 1300: 1.136986
Training accuracy: 77.5%
Validation accuracy: 77.0%
Loss at step 1400: 1.108667
Training accuracy: 77.7%
Validation accuracy: 77.2%
Loss at step 1500: 1.083003
Training accuracy: 77.9%
Validation accuracy: 77.4%
Loss at step 1600: 1.059617
Training accuracy: 78.0%
Validation accuracy: 77.5%
Loss at step 1700: 1.038203
Training accuracy: 78.2%
Validation accuracy: 77.7%
Loss at step 1800: 1.018513
Training accuracy: 78.3%
Validation accuracy: 77.9%
Loss at step 1900: 1.000341
Training accuracy: 78.4%
Validation accuracy: 78.1%
Loss at step 2000: 0.983513
Training accuracy: 78.6%
Validation accuracy: 78.2%
Loss at step 2100: 0.967883
Training accuracy: 78.7%
Validation accuracy: 78.3%
Loss at step 2200: 0.953329
Training accuracy: 78.8%
Validation accuracy: 78.4%
Loss at step 2300: 0.939741
Training accuracy: 78.9%
Validation accuracy: 78.5%
Loss at step 2400: 0.927028
Training accuracy: 79.0%
Validation accuracy: 78.5%
Loss at step 2500: 0.915109
Training accuracy: 79.1%
Validation accuracy: 78.6%
Loss at step 2600: 0.903912
Training accuracy: 79.2%
Validation accuracy: 78.7%
Loss at step 2700: 0.893375
Training accuracy: 79.3%
Validation accuracy: 78.9%
Loss at step 2800: 0.883442
Training accuracy: 79.4%
Validation accuracy: 79.0%
Loss at step 2900: 0.874063
Training accuracy: 79.5%
Validation accuracy: 79.0%
Loss at step 3000: 0.865195
Training accuracy: 79.6%
Validation accuracy: 79.1%
Test accuracy: 86.4%
"""
  
  
  
"""
###################stochastic gradient descent training###########################
"""
batch_size = 128

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)



num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
  
  
  """
  Initialized
Minibatch loss at step 0: 17.087204
Minibatch accuracy: 16.4%
Validation accuracy: 18.3%
Minibatch loss at step 500: 0.952143
Minibatch accuracy: 80.5%
Validation accuracy: 76.2%
Minibatch loss at step 1000: 1.315179
Minibatch accuracy: 81.2%
Validation accuracy: 76.5%
Minibatch loss at step 1500: 0.837123
Minibatch accuracy: 82.8%
Validation accuracy: 76.9%
Minibatch loss at step 2000: 0.977976
Minibatch accuracy: 80.5%
Validation accuracy: 77.2%
Minibatch loss at step 2500: 0.968226
Minibatch accuracy: 77.3%
Validation accuracy: 77.8%
Minibatch loss at step 3000: 1.051636
Minibatch accuracy: 77.3%
Validation accuracy: 78.6%
Test accuracy: 86.6%
  """
  
  
"""
Problem
Turn the logistic regression example with SGD into a 1-hidden layer neural network with rectified linear units nn.relu()
 and 1024 hidden nodes. This model should improve your validation / test accuracy.

"""