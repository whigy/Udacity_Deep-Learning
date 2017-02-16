# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:25:59 2017

@author: HWAG
"""

# -*- coding: utf-8 -*-
"""
Udacity assignment 3
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/3_regularization.ipynb

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

def accuracy(predictions, labels):
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
      

  
"""
###############################Problem 1###############################
###################training (with regulation)###########################
Introduce and tune L2 regularization for both logistic and neural network models. 
Remember that L2 amounts to adding a penalty on the norm of the weights to the loss. 
In TensorFlow, you can compute the L2 loss for a tensor t using nn.l2_loss(t). 
The right amount of regularization should improve your validation / test accuracy.
"""
batch_size = 128
node_size = 1024
beta = 0.0001

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
  weights1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, node_size]))
  biases1 = tf.Variable(tf.zeros([node_size]))
  
  weights2 = tf.Variable(
    tf.truncated_normal([node_size, num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))
  
  
  def forward_pass(data):
  # Training computation.
      layer1 = tf.matmul(data, weights1) + biases1
      relu = tf.nn.relu(layer1)
      layer2 = tf.matmul(relu, weights2) + biases2
      return layer2

  logits = forward_pass(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  loss += 0.5 * beta * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(forward_pass(tf_valid_dataset))
  test_prediction = tf.nn.softmax(forward_pass(tf_test_dataset))



num_steps = 3001

for i in range(3):
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
        if (step % 1500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
      print("Beta = %f; Test accuracy: %.1f%%" % (beta, accuracy(test_prediction.eval(), test_labels)))
      beta *=10
  
"""
Initialized
Minibatch loss at step 0: 309.463959
Minibatch accuracy: 10.2%
Validation accuracy: 29.2%
Minibatch loss at step 1500: 21.453453
Minibatch accuracy: 85.9%
Validation accuracy: 81.6%
Minibatch loss at step 3000: 15.537707
Minibatch accuracy: 79.7%
Validation accuracy: 82.8%
Beta = 0.000100; Test accuracy: 89.9%
Initialized
Minibatch loss at step 0: 362.480347
Minibatch accuracy: 12.5%
Validation accuracy: 33.4%
Minibatch loss at step 1500: 18.536198
Minibatch accuracy: 90.6%
Validation accuracy: 79.3%
Minibatch loss at step 3000: 15.198601
Minibatch accuracy: 78.9%
Validation accuracy: 82.6%
Beta = 0.001000; Test accuracy: 89.9%
Initialized
Minibatch loss at step 0: 341.851013
Minibatch accuracy: 11.7%
Validation accuracy: 27.4%
Minibatch loss at step 1500: 19.417274
Minibatch accuracy: 89.1%
Validation accuracy: 80.5%
Minibatch loss at step 3000: 14.554264
Minibatch accuracy: 82.8%
Validation accuracy: 82.8%
Beta = 0.010000; Test accuracy: 90.0%
"""

"""
###############################Problem 2###############################
Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?

SKIP
"""

"""
###############################Problem 3###############################
Introduce Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides nn.dropout() for that, but you have to make sure it's only inserted during training.
What happens to our extreme overfitting case?
"""
batch_size = 128
node_size = 1024
beta = 0.01
dropout_rate = 0.5

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
  weights1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, node_size]))
  biases1 = tf.Variable(tf.zeros([node_size]))
  
  weights2 = tf.Variable(
    tf.truncated_normal([node_size, num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))
  
  
  def forward_pass(data, dropout_rate):
  # Training computation.
      layer1 = tf.matmul(data, weights1) + biases1
      relu = tf.nn.relu(layer1)
      drop1 = tf.nn.dropout(relu, dropout_rate)
      layer2 = tf.matmul(drop1, weights2) + biases2
      #drop2 = tf.nn.dropout(layer2, dropout_rate)
      return layer2
      
  logits = forward_pass(tf_train_dataset, dropout_rate)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  loss += 0.5 * beta * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(forward_pass(tf_train_dataset, 1))
  valid_prediction = tf.nn.softmax(forward_pass(tf_valid_dataset, 1))
  test_prediction = tf.nn.softmax(forward_pass(tf_test_dataset, 1))



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
  print("Beta = %f; Test accuracy: %.1f%%" % (beta, accuracy(test_prediction.eval(), test_labels)))

"""
Initialized
Minibatch loss at step 0: 2100.224854
Minibatch accuracy: 12.5%
Validation accuracy: 24.1%
Minibatch loss at step 500: 129.187683
Minibatch accuracy: 84.4%
Validation accuracy: 81.9%
Minibatch loss at step 1000: 11.073137
Minibatch accuracy: 83.6%
Validation accuracy: 84.3%
Minibatch loss at step 1500: 1.411623
Minibatch accuracy: 87.5%
Validation accuracy: 84.1%
Minibatch loss at step 2000: 0.707403
Minibatch accuracy: 90.6%
Validation accuracy: 84.0%
Minibatch loss at step 2500: 0.726740
Minibatch accuracy: 86.7%
Validation accuracy: 84.0%
Minibatch loss at step 3000: 0.790406
Minibatch accuracy: 84.4%
Validation accuracy: 84.2%
Beta = 0.010000; Test accuracy: 91.3%
"""  
  
"""
###############################Problem 4###############################
Try to get the best performance you can using a multi-layer model! 
The best reported test accuracy using a deep network is 97.1%.
One avenue you can explore is to add multiple layers.
Another one is to use learning rate decay:
global_step = tf.Variable(0)  # count the number of steps taken.
learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
"""