from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import time
import os
os.chdir("C:/Users/HWAG/Documents/Python Scripts/Udacity-DL/Udacity_code")
#import net_model as alex

#cd C:\Users\HWAG\Documents\Python Scripts\Udacity-DL\Udacity_code
pickle_file = '../assign1_notMNIST_data/notMNIST.pickle'

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  Part of the code from:
   https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/models/image/cifar10/cifar10.py
  """

  #var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  var =  weight_variable(shape)
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

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
num_channels = 1 # grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
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
  
batch_size = 32
patch_size = 5
depth = 16
num_hidden = 64
dropout = 0.5

graph = tf.Graph()

with graph.as_default():


  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
#   Variables.
  weights = {
        'w_conv_2':_variable_with_weight_decay('weights', shape=[5, 5, 1, 16], stddev=1e-4, wd=0.001),  #weight_decay = 0.0005
        'w_conv_3':_variable_with_weight_decay('weights', shape=[3, 3, 16, 32], stddev=1e-4, wd=0.0005),  #weight_decay = 0.0005
        'w_conv_4':_variable_with_weight_decay('weights', shape=[3, 3, 32, 64], stddev=1e-4, wd=0.0005),  #weight_decay = 0.0005
        'w_conv_5':_variable_with_weight_decay('weights', shape=[3, 3, 64, 48], stddev=1e-4, wd=0.0005), #weight_decay = 0.0005
        'w_full_1':_variable_with_weight_decay('weights', shape=[image_size // 4 * image_size // 4 * 48, 128], stddev=1e-4, wd=0.0005),  #weight_decay = 0.0005
        'w_full_2':_variable_with_weight_decay('weights', shape=[128, 64], stddev=1e-4, wd=0.0005),  #weight_decay = 0.0005
        'w_full_3':_variable_with_weight_decay('weights', shape=[64, 10], stddev=1e-4, wd=0.0005),  #weight_decay = 0.0005
  }
  biases = {
        'b_conv_2': tf.Variable(tf.random_normal([16])),
        'b_conv_3': tf.Variable(tf.random_normal([32])),
        'b_conv_4': tf.Variable(tf.random_normal([64])),
        'b_conv_5': tf.Variable(tf.random_normal([48])),
        'b_full_1': tf.Variable(tf.random_normal([128])),
        'b_full_2': tf.Variable(tf.random_normal([64])),
        'b_full_3': tf.Variable(tf.random_normal([10])),
        #'b_soft': tf.Variable(tf.random_normal([n_class]))
  }
#  weights = {
#        'w_conv_2':_variable_with_weight_decay('weights', shape=[5, 5, 1, 128], stddev=1e-4, wd=0.0005),  #weight_decay = 0.0005
#        'w_conv_3':_variable_with_weight_decay('weights', shape=[3, 3, 128, 192], stddev=1e-4, wd=0.0005),  #weight_decay = 0.0005
#        'w_conv_4':_variable_with_weight_decay('weights', shape=[3, 3, 192, 192], stddev=1e-4, wd=0.0005),  #weight_decay = 0.0005
#        'w_conv_5':_variable_with_weight_decay('weights', shape=[3, 3, 192, 128], stddev=1e-4, wd=0.0005), #weight_decay = 0.0005
#        'w_full_1':_variable_with_weight_decay('weights', shape=[image_size // 4 * image_size // 4 * 128, 1000], stddev=1e-4, wd=0.0005),  #weight_decay = 0.0005
#        'w_full_2':_variable_with_weight_decay('weights', shape=[1000, 1000], stddev=1e-4, wd=0.0005),  #weight_decay = 0.0005
#        'w_full_3':_variable_with_weight_decay('weights', shape=[1000, 10], stddev=1e-4, wd=0.0005),  #weight_decay = 0.0005
#  }
#  biases = {
#        'b_conv_2': tf.Variable(tf.random_normal([128])),
#        'b_conv_3': tf.Variable(tf.random_normal([192])),
#        'b_conv_4': tf.Variable(tf.random_normal([192])),
#        'b_conv_5': tf.Variable(tf.random_normal([128])),
#        'b_full_1': tf.Variable(tf.random_normal([1000])),
#        'b_full_2': tf.Variable(tf.random_normal([1000])),
#        'b_full_3': tf.Variable(tf.random_normal([10])),
#        #'b_soft': tf.Variable(tf.random_normal([n_class]))
#  }

  def alex_net(train_data, dropout):
    # second layer
    stride = [1, 1, 1, 1]
    conv2 = tf.nn.conv2d(train_data, weights['w_conv_2'], stride, padding = 'SAME', name = 'conv2')
    print(conv2)
    bias = tf.nn.bias_add(conv2, biases['b_conv_2'])
    relu2 = tf.nn.relu(bias, name = 'relu2')
    norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    print(pool2)
    
    # third layer
    conv3 = tf.nn.conv2d(pool2, weights['w_conv_3'], stride, padding = 'SAME', name = 'conv3')
    print(conv3)
    bias = tf.nn.bias_add(conv3, biases['b_conv_3'])
    relu3 = tf.nn.relu(bias, name = 'relu3')
    
    # froth layer
    conv4 = tf.nn.conv2d(relu3, weights['w_conv_4'], stride, padding = 'SAME', name = 'conv4')
    print(conv4)
    bias = tf.nn.bias_add(conv4, biases['b_conv_4'])
    relu4 = tf.nn.relu(bias, name = 'relu4')
    
    # fifth layer
    conv5 = tf.nn.conv2d(relu4, weights['w_conv_5'], stride, padding = 'SAME', name = 'conv5')
    print(conv5)
    bias = tf.nn.bias_add(conv5, biases['b_conv_5'])
    relu5 = tf.nn.relu(bias, name = 'relu5')
    pool5 = tf.nn.max_pool(relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    print(pool5)

    ## fully connected layers##
    #  first layer
    shape = pool5.get_shape().as_list()
    reshape1 = tf.reshape(pool5, [shape[0], shape[1] * shape[2] * shape[3]])
    #dim = weights['w_full_1'].get_shape().as_list()[0]
    #reshape1 = tf.reshape(pool5, [-1, dim]) # Reshape output to fit fc layer input
    print(reshape1)
    full1 = tf.matmul(reshape1, weights['w_full_1'])
    print(full1)
    relu_f1 = tf.nn.relu(full1 + biases['b_full_1'], name='fc1')
    print(relu_f1)
    drop1 = tf.nn.dropout(relu_f1, dropout)
    print(drop1)
    
      # second layer
    full2 = tf.matmul(drop1, weights['w_full_2'])
    print(full2)
    relu_f2 = tf.nn.relu(full2 + biases['b_full_2'], name='fc2')
    drop2 = tf.nn.dropout(relu_f2, dropout)
    print(drop2)

    # third layer
    full3 = tf.matmul(drop2, weights['w_full_3'])
    print(full3)
    return tf.nn.bias_add(full3, biases['b_full_3'])
  
  # Training computation.
  logits = alex_net(tf_train_dataset, 0.5)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(alex_net(tf_valid_dataset, 1))
  test_prediction = tf.nn.softmax(alex_net(tf_test_dataset, 1))
  
num_steps = 1001

with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as session:
  start = time.clock()
  tf.global_variables_initializer().run()
  run_metadata = tf.RunMetadata()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict, run_metadata=run_metadata)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  print('Time = ', time.clock()-start)
  
  """
  Minibatch loss at step 0: 10.445509
Minibatch accuracy: 9.4%
Validation accuracy: 10.0%
Minibatch loss at step 50: 2.413297
Minibatch accuracy: 6.2%
Validation accuracy: 10.0%
Minibatch loss at step 100: 2.474824
Minibatch accuracy: 3.1%
Validation accuracy: 12.6%
Minibatch loss at step 150: 2.277776
Minibatch accuracy: 15.6%
Validation accuracy: 13.8%
Minibatch loss at step 200: 2.304053
Minibatch accuracy: 6.2%
Validation accuracy: 15.6%
Minibatch loss at step 250: 2.192542
Minibatch accuracy: 21.9%
Validation accuracy: 35.5%
Minibatch loss at step 300: 1.706430
Minibatch accuracy: 40.6%
Validation accuracy: 54.7%
Minibatch loss at step 350: 1.894155
Minibatch accuracy: 43.8%
Validation accuracy: 65.7%
Minibatch loss at step 400: 1.259834
Minibatch accuracy: 59.4%
Validation accuracy: 60.9%
Minibatch loss at step 450: 1.431658
Minibatch accuracy: 56.2%
Validation accuracy: 66.0%
Minibatch loss at step 500: 1.216320
Minibatch accuracy: 46.9%
Validation accuracy: 66.8%
Minibatch loss at step 550: 1.402733
Minibatch accuracy: 50.0%
Validation accuracy: 74.0%
Minibatch loss at step 600: 1.493901
Minibatch accuracy: 53.1%
Validation accuracy: 76.5%
Minibatch loss at step 650: 0.952814
Minibatch accuracy: 68.8%
Validation accuracy: 76.0%
Minibatch loss at step 700: 0.918982
Minibatch accuracy: 62.5%
Validation accuracy: 79.0%
Minibatch loss at step 750: 1.074075
Minibatch accuracy: 62.5%
Validation accuracy: 79.0%
Minibatch loss at step 800: 1.066100
Minibatch accuracy: 56.2%
Validation accuracy: 76.8%
Minibatch loss at step 850: 1.039581
Minibatch accuracy: 65.6%
Validation accuracy: 78.7%
Minibatch loss at step 900: 0.928893
Minibatch accuracy: 81.2%
Validation accuracy: 79.0%
Minibatch loss at step 950: 0.930595
Minibatch accuracy: 78.1%
Validation accuracy: 80.1%
Minibatch loss at step 1000: 0.894674
Minibatch accuracy: 75.0%
Validation accuracy: 81.0%
Test accuracy: 88.0%
Time =  196.2303633969126
  """