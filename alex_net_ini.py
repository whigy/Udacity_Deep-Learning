# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:55:31 2017

@author: HWAG
"""

#from import_images import a, b, c, d
import tensorflow as tf

# initialize weight and bias
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


# build Alex net
def alex_net(train_data, weights, biases, dropout):

#    stride = [1, 4, 4, 1]
#    conv1 = tf.nn.conv2d(train_data, weights['w_conv_1'], stride, padding = 'VALID', name = 'conv1')
#    print(conv1)
#    bias = tf.nn.bias_add(conv1, biases['b_conv_1'])
#    relu1 = tf.nn.relu(bias, name = 'relu1')
#    norm1 = tf.nn.lrn(relu1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
#    #norm1 = tf.nn.lrn(conv1, 4, bias=2.0, alpha=0.0001, beta=0.75, name='norm1') # Alex's parameter
#    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
#    print(pool1)

    # second layer
    stride = [1, 1, 1, 1]
    conv2 = tf.nn.conv2d(train_data, weights['w_conv_2'], stride, padding = 'SAME', name = 'conv2')
    print(conv2)
    bias = tf.nn.bias_add(conv2, biases['b_conv_2'])
    relu2 = tf.nn.relu(bias, name = 'relu2')
    norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    #norm2 = tf.nn.lrn(conv1, 4, bias=2.0, alpha=0.0001, beta=0.75, name='norm1') # Alex's parameter
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
    dim = weights['w_full_1'].get_shape().as_list()[0]
    reshape1 = tf.reshape(pool5, [-1, dim]) # Reshape output to fit fc layer input
    print(reshape1)
    full1 = tf.matmul(reshape1, weights['w_full_1'])
    print(full1)
    relu_f1 = tf.nn.relu(full1 + biases['b_full_1'], name='fc1')
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
    out = tf.nn.bias_add(full3, biases['b_full_3'])

    # softmax prediction
    softmax = tf.add(tf.matmul(out, weights['w_soft']), biases['b_soft'], name='softmax')
    print(softmax)
    return softmax


## initial Alex Net
def para_init(n_class):
    weights = {
        #'w_conv_1':_variable_with_weight_decay('weights', shape=[11, 11, 3, 96], stddev=1e-4, wd=0.0),  #weight_decay = 0.0005
        'w_conv_2':_variable_with_weight_decay('weights', shape=[5, 5, 96, 256], stddev=1e-4, wd=0.0),  #weight_decay = 0.0005
        'w_conv_3':_variable_with_weight_decay('weights', shape=[3, 3, 256, 384], stddev=1e-4, wd=0.0),  #weight_decay = 0.0005
        'w_conv_4':_variable_with_weight_decay('weights', shape=[3, 3, 384, 384], stddev=1e-4, wd=0.0),  #weight_decay = 0.0005
        'w_conv_5':_variable_with_weight_decay('weights', shape=[3, 3, 384, 256], stddev=1e-4, wd=0.0), #weight_decay = 0.0005
        'w_full_1':_variable_with_weight_decay('weights', shape=[6*6*256, 4096], stddev=1e-4, wd=0.0),  #weight_decay = 0.0005
        'w_full_2':_variable_with_weight_decay('weights', shape=[4096, 4096], stddev=1e-4, wd=0.0),  #weight_decay = 0.0005
        'w_full_3':_variable_with_weight_decay('weights', shape=[4096, 1000], stddev=1e-4, wd=0.0),  #weight_decay = 0.0005
        'w_soft':_variable_with_weight_decay('weights', shape=[1000, n_class], stddev=1e-4, wd=0.0)
    }
    biases = {
        #'b_conv_1': tf.Variable(tf.random_normal([96])),
        'b_conv_2': tf.Variable(tf.random_normal([256])),
        'b_conv_3': tf.Variable(tf.random_normal([384])),
        'b_conv_4': tf.Variable(tf.random_normal([384])),
        'b_conv_5': tf.Variable(tf.random_normal([256])),
        'b_full_1': tf.Variable(tf.random_normal([4096])),
        'b_full_2': tf.Variable(tf.random_normal([4096])),
        'b_full_3': tf.Variable(tf.random_normal([1000])),
        'b_soft': tf.Variable(tf.random_normal([n_class]))
    }
    return weights, biases
