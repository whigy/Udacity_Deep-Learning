# -*- coding: utf-8 -*-
"""

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb

"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from assign1_notMNIST import *
import hashlib
import time

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

train_filename = maybe_download('../assign1_notMNIST_data/notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('../assign1_notMNIST_data/notMNIST_small.tar.gz', 8458043)

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

################Problem 1#####################
'''
Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an image of a character A through J rendered in a different font. Display a sample of the images that we just downloaded. Hint: you can use the package IPython.display.
Now let's load the data in a more manageable format. Since, depending on your computer setup you might not be able to fit it all in memory, we'll load each class into a separate dataset, store them on disk and curate them independently. Later we'll merge them into a single dataset of manageable size.
We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road.
A few images might not be readable, we'll just skip them.
'''



def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


################Problem 2#####################
"""
Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. 
Hint: you can use matplotlib.pyplot.

"""

trainApickle = open(train_datasets[0], 'rb')
trainA = pickle.load(trainApickle)

plt.imshow(trainA[0, :, :])

trainApickle.close()

################Problem 3#####################
#Another check: we expect the data to be balanced across classes. Verify that.
def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

# randomize data
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

################Problem 4#####################
# Convince yourself that the data is still good after shuffling!
pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


################Problem 5#####################
"""
By construction, this dataset might contain a lot of overlapping samples, 
including training data that's also contained in the validation and test set! 
Overlap between training and test can skew the results if you expect to use 
your model in an environment where there is never an overlap, but are actually 
ok if you expect to see training samples recur when you use it. 
Measure how much overlap there is between training, validation and test samples.

Optional questions:
What about near duplicates between datasets? (images that are almost identical)
Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
"""
def check_overlaps(images1, images2):
    images1.flags.writeable=False
    images2.flags.writeable=False
    start = time.clock()
    hasht = [hash(image1.tobytes()) for image1 in images1]
    hash1 = set([hash(image1.tobytes()) for image1 in images1])
    hash_raw = [hash(image2.tobytes()) for image2 in images2]
    hash2 = set(hash_raw)
    all_overlaps = set.intersection(hash1, hash2)
    idx_2 = [idx for idx in range(len(hash_raw)) if hash_raw[idx] in hash1]
    return all_overlaps, idx_2, time.clock()-start

r, idx, execTime = check_overlaps(train_dataset, test_dataset)
print("# overlaps between training and test sets:", len(r), "execution time:", execTime)
test_nonoverlap = test_dataset
test_nonoverlap.flags.writeable=True
test_nonoverlap = np.delete(test_nonoverlap, idx, 0)
test_nonoverlap_l = np.delete(test_labels, idx, 0)
print(len(test_nonoverlap), len(test_dataset))

r, idx, execTime = check_overlaps(train_dataset, valid_dataset)   
print("# overlaps between training and validation sets:", len(r), "execution time:", execTime)
valid_nonoverlap = valid_dataset
valid_nonoverlap.flags.writeable=True
valid_nonoverlap = np.delete(valid_nonoverlap, idx, 0)
valid_nonoverlap_l = np.delete(valid_labels, idx, 0)
print(len(valid_nonoverlap), len(valid_dataset))

r, idx, execTime = check_overlaps(valid_dataset, test_dataset) 
print("# overlaps between validation and test sets:", len(r), "execution time:", execTime)


# save the non-overlapping data
pickle_nonoverlap = os.path.join(data_root, 'notMNIST_nonoverlap.pickle')

try:
  f = open(pickle_nonoverlap, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_nonoverlap,
    'valid_labels': valid_nonoverlap_l,
    'test_dataset': test_nonoverlap,
    'test_labels': test_nonoverlap_l,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

################Problem 6#####################
"""
Let's get an idea of what an off-the-shelf classifier can give you on this data. It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.
Train a simple model on this data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression model from sklearn.linear_model.
Optional question: train an off-the-shelf model on all the data!
"""
from sklearn.preprocessing import MultiLabelBinarizer

#Create labels:
#def multilabelFormat(labels):
#    y = labels.reshape((len(labels), 1))
#    return MultiLabelBinarizer().fit_transform(y)
#    
#train_target = multilabelFormat(train_labels)
#valid_target = multilabelFormat(valid_labels)
#valid_target_nonop = multilabelFormat(valid_nonoverlap_l)
#test_target = multilabelFormat(test_labels)
#test_target_nonop = multilabelFormat(test_nonoverlap_l)

def flatData(dataset):
    (samples, width, height) = dataset.shape
    return dataset.reshape((samples, width * height))

num_of_samples = 10000
    
train_data = flatData(train_dataset)
test_data = flatData(test_dataset)
valid_data = flatData(valid_dataset)
test_data_nonop = flatData(test_nonoverlap)
valid_data_nonop = flatData(valid_nonoverlap)

clf = LogisticRegression() #C=1e-3, solver='liblinear', penalty='l2')
clf.fit(train_data[1:num_of_samples, :], train_labels[1:num_of_samples])

test_prediction = clf.predict(test_data[1:num_of_samples, :])

from sklearn.metrics import confusion_matrix
confusion_matrix(test_labels[1:num_of_samples], test_prediction)
