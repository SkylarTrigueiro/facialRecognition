import torch
import numpy as np
import pandas as pd
from glob import glob
from model.config import config
from collections import Counter
from sklearn.utils import shuffle

import tensorflow as tf
from keras.preprocessing import image

import logging

_logger = logging.getLogger(__name__)

def load_data( ) -> pd.DataFrame:
	
	print( str(config.DATASET_DIR / config.DATA_FILES) )
	_data = glob( str(config.DATASET_DIR / config.DATA_FILES) )
	
	return _data

def to_grayscale(img):
	return img.mean(axis=-1)

def load_img(filepath):
  # load image and downsample
  img = image.img_to_array(image.load_img(filepath, target_size=[config.H, config.W])).astype('uint8')
  return img

def prepare_data(data):

	N = len(data)
	shape = (N,config.H,config.W)
	images = np.zeros(shape)

	# load images as arrays
	for i, f in enumerate(data):
		img = to_grayscale(load_img(f)) / 255.
		images[i] = img

	# make the labels
	# all the filenames are something like 'subject12.happy'
	labels = np.zeros(N)
	for i, f in enumerate(data):
		filename = f.rsplit('\\', 1)[-1]
		subject_num = filename.split('.', 1)[0]    
	
		# subtract 1 since the filenames start from 1
		idx = int( subject_num.replace('subject', '') ) - 1
		labels[i] = idx

	# get the number of subjects
	n_subjects = len(set(labels))

	# let's make it so 3 images for each subject are test data
	# number of test points is then
	n_test = 3*n_subjects
	n_train = N - n_test

	# initialize arrays to hold train and test images
	train_images = np.zeros((n_train, config.H, config.W))
	train_labels = np.zeros(n_train)
	test_images = np.zeros((n_test, config.H, config.W))
	test_labels = np.zeros(n_test)

	count_so_far = {}
	train_idx = 0
	test_idx = 0
	images, labels = shuffle(images, labels, random_state=94019)
	for img, label in zip(images, labels):
		# increment the count
		count_so_far[label] = count_so_far.get(label,0) + 1
	
		if count_so_far[label] > 3:
			# we have already added 3 test images for this subject
			# so add the rest to train
		
			train_images[train_idx] = img
			train_labels[train_idx] = label
			train_idx += 1
		
		else:
			# add the first 3 images to test
			test_images[test_idx] = img
			test_labels[test_idx] = label
			test_idx += 1

	return train_images, train_labels, test_images, test_labels

def prepare_dataloader(train_images, train_labels, test_images, test_labels):

	# create label2idx mapping for easy access
	train_label2idx = {}
	test_label2idx = {}

	for i, label in enumerate(train_labels):
		if label not in train_label2idx:
			train_label2idx[label] = [i]
		else:
			train_label2idx[label].append(i)
		
	for i, label in enumerate(test_labels):
		if label not in test_label2idx:
			test_label2idx[label] = [i]
		else:
			test_label2idx[label].append(i)

	# come up with all possible training sample indices
	train_positives = []
	train_negatives = []
	test_positives = []
	test_negatives = []

	n_train = len(train_labels)
	n_test = len(test_labels)

	for label, indices in train_label2idx.items():
		# all indieces that do NOT belong to this subject
		other_indices = set(range(n_train)) - set(indices)
	
		for i, idx1 in enumerate(indices):
			for idx2 in indices[i+1:]:
				train_positives.append((idx1,idx2))
			
			for idx2 in other_indices:
				train_negatives.append((idx1,idx2))
			
	for label, indices in test_label2idx.items():
		# all indices that do NOT belong to this subject
		other_indices = set(range(n_test)) - set(indices)
	
		for i, idx1 in enumerate(indices):
			for idx2 in indices[i+1:]:
				test_positives.append((idx1,idx2))
			
			for idx2 in other_indices:
				test_negatives.append((idx1,idx2))

	return train_positives, train_negatives, test_positives, test_negatives

def train_generator(batch_size, train_images, train_positives, train_negatives):
	# for each batch, we will send 1 pair of each subject
	# and the same number of non-matching pairs
	n_batches = int(np.ceil(len(train_positives) / batch_size))
	
	while True:
		np.random.shuffle(train_positives)
		n_samples = batch_size * 2
		shape = (n_samples, config.H, config.W)
		x_batch_1 = np.zeros(shape)
		x_batch_2 = np.zeros(shape)
		y_batch = np.zeros(n_samples)
		
		for i in range(n_batches):
			pos_batch_indices = train_positives[ i*batch_size: (i+1)*batch_size]
			
			# fill up x_batch and y_batch
			j = 0
			for idx1, idx2 in pos_batch_indices:
				x_batch_1[j] = train_images[idx1]
				x_batch_2[j] = train_images[idx2]
				y_batch[j] = 1 # match
				j += 1
				
			# get negative samples
			neg_indices = np.random.choice(len(train_negatives), size=len(pos_batch_indices), replace=False)
			for neg in neg_indices:
				idx1, idx2 = train_negatives[neg]
				x_batch_1[j] = train_images[idx1]
				x_batch_2[j] = train_images[idx2]
				y_batch[j] = 0 # non-match
				j += 1
				
			x1 = x_batch_1[:j]
			x2 = x_batch_2[:j]
			y = y_batch[:j]
			
			# reshape
			x1 = x1.reshape(-1, 1, config.H, config.W)
			x2 = x2.reshape(-1, 1, config.H, config.W)
			
			# convert to torch tensor
			x1 = torch.from_numpy(x1).float()
			x2 = torch.from_numpy(x2).float()
			y = torch.from_numpy(y).float()
			
			yield [x1, x2], y

# same thing as the train generator except no shuffling and it uses the test set
def test_generator(batch_size, test_images, test_positives, test_negatives):
	n_batches = int(np.ceil(len(test_positives) / batch_size ))
	
	while True:
		n_samples = batch_size * 2
		shape = (n_samples, config.H, config.W)
		x_batch_1 = np.zeros(shape)
		x_batch_2 = np.zeros(shape)
		y_batch = np.zeros(n_samples)
					
		for i in range(n_batches):
			pos_batch_indices = test_positives[i*batch_size: (i+1)*batch_size]
					
			# fill up x_batch and y_batch
			j = 0
			for idx1, idx2 in pos_batch_indices:
				x_batch_1[j] = test_images[idx1]
				x_batch_2[j] = test_images[idx2]
				y_batch[j] = 1 # match
				j += 1
					
			# get negative samples
			neg_indices = np.random.choice(len(test_negatives), size=len(pos_batch_indices), replace=False)
			for neg in neg_indices:
				idx1, idx2 = test_negatives[neg]
				x_batch_1[j] = test_images[idx1]
				x_batch_2[j] = test_images[idx2]
				y_batch[j] = 0
				j += 1
					
			x1 = x_batch_1[:j]
			x2 = x_batch_2[:j]
			y = y_batch[:j]
			
			# reshape
			x1 = x1.reshape(-1, 1, config.H, config.W)
			x2 = x2.reshape(-1, 1, config.H, config.W)
			
			# convert to torch tensor
			x1 = torch.from_numpy(x1).float()
			x2 = torch.from_numpy(x2).float()
			y = torch.from_numpy(y).float()
			
			yield [x1, x2], y

#if __name__ == '__main__':

	
