import numpy as np
import torch

from model.processing.data_management import load_data, prepare_data, prepare_dataloader, train_generator, test_generator
from model.model import SiameseNN, batch_gd, contrastive_loss
from model.config import config
from model import __version__ as _version

import logging

_logger = logging.getLogger(__name__)


def run_training():
	"""Train the model"""
	
	#read training data
	data = load_data()
	train_images, train_labels, test_images, test_labels = prepare_data(data)
	train_positives, train_negatives, test_positives, test_negatives = prepare_dataloader(train_images, train_labels, test_images, test_labels)
	model = SiameseNN(config.FEATURE_DIM)

	train_steps = int(np.ceil(len(train_positives) / config.BATCH_SIZE))
	test_steps = int(np.ceil(len(test_positives) / config.BATCH_SIZE))

	# Loss and optimizer
	optimizer = torch.optim.Adam(model.parameters())

	train_losses, test_losses = batch_gd(
		model,
		contrastive_loss,
		optimizer,
		train_generator(config.BATCH_SIZE, train_images, train_positives, train_negatives),
		test_generator(config.BATCH_SIZE, test_images, test_positives, test_negatives),
		train_steps,
		test_steps,
		config.EPOCHS)

	return train_losses, test_losses

def continue_training(model):

	#read training data
	data = load_data()
	train_images, train_labels, test_images, test_labels = prepare_data(data)
	train_positives, train_negatives, test_positives, test_negatives = prepare_dataloader(train_images, train_labels, test_images, test_labels)

	train_steps = int(np.ceil(len(train_positives) / config.BATCH_SIZE))
	test_steps = int(np.ceil(len(test_positives) / config.BATCH_SIZE))

	# Loss and optimizer
	optimizer = torch.optim.Adam(model.parameters())

	train_losses, test_losses = batch_gd(
		model,
		contrastive_loss,
		optimizer,
		train_generator(config.BATCH_SIZE, train_images, train_positives, train_negatives),
		test_generator(config.BATCH_SIZE, test_images, test_positives, test_negatives),
		train_steps,
		test_steps,
		config.EPOCHS)

	return model, train_losses, test_losses
	
if __name__ == '__main__':
	run_training(save_result=True)