import numpy as np
import torch
from model.config import config
import matplotlib.pyplot as plt

# Convenience function to make predictions
def predict(model, device, x1, x2):

	x1 = torch.from_numpy(x1).float().to(device)
	x2 = torch.from_numpy(x2).float().to(device)
	with torch.no_grad():
		dist = model(x1, x2).cpu().numpy()
		return dist.flatten()

# calculate accuracy before training
# since the dataset is imbalanced, we'll report tp, tn, fp, fn
def get_train_accuracy(model, train_images, train_positives, train_negatives, threshold=0.85):
	positive_distances = []
	negative_distances = []

	tp = 0
	tn = 0
	fp = 0
	fn = 0

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)

	batch_size = config.BATCH_SIZE
	x_batch_1 = np.zeros((batch_size, 1, config.H, config.W))
	x_batch_2 = np.zeros((batch_size, 1, config.H, config.W))
	n_batches = int(np.ceil(len(train_positives) / batch_size))
	for i in range(n_batches):
		pos_batch_indices = train_positives[i * batch_size: (i + 1) * batch_size]

		# fill up x_batch and y_batch
		j = 0
		for idx1, idx2 in pos_batch_indices:
			x_batch_1[j,0] = train_images[idx1]
			x_batch_2[j,0] = train_images[idx2]
			j += 1

		x1 = x_batch_1[:j]
		x2 = x_batch_2[:j]
		distances = predict(model, device, x1, x2)
		positive_distances += distances.tolist()

		# update tp, tn, fp, fn
		tp += (distances < threshold).sum()
		fn += (distances > threshold).sum()

	n_batches = int(np.ceil(len(train_negatives) / batch_size))
	for i in range(n_batches):
		neg_batch_indices = train_negatives[i * batch_size: (i + 1) * batch_size]

		# fill up x_batch and y_batch
		j = 0
		for idx1, idx2 in neg_batch_indices:
			x_batch_1[j,0] = train_images[idx1]
			x_batch_2[j,0] = train_images[idx2]
			j += 1

		x1 = x_batch_1[:j]
		x2 = x_batch_2[:j]
		distances = predict(model, device, x1, x2)
		negative_distances += distances.tolist()

		# update tp, tn, fp, fn
		fp += (distances < threshold).sum()
		tn += (distances > threshold).sum()

	tpr = tp / (tp + fn)
	tnr = tn / (tn + fp)
	print(len(positive_distances))
	print(len(negative_distances))
	print(f"sensitivity (tpr): {tpr}, specificity (tnr): {tnr}")

	plt.hist(negative_distances, bins=20, density=True, label='negative_distances')
	plt.hist(positive_distances, bins=20, density=True, label='positive_distances')
	plt.legend()
	plt.show()

def get_test_accuracy(model, test_images, test_positives, test_negatives, threshold=0.85):
	positive_distances = []
	negative_distances = []

	tp = 0
	tn = 0
	fp = 0
	fn = 0

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)

	batch_size = config.BATCH_SIZE
	x_batch_1 = np.zeros((batch_size, 1, config.H, config.W))
	x_batch_2 = np.zeros((batch_size, 1,config.H, config.W))
	n_batches = int(np.ceil(len(test_positives) / batch_size))
	for i in range(n_batches):
		pos_batch_indices = test_positives[i * batch_size: (i + 1) * batch_size]

		# fill up x_batch and y_batch
		j = 0
		for idx1, idx2 in pos_batch_indices:
			x_batch_1[j,0] = test_images[idx1]
			x_batch_2[j,0] = test_images[idx2]
			j += 1

		x1 = x_batch_1[:j]
		x2 = x_batch_2[:j]
		distances = predict(model, device, x1, x2)
		positive_distances += distances.tolist()

		# update tp, tn, fp, fn
		tp += (distances < threshold).sum()
		fn += (distances > threshold).sum()

	n_batches = int(np.ceil(len(test_negatives) / batch_size))
	for i in range(n_batches):
		neg_batch_indices = test_negatives[i * batch_size: (i + 1) * batch_size]

		# fill up x_batch and y_batch
		j = 0
		for idx1, idx2 in neg_batch_indices:
			x_batch_1[j] = test_images[idx1]
			x_batch_2[j] = test_images[idx2]
			j += 1

		x1 = x_batch_1[:j]
		x2 = x_batch_2[:j]
		distances = predict(model, device, x1, x2)
		negative_distances += distances.tolist()

		# update tp, tn, fp, fn
		fp += (distances < threshold).sum()
		tn += (distances > threshold).sum()


	tpr = tp / (tp + fn)
	tnr = tn / (tn + fp)
	print(len(positive_distances))
	print(len(negative_distances))
	print(f"sensitivity (tpr): {tpr}, specificity (tnr): {tnr}")

	plt.hist(negative_distances, bins=20, density=True, label='negative_distances')
	plt.hist(positive_distances, bins=20, density=True, label='positive_distances')
	plt.legend()
	plt.show()