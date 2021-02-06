# Facial Recognition (Work in progress)

In this project, I will be trying to determine if two grey-scale mugshot style photos are of the same person or not. To accomplish this, I will be training a Siamese network using a convolutional network architecture which ultimately resulted in a model with $95.5\%$ sensitivity and $97.3\%$ specificity.


## Data

The images I used to train this model were sourced from the "Yale Face Databse" which can be found at http://vision.ucsd.edu/content/yale-face-database. The data set contains 165 images of 15 individuals with 11 photos taken of each person. For each of the 11 photos under different lighting conditions, different facial expressions, and with or without glasses, etc.

Since we are focused on pairs of images, the size of our dataset is actually (165 choose 2)  = 13530 of which 15\*(11 choose 2 ) = 825 are positive matches. Therefore the data is largely unbalanced with only 6\% being a positive match. For this reason, I will report sensitivity and specificity rather than accuracy.   

## Training

I divided my training using roughly a 70/30 split. To ensure that each individual person had representation in the training and testing set, I designed the split to take 3 photos taken at random from each individual's set of photos.

After the split in my training set will contain have (135 choose 2) = 9045 pairs of which 15\*(8 choose 2) = 420 are positive matches.

## References

1. This project started as a project in the Udemy course "Pytorch: Deep Learning and Artificial Intelligence", which I highly recommend.

2. The data can be found at http://vision.ucsd.edu/content/yale-face-database