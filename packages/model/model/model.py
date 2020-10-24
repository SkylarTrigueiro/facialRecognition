import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from datetime import datetime

# build the model
class SiameseNN(nn.Module):
    def __init__(self,feature_dim):
        super(SiameseNN, self).__init__()
        
        # define CNN featurizer
        self.cnn = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Flatten(),  # 32 > 16 > 8 > 4
            nn.Linear(15*20*128,256), # 120 > 60 > 30 > 15, 160>80>40>20
            #nn.Linear(15*20*64,128), # 120 > 60 > 30 > 15, 160>80>40>20
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )
        
    def forward(self, im1, im2):
        feat1 = self.cnn(im1)
        feat2 = self.cnn(im2)
        
        # Euclidean distance between feature 1 and feature 2
        return torch.norm(feat1 - feat2, dim=-1)