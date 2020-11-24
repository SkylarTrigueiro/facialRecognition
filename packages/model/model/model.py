import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from model.config import config

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

# loss function for siamese network
def contrastive_loss(y,t):
    nonmatch = F.relu(1-y) # max(margin-y,0)
    return torch.mean( t*y**2 + (1-t)*nonmatch**2)

# A function to encapsulate the training loop
def batch_gd(
    model,
    criterion,
    optimizer,
    train_gen,
    test_gen,
    train_steps_per_epoch,
    test_steps_per_epoch,
    epochs):
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    min_test_loss = float('inf')
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    
    for it in range(epochs):
        model.train()
        t0 = datetime.now()
        train_loss = []
        steps = 0
        for (x1,x2), targets in train_gen:
            # mode data to GPU
            x1, x2, targets = x1.to(device), x2.to(device), targets.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x1, x2)
            loss = criterion(outputs, targets)
            
            # Backward and optimizer
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            
            # Update steps
            steps += 1
            if steps >= train_steps_per_epoch:
                break
                
        # Get train loss and test loss
        train_loss = np.mean(train_loss) # a little misleading
            
        model.eval()
        test_loss = []
        steps = 0
        for(x1, x2), targets in test_gen:
                
            x1, x2, targets = x1.to(device), x2.to(device), targets.to(device)
            outputs = model(x1, x2)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
            steps += 1
            if steps >= train_steps_per_epoch:
                break
        test_loss = np.mean(test_loss)
        
        if( test_loss < min_test_loss):
            # Save the model
            torch.save(model.state_dict(), config.TRAINED_MODEL_DIR / config.MODEL_NAME)
            rv_train_losses = train_losses
            rv_test_losses = test_losses
            
        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss
            
        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
        Test Loss: {test_loss:.4f}, Duration: {dt}')
            
    return rv_train_losses, rv_test_losses