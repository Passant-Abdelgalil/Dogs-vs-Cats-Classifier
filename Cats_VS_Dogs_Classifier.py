# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:33:11 2020

@author: Basant
"""

# Important Librares

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
from torch.utils.data import DataLoader
import cv2
import glob
import os
from PIL import Image
import skimage.io as io
from sklearn.model_selection import train_test_split

#              ----------------------------------------------------
## variables

image_size = 128
get_data = False # put it True at the first time
batch_size = 64
gray_scale = True
cuda = False
#              ----------------------------------------------------
## loading data

if get_data: 
    if gray_scale:
        train_images = np.array([cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY), (image_size, image_size))
                             for image in glob.glob('train/*.jpg')])
        test_images  = np.array([cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY), (image_size, image_size))
                             for image in glob.glob('test1/*.jpg')])
    else:
        train_images = np.array([cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB), (image_size, image_size))
                             for image in glob.glob('train/*.jpg')])
        test_images  = np.array([cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB), (image_size, image_size))
                             for image in glob.glob('test1/*.jpg')])
    
    np.save("train_data0.npy", train_images)
    np.save("test_data0.npy", test_images)
else:
    train_images = np.load("train_data0.npy")
    test_images = np.load("test_data0.npy")


train_tensor = torch.from_numpy(train_images)
test_tensor = torch.from_numpy(test_images)
train_labels = torch.cat((torch.zeros((12500, 1)), torch.ones((12500, 1))), axis=0)

#              ----------------------------------------------------
# Data splitting

X_train, X_test, y_train, y_test = train_test_split(train_tensor, train_labels, test_size=.2, random_state=0, shuffle = True)

#              ----------------------------------------------------
# GPU Cells

torch.cuda.is_available()
device = torch.device("cuda:0")
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
    cuda = True
else:
    device = torch.device("cpu")
    print("Running on the CPU")
    cuda = False
    
#              ----------------------------------------------------
# defining a custom dataset

class Dogs_and_Cats():
    def __init__(self, data, labels):
        self.samples = data
        self.labels = labels
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


trainset = Dogs_and_Cats(X_train, y_train)
testset = Dogs_and_Cats(X_test, y_test)


#              ----------------------------------------------------
# Prepare the dataset to be loaded into the model

params = {'batch_size': 64,
          'shuffle': False, }

train_loader = DataLoader(trainset, **params)
test_loader = DataLoader(testset, **params)

#              ----------------------------------------------------
#Create the model

starting_layer = 3
if gray_scale: 
    starting_layer = 1
else:
    starting_layer = 3

class DogsVsCatsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(starting_layer, 32, 3), # 32 * 126 * 126
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output size: 32 * 63 * 63
            
            nn.Conv2d(32, 64, 3), # output size:  64 * 61 * 61
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #output size: 64 * 30 * 30
            
            nn.Conv2d(64, 128, 3), # output size: 128 * 28 * 28
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output size: 128 * 14 * 14
            
            nn.Conv2d(128, 256, 3), # output size: 256 * 12 * 12
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output size: 256 * 6 * 6
            
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 1000),
            nn.ReLU(),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
    
    def forward(self, input_batch):
        return torch.sigmoid(self.model(input_batch))
    
    
model = DogsVsCatsModel()
if cuda:        
    model = model.to(device)
    
#              ----------------------------------------------------
# Training the model

optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_function = nn.MSELoss()

epochs = 15
counter = 1
for epoch in range(epochs):
    for x, y in train_loader:
        x = x.float().reshape(-1, 1, 128, 128)
        y = y.float()
        
        if cuda:
            x, y = x.to(device), y.to(device)
        
        model.zero_grad()
        output = model(x)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
        
        if counter % 10 == 0:
            print(f"batch num {counter} from {int(len(trainset)/batch_size)} with loss: {loss}")
        counter += 1
    print(f"epoch num {epoch} from {epochs} with loss: {loss}")
    counter = 1
    
#              ----------------------------------------------------
# Testing the model on the X_test from the split function
    
if cuda:
    X_test, y_test = X_test.to(device), y_test.to(device)

correct = 0
total = 0
with torch.no_grad():
    for i in range(len(X_test)):
        real_class = y_test[i]
        x = X_test[i].float().view(-1, 1, 128, 128)
        net_out = model(x)  # returns a list, 
        predicted_class = (net_out>=0.5)
        if predicted_class == real_class:
            correct += 1
        total += 1
        if total % 100 == 0:
            print(f"{total} from {len(X_test)}")
        
print("Accuracy: ", round(correct/total, 3))

