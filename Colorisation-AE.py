#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:15:41 2023

@author: ruby
"""

# Reconstructing the grayscale images from galaxy inspector zooniverse to feed as input
# to a covolutional autoencoder to reconstruct the images in colour

import torch
#import torchvision
#import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
#from torchvision.transforms import ToTensor
import os
#from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from skimage.color import rgb2lab, lab2rgb
from torch import unsqueeze, squeeze, permute

# paths to grey and rgb images 
#grey_path = "/Users/ruby/Documents/Zooniverse Images/Grey"
#label_path = "/Users/ruby/Documents/Zooniverse Images/Subset 2"
home = "/Users/ruby/Documents/Zooniverse Images/"

total_images = (len(os.listdir(home+'Subset 2')) - 1)

# take and random sample from the colour images and use 80% of images for training
random_indices = random.sample(list(range(total_images)), total_images)
#we have 79 images so round to the nearest whole number 
train_nums = round(total_images * 0.8)
# split into training and testing
train_indices = random_indices[:train_nums]
test_indices = random_indices[train_nums:]
#len(train_indices), len(test_indices)

# create the dataset from the convolutional autoencoder
class EncoderDataset(Dataset):
    ''' EncoderDataset takes input of training and test indices and creates 
        datasets for training and testing accordingly '''
    def __init__(self, indices, img_dir, transform=None):
        self.img_dir = img_dir
        #self.label_dir = label_dir
        self.transform = transform
        self.img_indices = indices
        self.grey_path = img_dir+'Grey/'
        self.rgb_path = img_dir+'Subset 2/'
        
    def __len__(self):
        return len(self.img_indices)
    
    def __getitem__(self, idx):
        ''' Transformation of the images is done here only '''
        img_name = str(idx)+'.jpg'
        # read image within the grey image folder
        image = read_image(self.grey_path+img_name)
        # turn image into 1D array across dim=0
        image = image.unsqueeze(0)
        # transform image to size (160,160)
        image = F.interpolate(image, (160,160))
        # keep image tensor in the same format
        image = image.squeeze(0)
        # permute the dimensions of image tensor i.e -> [160, 160, 1]
        image = image.permute(1,2,0)
        # repeat the greyscale channel 3 times to create an rgb image 
        image = image.repeat(1,1,3) # tensor.repeat needed??
        # permute the image tensor's dimensions to -> [3, 160, 160]
        image = image.permute(2,0,1)
        # read the rgb image within the rgb image folder i.e the labels
        # we are trying to get the network to predict
        label = read_image(self.rgb_path+img_name)
        # turn the rgb image into a 1D array
        label = label.unsqueeze(0)
        # reshape the rgb image to size (160,160)
        label = F.interpolate(label, (160,160))
        label = label.squeeze(0)
        # permute dimensions of label tensor to -> [160,160,3]
        label = label.permute(1,2,0)
        # and then to [3,160,160]
        label = label.permute(2,0,1)
        # convert the image and label to lab colour space 
        # the output will give L~(0,100), a~(-128,127) and b~(-128,127)
        # divide by 255 to give values between (0,1)
        image = torch.tensor(rgb2lab(image.permute(1,2,0)/255))
        label = torch.tensor(rgb2lab(label.permute(1,2,0)/255))
        # add tensor([0,128,128]) since we have divided the values above by 255
        # then normalise by tensor([100,255,255])
        image = (image + torch.tensor([0, 128, 128])) / torch.tensor([100, 255, 255])
        label = (label + torch.tensor([0, 128, 128])) / torch.tensor([100, 255, 255])
        # change dimensions of image and label so they are in the form [3,128,128]
        image = image.permute(2,0,1)
        label = label.permute(2,0,1)
        # Use the L channel from image (greyscale channel) to predict a, b (colour) channels of label
        image = image[:1,:,:]
        label = label[1:,:,:]
        return image, label

# create the datasets
train_dataset = EncoderDataset(indices=train_indices, img_dir=home)
test_dataset = EncoderDataset(indices=test_indices, img_dir=home)
trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# initialise the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ''' Network takes 1 input channel i.e greyscale channel and outputs
            a and b channels of rgb label.
            Convolution takes place in the encoder step.
            Transpose convolution takes place in the decoder step.
            Outpus from decoder are concatenated with output of the encoder from the same layer.
            Dropout layer is added to decoder only.
            Final layer is a convolutional layer to converge output to two channels.'''
        # encoder layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        # max pool layer not required REMOVE
        self.pool = nn.MaxPool2d(2, 2)
        
        # decoder layers
        self.convt1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1) # gradients of conv2d
        self.convt2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt4 = nn.ConvTranspose2d(in_channels=192, out_channels=15, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.2) # use dropout to randomly zero elements of input tensor - regularisation
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1) #out channels=3?
        
    def forward(self, x):
        # encode
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        # now decode: xd
        xd = F.relu(self.convt1(x4))
        xd = torch.cat((xd,x3), dim=1) # merge output and x3 along columns
        xd = self.dropout(xd)
        xd = F.relu(self.convt2(xd))
        xd = torch.cat((xd,x2), dim=1)
        xd = self.dropout(xd)
        xd = F.relu(self.convt3(xd))
        xd = torch.cat((xd,x1), dim=1)
        xd = self.dropout(xd)
        xd = F.relu(self.convt4(xd))
        xd = torch.cat((xd,x), dim=1)
        output = F.relu(self.conv5(xd))
        return output
    
# state the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialise the model 
model = ConvAutoencoder()
model.to(device)

# initialise hyperparameters
N_EPOCHS = 30
L_RATE = 1e-3

# initialise loss function and optimiser
loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=L_RATE)

# start training
print("Starting training..........")
train_losses = []
test_losses = []

for epoch in range(1, N_EPOCHS+1):
    train_loss = 0
    for sample in tqdm(trainloader):
        images, labels = sample
        images = images.float().to(device)
        labels = labels.float().to(device)
        optimiser.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimiser.step()
        train_loss += loss.item() * images.size(0)
    train_loss = train_loss / len(trainloader)
    train_losses.append(train_loss)
    print("Epoch: {}, Training Loss: {:.4f}".format(epoch, train_loss))
   
    test_loss = 0
    # turn off gradient tracking for testing
    with torch.no_grad():
        model.eval()
        for imgs, labs in testloader:
            imgs, labs = imgs.to(device), labs.to(device)
            output = model(imgs)
            loss = loss_fn(output, labs)
            test_loss += loss.item() * imgs.size(0)
   # model.train()
    test_loss = test_loss / len(testloader)
    #print("Test Loss: {:.4f}".format(test_loss))
    test_losses.append(test_loss)

# plot the training and testing loss
plt.plot(range(1, N_EPOCHS+1), train_losses, label='Train Loss', color='r')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss during Training')
plt.show()
        
  #%%  
# make plots of the grey and predicted images
i = 0
while i < 15:
    sample_img, sample_label = next(iter(trainloader))
    img, label = sample_img[0], sample_label[0]
    fig = plt.figure(figsize=(16,16))
    plt.subplot(441)
    plt.imshow(img.permute(1,2,0), cmap='gray')
    plt.title('Grayscale Image')
    plt.subplot(442)
    plt.imshow(label.permute(1,2,0)[:,:,0], cmap='Greens')
    plt.title('"a" colour channel')
    plt.subplot(443)
    plt.imshow(label.permute(1,2,0)[:,:,1], cmap='Blues')
    plt.title('"b" colour channel')
    plt.subplot(444)
    rgb_image = torch.cat((img, label), dim=0).permute(1,2,0)
    rgb_image = rgb_image * torch.tensor([100,255,255]) - torch.tensor([0,128,128])
    rgb_image = lab2rgb(rgb_image)
    plt.imshow(rgb_image)
    plt.title('Predicted RGB Image')
    plt.show()
    i+=1
  

    
    
    
    
    
    
    
        
        
        
        
        