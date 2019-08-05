
#!/usr/bin/env python
# coding: utf-8

# ## First tests with Autonecoder

# In[1]:


import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision


# In[2]:


import torch
import tqdm




import FINAL_PROJECT_dataset_class as fp


# In[5]:




# In[6]:


#dataset.train_images = np.mean(dataset.train_images,axis = 1)[:,None,:,:]


# In[7]:


#dataset.test_images = np.mean(dataset.test_images,axis = 1)[:,None,:,:]
class Autoencoder_near_copy(nn.Module):
    def __init__(self):
        super(Autoencoder_near_copy,self).__init__()
        self.conv1 = nn.Conv2d(1,64,3,2,1)
        self.conv2 = nn.Conv2d(64,128,3,2,1)
        self.batchnorm = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout2d(p=0.2) #ADDED later ! 
        self.conv3 = nn.Conv2d(128,256,3,2,1)
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.Conv2d(256, 128,kernel_size= 3,stride = 1,padding= 1),
                                nn.BatchNorm2d(128),
                                nn.LeakyReLU(True))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.Conv2d(128, 64,kernel_size= 3,stride = 1, padding = 1),
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(True))
        self.conv4 = nn.Conv2d(64,64,3,1,1)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.Conv2d(64, 32,kernel_size= 3,stride = 1, padding =1),
                                nn.BatchNorm2d(32),
                                nn.LeakyReLU(True))
        self.conv5 = nn.Conv2d(32,3,1,1)
    def forward(self,input_,train=True):
        out = self.conv1(input_)
        out = F.relu(out)
        #out = self.maxpool(out)
        #print("conv1",out.shape)
        out = F.relu(self.conv2(out))
        #print('conv2',out.shape)
        out = F.relu(self.conv3(out))
        #print('conv3',out.shape)
        out = self.up1(out)
        #print('up1',out.shape)
        out = self.up2(out)
        out = self.conv4(out)
        #print('up2',out.shape)
        out = self.up3(out)
        #print('up3',out.shape)
        out = self.conv5(out)
        #print('up',out.shape)
        return out



# In[23]:


class Autoencoder_conv(nn.Module):
    def __init__(self):
        super(Autoencoder_conv,self).__init__()
        self.conv1 = nn.Conv2d(1,64,3,2)
        self.conv2 = nn.Conv2d(64,128,3,2)
        self.batchnorm = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,256,3,2)
        self.deconv1 = nn.ConvTranspose2d(256,128,3,2)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128,64,3,2)
        self.deconv3 = nn.ConvTranspose2d(64,3,4,2)
    def forward(self,input_,train=True):
        out = self.conv1(input_)
        out = F.relu(out)
        
        #print("conv1",out.shape)
        out = F.relu(self.conv2(out))
        if train:
            out = self.batchnorm(out)
        
        #print('conv2',out.shape)
        out = F.relu(self.conv3(out))
        #print("conv3",out.shape)
        out = self.deconv1(out)
        out = F.relu(out)
        if train:
            out= self.batchnorm2(out)
        #print("deconv1",out.shape)
        out = F.relu(self.deconv2(out))
        #print("deconv2",out.shape)
        out = self.deconv3(out)
        out = torch.sigmoid(out)
        #print("deconv3",out.shape)
        return out


# In[6]:


class Autoencoder_near(nn.Module):
    def __init__(self):
        super(Autoencoder_near,self).__init__()
        self.conv1 = nn.Conv2d(1,8,3,2,padding=1)
        self.maxpool = nn.MaxPool2d(3,2,padding = 1)
        self.conv2 = nn.Conv2d(8,16,3,2,padding = 1)
        self.conv3 = nn.Conv2d(16,16,3,2,padding = 1)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.Conv2d(16, 16,kernel_size= 3,stride = 1,padding=1),
                                nn.BatchNorm2d(16),
                                nn.ReLU(True))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.Conv2d(16, 8,kernel_size= 3,stride = 1,padding=1),
                                nn.BatchNorm2d(8),
                                nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.Conv2d(8, 3,kernel_size= 3,stride = 1,padding=1),
                            
                                nn.Sigmoid())
    def forward(self,input_):
        out = self.conv1(input_)
        out = F.relu(out)
        #out = self.maxpool(out)
        #print("conv1",out.shape)
        out = F.relu(self.conv2(out))
        #print('conv2',out.shape)
        out = F.relu(self.conv3(out))
        #print('conv3',out.shape)
        out = self.up1(out)
        #print('up',out.shape)
        out = self.up2(out)
        #print('up',out.shape)
        out = self.up3(out)
        #print('up',out.shape)
        return out


# In[9]:


def train(model):
   for epoch in range(EPOCHS):
       running_loss = 0.0
       data_in_epoch = dataset.load_data()

       for batch in data_in_epoch:
           #print('we here')
           #print(batch[0].shape)
           optimizer.zero_grad()
           #X = torch.Tensor(batch).to('cuda')
           #outputs = model(X)
           #cost = loss(outputs,X)
           outputs = model(torch.Tensor(np.mean(batch,axis = 1)[:,None,:,:]).to('cuda'))
           cost = loss(outputs, torch.Tensor(batch).to('cuda'))
           cost.backward()
           optimizer.step()
           running_loss += cost.item()
       print('training loss in epoch {} is : '.format(epoch),running_loss/N_BATCHES_TRAIN)
       train_losses.append(running_loss/N_BATCHES_TRAIN)
       data_in_epoch = dataset.load_data(False)
       for batch in data_in_epoch:
           #print(batch.shape)
           #X = torch.Tensor(batch).to('cuda')
           #outputs = model(X)
           #cost = loss(outputs,X)
           outputs = model(torch.Tensor(np.mean(batch,axis = 1)[:,None,:,:]).to('cuda'),train=False)
           cost = loss(outputs, torch.Tensor(batch).to('cuda'))
           running_loss += cost.item()
       print('test loss in epoch {} is : '.format(epoch),running_loss/N_BATCHES_TEST)
       test_losses.append(running_loss/N_BATCHES_TRAIN)
       if (epoch%100 == 0) and epoch!=0:
           name = 'flowers1/model_BCE_auto_'+str(epoch)
           torch.save(model.state_dict(),name)
           np.save('flowers1/trainlosses_BCE_auto.npy',np.array(train_losses))
           np.save('flowers1/testlosses_BCE_auto.npy',np.array(test_losses))

def train_CIELAB(model):
    for epoch in range(EPOCHS):
        running_loss = 0.0
        data_in_epoch = dataset.load_data()

        for batch in data_in_epoch:
            #print('we here')
            #print(batch[0].shape)
            optimizer.zero_grad()
            #X = torch.Tensor(batch).to('cuda')
            #outputs = model(X)
            #cost = loss(outputs,X)
            outputs = model(torch.Tensor(batch[:,0,:,:][:,None,:,:]).float().to('cuda'))
            cost = loss(outputs, torch.Tensor(batch[:,1:,:,:]).float().to('cuda'))
            cost.backward()
            #print(model.conv1.weight.grad)
            optimizer.step()
            running_loss += cost.item()
        print('training loss in epoch {} is : '.format(epoch),running_loss/N_BATCHES_TRAIN)

        data_in_epoch = dataset.load_data(False)
        for batch in data_in_epoch:
            #print(batch.shape)
            #X = torch.Tensor(batch).to('cuda')
            #outputs = model(X)
            #cost = loss(outputs,X)
            outputs = model(torch.Tensor(batch[:,0,:,:][:,None,:,:]).float().to('cuda'),train = False)
            cost = loss(outputs, torch.Tensor(batch[:,1:,:,:]).float().to('cuda'))
            running_loss += cost.item()
        print('test loss in epoch {} is : '.format(epoch),running_loss/N_BATCHES_TEST)
        if (epoch%100 == 0) and epoch!=0:
            name = 'more_data_cialab/model_BCE_auto'+str(epoch)
            torch.save(model.state_dict(),name)
            np.save('more_data_cialab/trainlosses_BCE_auto.npy',np.array(train_losses))
            np.save('more_data_cialab/testlosses_BCE_auto.npy',np.array(test_losses))



def init_weights(layer):
    torch.nn.init.xavier_uniform_(layer.weight)
    layer.bias.data.fill_(0.01)

# In[10]:
class Autoencoder_conv_dropout(nn.Module):
    def __init__(self):
        super(Autoencoder_conv_dropout,self).__init__()
        self.conv1 = nn.Conv2d(1,64,3,2)
        self.conv2 = nn.Conv2d(64,128,3,2)
        self.batchnorm = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout2d(p=0.1) #ADDED later ! 
        self.conv3 = nn.Conv2d(128,256,3,2)
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.deconv1 = nn.ConvTranspose2d(256,128,3,2)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128,64,3,2)
        self.deconv32 = nn.Conv2d(64,32,3,1,1)
        self.deconv3 = nn.ConvTranspose2d(32,3,4,2)
    def forward(self,input_,train=True):
        out = self.conv1(input_)
        out = F.relu(out)
        
        #print("conv1",out.shape)
        out = F.relu(self.conv2(out))
        
        out = self.batchnorm(out)
        if train:
        #print('conv2',out.shape)
            out = self.dropout(out)
        out = F.relu(self.conv3(out))
        #print("conv3",out.shape)
        out = self.batchnorm1(out)
        out = self.deconv1(out)
        out = F.relu(out)
        
        out= self.batchnorm2(out)
        #print("deconv1",out.shape)
        out = F.relu(self.deconv2(out))
        if train:
            out = self.dropout(out)
        #print("deconv2",out.shape)
        out = torch.relu(self.deconv32(out))
        out = torch.sigmoid(self.deconv3(out))

        #print("deconv3",out.shape)
        return out

class Autoencoder_conv_CIALAB(nn.Module):
    def __init__(self):
        super(Autoencoder_conv_CIALAB,self).__init__()
        self.conv1 = nn.Conv2d(1,64,3,2)
        self.conv2 = nn.Conv2d(64,128,3,2)
        self.batchnorm = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,256,3,2)
        self.deconv1 = nn.ConvTranspose2d(256,128,3,2)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128,64,3,2)
        #self.deconv32 = nn.Conv2d(64,32,3,1,1)
        self.deconv3 = nn.ConvTranspose2d(64,2,4,2)
    def forward(self,input_,train=True):
        out = self.conv1(input_)
        out = F.relu(out)
        
        #print("conv1",out.shape)
        out = F.relu(self.conv2(out))
        if train:
            pass
        out = self.batchnorm(out)
        
        #print('conv2',out.shape)
        out = F.relu(self.conv3(out))
        #print("conv3",out.shape)
        out = self.deconv1(out)
        out = F.relu(out)
        if train:
            pass
        out= self.batchnorm2(out)
        #print("deconv1",out.shape)
        out = F.relu(self.deconv2(out))
        #print("deconv2",out.shape)
        #out = F.relu(self.deconv32(out))
        #print(out.shape)
        out = self.deconv3(out)
        out = torch.sigmoid(out)
        #print("deconv3",out.shape)
        return out

if __name__== "__main__":
    dataset = fp.Dataset(batch_size=64,transpose=True,load_texts=False,aug_factor=2)
    EPOCHS = 1000
    N_BATCHES_TRAIN = (dataset.n_samples-dataset.test_set_size)//dataset.batch_size
    N_BATCHES_TEST = dataset.test_set_size//dataset.batch_size
    train_losses = []
    test_losses  = []

    model = Autoencoder_conv_dropout() #without batch normalization
    init_weights(model.conv1)
    init_weights(model.conv2)
    init_weights(model.conv3)
    init_weights(model.deconv1)
    init_weights(model.deconv2)
    init_weights(model.deconv3)
    init_weights(model.deconv32)
    #model.load_state_dict(torch.load('more_data_drop/model_BCE_auto_1200'))
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.to('cuda')
    train(model)
    np.save('flowers_wa/trainlosses_MSE_auto.npy',np.array(train_losses))
    np.save('flowers_wa/testlosses_MSE_auto.npy',np.array(test_losses))
