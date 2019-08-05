#!/usr/bin/env python
# coding: utf-8

# In[1]:

from FINAL_PROJECT_autoencoder import Autoencoder_conv_dropout
import numpy as np
import matplotlib.pyplot as plt
import torch
#import keras
import os


# In[2]:


import torch.nn as nn
import torch.nn.functional as F


# In[3]:



# In[4]:


import FINAL_PROJECT_dataset_class as fp


# In[5]:


dataset = fp.Dataset(transpose=True,load_texts=True,aug_factor=2,load_embeddings = True)


# In[6]:


NUM_EMBEDDINGS = dataset.embeddings.shape[0]
N_DIM = dataset.embeddings.shape[1]


# In[7]:


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.conv1 = nn.Conv2d(1,64,3,2)
        self.conv2 = nn.Conv2d(64,128,3,2)
        self.batchnorm = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,256,3,2)
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
        return out


# In[8]:


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.deconv1 = nn.ConvTranspose2d(256,128,3,2)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128,64,3,2)
        self.deconv3 = nn.ConvTranspose2d(64,3,4,2)
    def forward(self,input_,train=True):
        out = self.deconv1(input_)
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


# In[9]:



# In[9]:


class Embedding(torch.nn.Module):
    def __init__(self,embedding):
        super(Embedding,self).__init__()
        self.embedding = torch.nn.Embedding(NUM_EMBEDDINGS,N_DIM,_weight=embedding)
        self.embedding.requires_grad = False
        self.lstm = torch.nn.LSTM(N_DIM,128,bidirectional = False,batch_first = True)
    def forward(self,input_text):
        out = self.embedding(input_text)
        
        #print('embedded shape : ',out.shape)
        out = self.lstm(out)
        #print('lstm shape:',out[0].shape)
        #print(out[0][-1])
        return out[1][0].view(out[1][0].shape[1],1,128)


# In[175]:


# In[10]:


class Fusion_layer(torch.nn.Module):
    def __init__(self):
        super(Fusion_layer,self).__init__()
        self.weight = nn.Parameter(torch.autograd.Variable(torch.randn(256,384)))
        self.bias = nn.Parameter(torch.autograd.Variable(torch.zeros(256,)))
        self.batchnorm = torch.nn.BatchNorm2d(256)
    def forward(self, encoder_out, embedding_out):
        batch_size = encoder_out.shape[0]
        self.weight.to('cuda')
        self.bias.to('cuda')
        embedding_out=torch.transpose(embedding_out.repeat((1,49,1)),-1,1).view(batch_size,128,7,7,1)
        out = torch.cat((encoder_out.view(batch_size,256,7,7,1),embedding_out),dim=1)
        #print('concatenated :',out.shape)
        out = out.permute(0,2,3,1,4) # batch,8,8,384,1 
        out = torch.nn.functional.relu(torch.matmul(self.weight.cuda(),out).view(batch_size,7,7,256)+self.bias.cuda())
        #print('output shape',out.shape)
        out = out.permute(0,3,1,2)
        out = self.batchnorm(out)
        return out



# In[11]:


class Colorizer(torch.nn.Module):
    def __init__(self,embeddings):
        super(Colorizer,self).__init__()
        self.encoder = Encoder()
        self.embedding = Embedding(embeddings)
        self.fusion = Fusion_layer()
        self.decoder = Decoder()
    def forward(self,input_image,input_text):
        out = self.encoder(input_image)
        #print("shape at encoder out",out.shape)
        embedding = self.embedding(input_text)
        #print('embedding layer: ',embedding.shape)
        out = self.fusion(out,embedding)
        #print('fusion layer: ',out.shape)
        out = self.decoder(out)
        #print('final output: ',out.shape)
        return out


# In[419]:


EPOCHS = 1500
N_BATCHES_TRAIN = (dataset.n_samples-dataset.test_set_size)//dataset.batch_size
N_BATCHES_TEST = dataset.test_set_size//dataset.batch_size


# In[13]:


def train(model,save = False):
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
           outputs = model(torch.Tensor(np.mean(batch[0],axis = 1)[:,None,:,:]).to('cuda'),torch.LongTensor(batch[1][:,3]).cuda())
           cost = loss(outputs, torch.Tensor(batch[0]).to('cuda'))
           cost.backward()
           optimizer.step()
           running_loss += cost.item()
       print('training loss in epoch {} is : '.format(epoch),running_loss/N_BATCHES_TRAIN)
       train_loss.append(running_loss/N_BATCHES_TRAIN)
       data_in_epoch = dataset.load_data(False)
       for batch in data_in_epoch:
           #print(batch.shape)
           #X = torch.Tensor(batch).to('cuda')
           #outputs = model(X)
           #cost = loss(outputs,X)
           outputs = model(torch.Tensor(np.mean(batch[0],axis = 1)[:,None,:,:]).to('cuda'),torch.LongTensor(batch[1][:,3]).cuda())
           cost = loss(outputs, torch.Tensor(batch[0]).to('cuda'))
           running_loss += cost.item()
       print('test loss in epoch {} is : '.format(epoch),running_loss/N_BATCHES_TEST)
       test_loss.append(running_loss/N_BATCHES_TRAIN)
       if (epoch%100 == 0) and epoch!=0 and save:
           name = 'full_brd/model_full'+str(epoch)
           torch.save(model.state_dict(),name)
           np.save('full_brd/train_loss',np.array(train_loss))
           np.save('full_brd/test_loss',np.array(test_loss))

def init_weights(layer):
    torch.nn.init.xavier_uniform_(layer.weight)
    layer.bias.data.fill_(0.01)
# In[ ]:
test_loss = []
train_loss = []
model = Colorizer(torch.Tensor(dataset.embeddings))
init_weights(model.encoder.conv1)
init_weights(model.encoder.conv2)
init_weights(model.encoder.conv3)
init_weights(model.decoder.deconv1)
init_weights(model.decoder.deconv2)
init_weights(model.decoder.deconv3)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
model.to('cuda')
train(model,save=True)


# In[ ]:





