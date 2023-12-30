#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


get_ipython().system('pip install efficientnet_pytorch')
get_ipython().system('pip install torchsummary')


# In[3]:


get_ipython().system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')


# In[4]:


import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchsummary import summary
from efficientnet_pytorch import EfficientNet  # Import EfficientNet


# In[5]:


data_dir = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"
diseases = os.listdir(train_dir)


# In[6]:


print(diseases)


# In[7]:


print("Total disease classes are :{}".format(len(diseases)))


# In[8]:


plants = []
NumberOfDiseases = 0 
for plant in diseases:
    if plant.split("__")[0] not in plants:
        plants.append(plant.split("__")[0])
    if plant.split("__")[1] != 'healthy':
        NumberOfDiseases += 1


# In[9]:


print(f"Unique Plants are: \n{plants}")


# In[10]:


print("Number of plants: {}".format(len(plants)))


# In[11]:


print("Number of diseases: {}".format(NumberOfDiseases))


# In[12]:


# Number of images for each disease
nums = {}
for disease in diseases:
    nums[disease] = len(os.listdir(train_dir + '/' + disease))
    
# converting the nums dictionary to pandas dataframe passing index as plant name and number of images as column

img_per_class = pd.DataFrame(nums.values(), index=nums.keys(), columns=["no. of images"])
img_per_class


# In[13]:


index = [n for n in range(38)]
plt.figure(figsize=(20, 5))
plt.bar(index, [n for n in nums.values()], width=0.3)
plt.xlabel('Plants/Diseases', fontsize=10)
plt.ylabel('No of images available', fontsize=10)
plt.xticks(index, diseases, fontsize=5, rotation=90)
plt.title('Images per each class of plant disease')


# In[14]:


n_train = 0
for value in nums.values():
    n_train += value
print(f"There are {n_train} images for training")


# In[15]:


train = ImageFolder(train_dir, transform=transforms.ToTensor())
valid = ImageFolder(valid_dir, transform=transforms.ToTensor())


# In[16]:


img, label = train[0]
print(img.shape, label)


# In[17]:


len(train.classes)


# In[18]:


def show_image(image,label):
    print("label:"+ train.classes[label]+"("+str(label)+")")
    plt.imshow(image.permute(1,2,0))


# In[19]:


show_image(*train[0])


# In[20]:


show_image(*train[70000])


# In[21]:


# Setting the seed value
random_seed = 7
torch.manual_seed(random_seed)


# In[22]:


# setting the batch size
batch_size = 10


# In[23]:


# DataLoaders for training and validation
train_dl = DataLoader(train, batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_dl = DataLoader(valid, batch_size, num_workers=2, pin_memory=True)


# In[24]:


# helper function to show a batch of training instances
def show_batch(data):
    for images, labels in data:
        fig, ax = plt.subplots(figsize=(30, 30))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        break


# In[25]:


# Images for first batch of training
show_batch(train_dl) 


# In[26]:


# for moving data into GPU (if available)
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# for moving data to device (CPU or GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# for loading in the device (GPU if available else CPU)
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)
        
    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[27]:


device = get_default_device()
device


# In[28]:


# Moving data into GPU
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)


# In[29]:


class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x


# In[47]:


from sklearn.metrics import f1_score, recall_score

# base class for the model
class ImageClassificationBase(nn.Module):
    
    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        return acc
    
    def recall(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        recall = torch.tensor(recall_score(labels.cpu(), preds.cpu(), average='weighted', zero_division=1))
        return recall
    
    def f1_score(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        f1 = torch.tensor(f1_score(labels.cpu(), preds.cpu(), average='weighted'))
        return f1
    
    def confusion_matrix(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        cm = confusion_matrix(labels.cpu(), preds.cpu())
        return cm
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = self.accuracy(out, labels)     # Calculate accuracy
        recall = self.recall(out, labels)    # Calculate recall
        f1 = self.f1_score(out, labels)      # Calculate F1 score
        
        return {"val_loss": loss.detach(), "val_accuracy": acc, "val_recall": recall, "val_f1": f1}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        batch_recall = [x["val_recall"] for x in outputs]
        batch_f1 = [x["val_f1"] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss  
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        epoch_recall = torch.stack(batch_recall).mean()
        epoch_f1 = torch.stack(batch_f1).mean()

        return {
            "val_loss": epoch_loss,
            "val_accuracy": epoch_accuracy,
            "val_recall": epoch_recall,
            "val_f1": epoch_f1
        }
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy'], result['val_recall'], result['val_f1']))


# In[48]:


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


# In[49]:


# Define a new class for EfficientNet model
class EfficientNetModel(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

    def forward(self, xb):
        return self.efficientnet(xb)


# In[50]:


# Create an instance of the EfficientNet model
efficientnet_model = to_device(EfficientNetModel(len(train.classes)), device)
efficientnet_model


# In[51]:


# Get a summary of the EfficientNet model
INPUT_SHAPE = (3, 256, 256)
print(summary(efficientnet_model.cuda(), (INPUT_SHAPE)))


# In[56]:


# for training
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def fit_OneCycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0,
                grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # scheduler for one cycle learning rate
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_accuracies = []
        train_recalls = []
        train_f1_scores = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            # recording and updating learning rates
            lrs.append(get_lr(optimizer))
            sched.step()
            
            # Calculate and record training metrics
            train_acc = model.accuracy(model(batch[0]), batch[1])
            train_rec = model.recall(model(batch[0]), batch[1])
            train_f1 = model.f1_score(model(batch[0]), batch[1])
            
            train_accuracies.append(train_acc)
            train_recalls.append(train_rec)
            train_f1_scores.append(train_f1)
            
        # Validation
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_accuracy'] = torch.stack(train_accuracies).mean().item()
        result['train_recall'] = torch.stack(train_recalls).mean().item()
        result['train_f1_score'] = torch.stack(train_f1_scores).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
        
    return history



# In[57]:


get_ipython().run_cell_magic('time', '', 'history = [evaluate(efficientnet_model, valid_dl)]\nhistory\n')


# In[61]:


epochs = 30
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam


# In[62]:


# Train the EfficientNet model
history_efficientnet = [evaluate(efficientnet_model, valid_dl)]
history_efficientnet += fit_OneCycle(epochs, max_lr, efficientnet_model, train_dl, valid_dl,
                                     grad_clip=grad_clip, weight_decay=1e-4, opt_func=opt_func)


# In[6]:


def plot_accuracies(history):
    accuracies = [x['val_accuracy'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');
    
def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');


# In[8]:


model.save()


# In[7]:


import matplotlib.pyplot as plt
import torch

def plot_losses(history):
    train_losses = [x.get('train_loss', None) for x in history]
    val_losses = [x['val_loss'] for x in history if 'val_loss' in x]

    # Filter out None values
    train_losses = [x for x in train_losses if x is not None]

    # Move tensors to CPU
    train_losses = torch.Tensor(train_losses).cpu().numpy()
    val_losses = torch.Tensor(val_losses).cpu().numpy()

    plt.plot(train_losses, '-bx', label='Training loss')
    plt.plot(val_losses, '-rx', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.show()

def plot_accuracies(history):
    train_accs = [x.get('train_accuracy', None) for x in history]
    val_accs = [x['val_accuracy'] for x in history if 'val_accuracy' in x]

    # Filter out None values
    train_accs = [x for x in train_accs if x is not None]

    # Move tensors to CPU
    train_accs = torch.Tensor(train_accs).cpu().numpy()
    val_accs = torch.Tensor(val_accs).cpu().numpy()

    plt.plot(train_accs, '-bx', label='Training accuracy')
    plt.plot(val_accs, '-rx', label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracies')
    plt.show()

# Call the modified functions
plot_losses(history_efficientnet)
plot_accuracies(history_efficientnet)


# In[5]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_metrics(model, data_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, precision, recall, f1

# Calculate metrics on the validation set
accuracy, precision, recall, f1 = get_metrics(model, valid_dl)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Plot the metrics
def plot_metrics(accuracy, precision, recall, f1):
    metrics = [accuracy, precision, recall, f1]
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    plt.figure(figsize=(10, 6))

    for metric, label in zip(metrics, labels):
        plt.plot(metric, '-o', label=label)

    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Metrics over Validation Set')
    plt.legend()
    plt.show()

# Call the function to plot metrics
plot_metrics(accuracy, precision, recall, f1)


# In[ ]:




