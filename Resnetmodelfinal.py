#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install torchsummary')


# In[4]:


# Importing necessary libraries
import os
get_ipython().system('pip install numpy==1.22.0  # Use the version that fits the SciPy requirements')
get_ipython().system('pip install --upgrade scipy')
#import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchsummary import summary
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


# In[7]:


data_dir = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"
diseases = os.listdir(train_dir)


# In[8]:


print(diseases)


# In[9]:


print("Total disease classes are :{}".format(len(diseases)))


# In[10]:


plants = []
NumberOfDiseases = 0 
for plant in diseases:
    if plant.split("__")[0] not in plants:
        plants.append(plant.split("__")[0])
    if plant.split("__")[1] != 'healthy':
        NumberOfDiseases += 1


# In[11]:


print(f"Unique Plants are: \n{plants}")


# In[12]:


print("Number of plants: {}".format(len(plants)))


# In[13]:


print("Number of diseases: {}".format(NumberOfDiseases))


# In[14]:


# Number of images for each disease
nums = {}
for disease in diseases:
    nums[disease] = len(os.listdir(train_dir + '/' + disease))
    
# converting the nums dictionary to pandas dataframe passing index as plant name and number of images as column

img_per_class = pd.DataFrame(nums.values(), index=nums.keys(), columns=["no. of images"])
img_per_class


# In[15]:


index = [n for n in range(38)]
plt.figure(figsize=(20, 5))
plt.bar(index, [n for n in nums.values()], width=0.3)
plt.xlabel('Plants/Diseases', fontsize=10)
plt.ylabel('No of images available', fontsize=10)
plt.xticks(index, diseases, fontsize=5, rotation=90)
plt.title('Images per each class of plant disease')


# In[16]:


n_train = 0
for value in nums.values():
    n_train += value
print(f"There are {n_train} images for training")


# In[17]:


train = ImageFolder(train_dir, transform=transforms.ToTensor())
valid = ImageFolder(valid_dir, transform=transforms.ToTensor())


# In[18]:


img, label = train[0]
print(img.shape, label)


# In[19]:


len(train.classes)


# In[20]:


def show_image(image,label):
    print("label:"+ train.classes[label]+"("+str(label)+")")
    plt.imshow(image.permute(1,2,0))


# In[21]:


show_image(*train[0])


# In[22]:


show_image(*train[70000])


# In[23]:


# Checking if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[24]:


# Setting the seed value
random_seed = 7
torch.manual_seed(random_seed)


# In[25]:


# setting the batch size
batch_size = 15


# In[26]:


# DataLoaders for training and validation
train_dl = DataLoader(train, batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_dl = DataLoader(valid, batch_size, num_workers=2, pin_memory=True)


# In[27]:


# helper function to show a batch of training instances
def show_batch(data):
    for images, labels in data:
        fig, ax = plt.subplots(figsize=(30, 30))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        break


# In[28]:


# Images for first batch of training
show_batch(train_dl) 


# In[60]:


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


# In[61]:


device = get_default_device()
device


# In[62]:


# Moving data into GPU
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)


# In[63]:


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


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        preds = torch.argmax(out, dim=1)
        return {'val_loss': loss.detach(), 'val_accuracy': acc, 'preds': preds.tolist(), 'labels': labels.tolist()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {'val_loss': epoch_loss, 'val_accuracy': epoch_accuracy}

    def epoch_end(self, epoch, result):
        if 'val_loss' in result:
            print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, "
                  "F1 Score: {:.4f}, Recall: {:.4f}, Precision: {:.4f}".format(
                epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy'],
                result['f1_score'], result['recall'], result['precision']))
            print("Confusion Matrix:")
            print(result['confusion_matrix'])
        else:
            print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}".format(
                epoch, result['lrs'][-1], result['train_loss']))


# In[64]:


# resnet architecture 
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                         nn.Flatten(),
                                         nn.Linear(512, num_diseases))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))



# In[65]:


# defining the model and moving it to the GPU
model = to_device(ResNet9(3, len(train.classes)), device) 
model


# In[66]:


# getting summary of the model
INPUT_SHAPE = (3, 256, 256)
print(summary(model.cuda(), (INPUT_SHAPE)))


# In[109]:


from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import seaborn as sns
import matplotlib.pyplot as plt
#for training

@torch.no_grad()
def evaluate_with_metrics(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    outputs = [model.validation_step(batch) for batch in val_loader]

    for output in outputs:
        all_preds.extend(output['preds'])
        all_labels.extend(output['labels'])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')

    # Plot confusion matrix as heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    return {
        'val_loss': torch.stack([x['val_loss'] for x in outputs]).mean().item(),
        'val_accuracy': torch.stack([x['val_accuracy'] for x in outputs]).mean().item(),
        'f1_score': f1,
        'recall': recall,
        'precision': precision,
        'confusion_matrix': conf_matrix
    }
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [accuracy_score(all_labels, all_preds),
                      precision, recall, f1]

    plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red'])
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Model Evaluation Metrics')
    plt.show()



# In[110]:


def fit_OneCycle_with_metrics(epochs, max_lr, model, train_loader, val_loader, weight_decay=0,
                              grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))
            sched.step()

        result = evaluate_with_metrics(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)

    return history


# In[111]:


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


# In[112]:


#precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)


# In[113]:


# Evaluating the model before training

history = [evaluate_with_metrics(model, valid_dl)]
history


# In[ ]:


# Training and evaluating the model
epochs = 15
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam


# In[116]:


#training the model
history += fit_OneCycle_with_metrics(epochs, max_lr, model, train_dl, valid_dl,
                                      grad_clip=grad_clip, weight_decay=weight_decay,
                                      opt_func=opt_func)
# Saving the model
PATH = './plant_disease_model1.pth'
torch.save(model.state_dict(), PATH)


# In[117]:


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
plot_accuracies(history)
plot_metrics(history)


# In[120]:


metrics_names = ['Accuracy', 'Recall', 'F1 Score']
metrics_values = [accuracy, recall, f1]

plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Model Evaluation Metrics')
plt.show()


# In[ ]:


# Saving the model
PATH = './plant_disease_model1.pth'
torch.save(model.state_dict(), PATH)


# In[118]:


plot_accuracies(history)
plot_metrics(history)


# In[85]:


# Saving the model
PATH = './plant_disease_model1.pth'
torch.save(model.state_dict(), PATH)


# In[96]:


import seaborn as sns


# In[121]:


sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

return {
    'val_loss': torch.stack([x['val_loss'] for x in outputs]).mean().item(),
    'val_accuracy': torch.stack([x['val_accuracy'] for x in outputs]).mean().item(),
    'f1_score': f1,
    'recall': recall,
    'precision': precision,
    'confusion_matrix': conf_matrix
}


# In[122]:


plot_losses(history)


# In[123]:


test_dir = "../input/new-plant-diseases-dataset/test"
test = ImageFolder(test_dir, transform=transforms.ToTensor())


# In[124]:


test_images = sorted(os.listdir(test_dir + '/test')) # since images in test folder are in alphabetical order
test_images


# In[125]:


def predict_image(img, model):
    """Converts image to array and return the predicted class
        with highest probability"""
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label

    return train.classes[preds[0].item()]


# In[126]:


# predicting first image
img, label = test[0]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[0], ', Predicted:', predict_image(img, model))


# In[127]:


# getting all predictions (actual label vs predicted)
for i, (img, label) in enumerate(test):
    print('Label:', test_images[i], ', Predicted:', predict_image(img, model))


# In[130]:


# saving to the kaggle working directory
PATH = './plant-disease-model-adnan-shaikh1.pth'  
torch.save(model.state_dict(), PATH)


# In[129]:


# saving the entire model to working directory
PATH = './plant-disease-model-complete.pth'
torch.save(model, PATH)


# In[ ]:




