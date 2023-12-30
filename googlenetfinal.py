#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install efficientnet_pytorch')
get_ipython().system('pip install torchsummary')


# In[ ]:


get_ipython().system('pip install torchsummary')
get_ipython().system('pip install numpy==1.22.0  # Use the version that fits the SciPy requirements')
import numpy
import scipy
get_ipython().system('pip install --upgrade scipy')


# In[ ]:


import os
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
#from torchvision.datasets import ImageFolder
from torchsummary import summary
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns

from torchvision.models import googlenet


# In[ ]:


data_dir = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"
diseases = os.listdir(train_dir)


# In[ ]:


print(diseases)


# In[ ]:


print("Total disease classes are: {}".format(len(diseases)))


# In[ ]:


plants = []
NumberOfDiseases = 0
for plant in diseases:
    if plant.split('___')[0] not in plants:
        plants.append(plant.split('___')[0])
    if plant.split('___')[1] != 'healthy':
        NumberOfDiseases += 1


# In[ ]:


# unique plants in the dataset
print(f"Unique Plants are: \n{plants}")


# In[ ]:


# number of unique plants
print("Number of plants: {}".format(len(plants)))


# In[ ]:


# number of unique diseases
print("Number of diseases: {}".format(NumberOfDiseases))


# In[ ]:


# Number of images for each disease
nums = {}
for disease in diseases:
    nums[disease] = len(os.listdir(train_dir + '/' + disease))
    
# converting the nums dictionary to pandas dataframe passing index as plant name and number of images as column

img_per_class = pd.DataFrame(nums.values(), index=nums.keys(), columns=["no. of images"])
img_per_class


# In[ ]:


# plotting number of images available for each disease
index = [n for n in range(38)]
plt.figure(figsize=(20, 5))
plt.bar(index, [n for n in nums.values()], width=0.3)
plt.xlabel('Plants/Diseases', fontsize=10)
plt.ylabel('No of images available', fontsize=10)
plt.xticks(index, diseases, fontsize=5, rotation=90)
plt.title('Images per each class of plant disease')


# In[ ]:


n_train = 0
for value in nums.values():
    n_train += value
print(f"There are {n_train} images for training")


# In[ ]:


class PlantDiseaseGoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseGoogLeNet, self).__init__()
        self.googlenet = googlenet(pretrained=True, aux_logits=True)
        in_features = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.googlenet(x)

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


# In[ ]:


# datasets for validation and training
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
model = to_device(PlantDiseaseGoogLeNet(len(train.classes)), device)


# In[ ]:


img, label = train[0]
print(img.shape, label)


# In[ ]:


# total number of classes in train set
len(train.classes)


# In[ ]:


# for checking some images from training dataset
def show_image(image, label):
    print("Label :" + train.classes[label] + "(" + str(label) + ")")
    plt.imshow(image.permute(1, 2, 0))


# In[ ]:


show_image(*train[0])


# In[ ]:


show_image(*train[70000])


# In[ ]:


show_image(*train[30000])


# In[ ]:


# Setting the seed value
random_seed = 7
torch.manual_seed(random_seed)


# In[ ]:


# setting the batch size
batch_size = 15


# In[ ]:


# Set up DataLoaders
batch_size = 15
train_dl = DataLoader(train, batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_dl = DataLoader(valid, batch_size, num_workers=2, pin_memory=True)


# In[ ]:


# helper function to show a batch of training instances
def show_batch(data):
    for images, labels in data:
        fig, ax = plt.subplots(figsize=(30, 30))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        break


# In[ ]:


# Images for first batch of training
show_batch(train_dl) 


# In[ ]:


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


# In[ ]:


device = get_default_device()
device


# In[ ]:


import torch.nn as nn
from torchvision.models import googlenet


# In[ ]:


class PlantDiseaseGoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseGoogLeNet, self).__init__()
        self.googlenet = googlenet(pretrained=True, aux_logits=True)
        in_features = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.googlenet(x)

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



# In[ ]:


get_ipython().system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')


# In[ ]:


device = torch.device("cpu")


# In[ ]:


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
    accuracy = accuracy_score(all_labels, all_preds)

    # Plot confusion matrix as heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot precision, recall, and F1 score
    metrics_names = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    metrics_values = [precision, recall, f1, accuracy]

    plt.bar(metrics_names, metrics_values, color=['green', 'orange', 'red', 'blue'])
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Metrics')
    plt.show()

    return {
        'val_loss': torch.stack([x['val_loss'] for x in outputs]).mean().item(),
        'val_accuracy': accuracy,
        'f1_score': f1,
        'recall': recall,
        'precision': precision,
        'confusion_matrix': conf_matrix
    }


# In[ ]:


# Model, Optimizer, and Training Loop
model = to_device(PlantDiseaseGoogLeNet(len(train.classes)), device)


# In[ ]:


import torch
get_ipython().system('pip uninstall torch torchvision torchaudio -y')
get_ipython().system('pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')


# In[ ]:


print("CUDA Available:", torch.cuda.is_available())


# In[ ]:


history = [evaluate_with_metrics(model, valid_dl)]


# In[ ]:





# In[ ]:


epochs = 15
max_lr = 0.001
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam




# In[ ]:


get_ipython().system('pip install torchsummary')
get_ipython().system('pip install numpy==1.22.0  # Use the version that fits the SciPy requirements')
import numpy
import scipy
get_ipython().system('pip install --upgrade scipy')


# In[ ]:


# Import necessary libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import googlenet
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchsummary import summary
import torchsummary

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Set the seed value
random_seed = 7
torch.manual_seed(random_seed)

# Define the data directories
data_dir = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "valid")

# Create data loaders
train = ImageFolder(train_dir, transform=ToTensor())
valid = ImageFolder(valid_dir, transform=ToTensor())
train_dl = DataLoader(train, batch_size=10, shuffle=True, num_workers=2, pin_memory=True)
valid_dl = DataLoader(valid, batch_size=10, num_workers=2, pin_memory=True)

# Helper functions
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def to_device(data, device):
    """Move tensor(s) to the chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a data loader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to the device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()

def googlenet_model(num_classes):
    model = googlenet(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

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
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = self.accuracy(out, labels)
        recall = self.recall(out, labels)
        f1 = self.f1_score(out, labels)

        return {"val_loss": loss.detach(), "val_accuracy": acc, "val_recall": recall, "val_f1": f1}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        batch_recall = [x["val_recall"] for x in outputs]
        batch_f1 = [x["val_f1"] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()
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

class PlantDiseaseGoogleNet(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.googlenet = googlenet_model(num_classes)

    def forward(self, xb):
        return self.googlenet(xb)

def summary(model, input_size):
    """Display model summary"""
    return torchsummary.summary(model, input_size)

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
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
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

            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))
            sched.step()

            train_acc = model.accuracy(model(batch[0]), batch[1])
            train_rec = model.recall(model(batch[0]), batch[1])
            train_f1 = model.f1_score(model(batch[0]), batch[1])

            train_accuracies.append(train_acc)
            train_recalls.append(train_rec)
            train_f1_scores.append(train_f1)

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_accuracy'] = torch.stack(train_accuracies).mean().item()
        result['train_recall'] = torch.stack(train_recalls).mean().item()
        result['train_f1_score'] = torch.stack(train_f1_scores).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)

    return history

# Initialize the GoogleNet-based model
googlenet_model = PlantDiseaseGoogleNet(len(train.classes))
googlenet_model = to_device(googlenet_model, device)

# Move the validation data to the same device as the model
valid_dl = DeviceDataLoader(valid_dl, device)

# Get a summary of the GoogleNet-based model
INPUT_SHAPE = (3, 256, 256)
print(summary(googlenet_model, (INPUT_SHAPE)))

# Train the GoogleNet-based model
epochs = 20
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

# Move the model to the same device as the training data
train_dl = DeviceDataLoader(train_dl, device)

history_googlenet = [evaluate(googlenet_model, valid_dl)]
history_googlenet += fit_OneCycle(epochs, max_lr, googlenet_model, train_dl, valid_dl,
                                  grad_clip=grad_clip, weight_decay=1e-4, opt_func=opt_func)

# Plot training and validation metrics
plot_losses(history_googlenet)
plot_accuracies(history_googlenet)

# Calculate metrics on the validation set
accuracy, precision, recall, f1 = get_metrics(googlenet_model, valid_dl)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Plot the metrics
plot_metrics(accuracy, precision, recall, f1)


# **Validation Accuracy: 99.19%**
# 
# **Validation Recall: 99.19%**
# 
# **Validation F1 Score: 99.56%**
# 

# In[ ]:


def plot_accuracies(history_googlenet):
    accuracies = [x['val_accuracy'] for x in history_googlenet]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');

def plot_losses(history_googlenet):
    train_losses = [x.get('train_loss') for x in history_googlenet]
    val_losses = [x['val_loss'] for x in history_googlenet]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');
    
def plot_lrs(history_googlenet):
    lrs = np.concatenate([x.get('lrs', []) for x in history_googlenet])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');
plot_accuracies(history_googlenet)
plot_metrics(history_googlenet)



# In[ ]:


# Saving the model
#PATH = './plant_disease_modelgooglnet.pth'
#torch.save(model.state_dict(), PATH)



# In[ ]:


plot_losses(history_googlenet)


# In[ ]:


#test_images = sorted(os.listdir(valid_dir + '/valid')) # since images in test folder are in alphabetical order
#test_images


# In[ ]:




