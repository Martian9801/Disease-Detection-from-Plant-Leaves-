#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install torchsummary')


# In[6]:


pip install timm


# In[7]:


# Import necessary libraries
import timm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import xception
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
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

class PlantDiseaseXception(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.xception = timm.create_model('xception', pretrained=True)
        in_features = self.xception.fc.in_features
        self.xception.fc = nn.Linear(in_features, num_classes)

    def forward(self, xb):
        return self.xception(xb)

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
        result['train_recall'] = torch.stack(train_recalls).mean().item


# In[8]:


def plot_metrics(history):
    train_losses = [x['train_loss'] for x in history]
    val_losses = [x['val_loss'] for x in history]
    train_accuracies = [x['train_accuracy'] for x in history]
    val_accuracies = [x['val_accuracy'] for x in history]
    train_recalls = [x['train_recall'] for x in history]
    val_recalls = [x['val_recall'] for x in history]
    train_f1_scores = [x['train_f1_score'] for x in history]
    val_f1_scores = [x['val_f1'] for x in history]

    # Plotting training and validation losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.legend()
    plt.title('Training and Validation Losses')

    # Plotting training and validation accuracies
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='train_accuracy')
    plt.plot(val_accuracies, label='val_accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracies')

    # Plotting training and validation recalls
    plt.subplot(1, 3, 3)
    plt.plot(train_recalls, label='train_recall')
    plt.plot(val_recalls, label='val_recall')
    plt.legend()
    plt.title('Training and Validation Recalls')

    # Display the plots
    plt.show()


# In[9]:


epochs = 20
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam


# In[11]:


get_ipython().system(' pip install torchsummary')


# In[ ]:


# Import necessary libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import timm
from torchsummary import summary as torch_summary  # Fix import
from sklearn.metrics import recall_score, f1_score, confusion_matrix
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

class PlantDiseaseXception(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.xception = timm.create_model('xception', pretrained=True)
        in_features = self.xception.fc.in_features
        self.xception.fc = nn.Linear(in_features, num_classes)

    def forward(self, xb):
        return self.xception(xb)

def summary(model, input_size):
    """Display model summary"""
    return torch_summary(model, input_size)  # Fix function call

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
        result['train_f1'] = torch.stack(train_f1_scores).mean().item()

        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)

    return history

def plot_metrics(history):
    train_losses = [x['train_loss'] for x in history]
    val_losses = [x['val_loss'] for x in history]
    train_accuracies = [x['train_accuracy'] for x in history]
    val_accuracies = [x['val_accuracy'] for x in history]
    train_recalls = [x['train_recall'] for x in history]
    val_recalls = [x['val_recall'] for x in history]
    train_f1_scores = [x['train_f1'] for x in history]
    val_f1_scores = [x['val_f1'] for x in history]

    # Plotting training and validation losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.legend()
    plt.title('Training and Validation Losses')

    # Plotting training and validation accuracies
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='train_accuracy')
    plt.plot(val_accuracies, label='val_accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracies')

    # Plotting training and validation recalls
    plt.subplot(1, 3, 3)
    plt.plot(train_recalls, label='train_recall')
    plt.plot(val_recalls, label='val_recall')
    plt.legend()
    plt.title('Training and Validation Recalls')

    # Display the plots
    plt.show()

# Initialize the Xception-based model
xception_model = PlantDiseaseXception(len(train.classes))

# Move the model and data loaders to the same device
device = get_default_device()
xception_model = to_device(xception_model, device)
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)

# Get a summary of the Xception-based model
INPUT_SHAPE = (3, 256, 256)
print(summary(xception_model, INPUT_SHAPE))

# Train the Xception-based model
epochs = 20
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

history_xception = fit_OneCycle(epochs, max_lr, xception_model, train_dl, valid_dl,
                                 grad_clip=grad_clip, weight_decay=1e-4, opt_func=opt_func)

# Plot training and validation metrics
plot_metrics(history_xception)

# Calculate metrics on the validation set
accuracy, recall, f1 = get_metrics(xception_model, valid_dl)

print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')


# In[ ]:




