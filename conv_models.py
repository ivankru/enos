import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
    
transform_resnet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# Transformations
transform_cnn = transforms.Compose([
    transforms.ToTensor(),  # (H,W) -> (1,H,W), [0,1]
])

class CustomDataset_ResNet(Dataset):
    def __init__(self, data, labels, transform=transform_resnet):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample_1 = self.data[idx]       # (267, 358) np.float32
        label = int(self.labels[idx]) # 0..7
        sample = np.stack([sample_1, sample_1, sample_1], axis=-1)
        if self.transform:
            sample = self.transform(sample)  # -> (1,267,358) torch.float32
        return sample, label

class ResNet4Nose(nn.Module):
    def __init__(self, num_classes=2,):
        super().__init__()

        self.model = models.resnet18(weights='DEFAULT')
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=num_classes)
    
    def forward(self, x):

        out = self.model(x)
        return out
    

class HeightWiseCNN(nn.Module):
    def __init__(self, num_classes=8, n_channels=267):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(n_channels, 15), padding=(0, 2))
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        # "same"-паддинг по ширине
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 10), padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 20), padding='same')

        self.head_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # (B,32,1,358)
        x = self.pool(x)           # (B,32,1,179)
        #x = self.norm1(x)
        x = F.relu(self.conv2(x))  # (B,64,1,179)
        x = self.pool(x)            # (B,64,1,89)
        x = F.relu(self.conv3(x))  # (B,128,1,89)
        x = self.head_pool(x)      # (B,128,1,1)
        x = x.flatten(1)           # (B,128)
        x = self.dropout(x)
        #x = F.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=transform_cnn):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]       # (267, 358) np.float32
        label = int(self.labels[idx]) # 0..7
        if self.transform:
            sample = self.transform(sample)  # -> (1,267,358) torch.float32
        return sample, label

def accuracy(outputs, labels):
    preds = outputs.argmax(1)
    return (preds == labels).float().mean().item()

def roc_auc(outputs, labels):
    # probs for pos class (second col)
    probs = F.softmax(outputs, dim=1)[:, 1]  # shape: (batch_size,)
    return roc_auc_score(labels.cpu().numpy(), probs.detach().cpu().numpy())
