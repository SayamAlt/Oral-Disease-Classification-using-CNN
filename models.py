import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1) # 64,64,32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # 32,32,64
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # 16,16,128
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # 8,8,256
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(in_features=8*8*256, out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=128)
        self.fc3 = nn.Linear(in_features=128,out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=1)

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = F.relu(self.bn2(self.conv2(X)))
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = F.relu(self.bn3(self.conv3(X)))
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = F.relu(self.bn4(self.conv4(X)))
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = X.view(-1,8*8*256)
        X = F.relu(self.fc1(X))
        X = F.dropout(X,p=0.2)
        X = F.relu(self.fc2(X))
        X = F.dropout(X,p=0.1)
        X = F.relu(self.fc3(X))
        return self.fc4(X)  # Raw output for use with BCEWithLogitsLoss