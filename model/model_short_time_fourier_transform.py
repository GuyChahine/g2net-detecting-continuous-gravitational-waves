import torch
from torch import nn
from torchviz import make_dot

class C2DNN_v1(torch.nn.Module):
    
    def __init__(self):
        super(C2DNN_v1, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 16, 5, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, 2)
        self.conv3 = nn.Conv2d(32, 64, 5, 2)
        
        self.maxpool2d = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool2d(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.maxpool2d(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.maxpool2d(x)
        x = self.relu(x)
        
        x = torch.flatten(x, start_dim=1)
        
        return x
    
class NeuralNet_v1(torch.nn.Module):
    
    def __init__(self):
        super(NeuralNet_v1, self).__init__()
        
        self.feature1 = C2DNN_v1()
        self.feature2 = C2DNN_v1()
        
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x1, x2, x3):
        x1 = self.feature1(x1)
        x2 = self.feature2(x2)
        
        x = torch.cat([x1, x2, x3], dim=1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x


class C1DNN_v2(torch.nn.Module):
    
    def __init__(self):
        super(C1DNN_v2, self).__init__()
        
        self.conv1 = nn.Conv1d(360, 512, 5, 2)
        self.conv2 = nn.Conv1d(512, 1024, 5, 2)
        self.conv3 = nn.Conv1d(1024, 2048, 5, 2)
        
        self.maxpool1d = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1d(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool1d(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool1d(x)
        
        x = torch.flatten(x, start_dim=1)
        
        return x
    
class NeuralNet_v2(torch.nn.Module):
    
    def __init__(self):
        super(NeuralNet_v2, self).__init__()
        
        self.feature1 = C1DNN_v2()
        self.feature2 = C1DNN_v2()
        self.feature3 = C1DNN_v2()
        self.feature4 = C1DNN_v2()
        
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, 1)
        
        self.bn1 = nn.LazyBatchNorm1d()
        self.bn2 = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x1, x2, x3, x4):
        x1 = self.feature1(x1)
        x2 = self.feature2(x2)
        x3 = self.feature3(x3)
        x4 = self.feature4(x4)
        
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x
    
    
class C2DNN_v3(torch.nn.Module):
    
    def __init__(self):
        super(C2DNN_v3, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 8, 5, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, 2)
        self.conv3 = nn.Conv2d(16, 32, 5, 2)
        
        self.maxpool2d = nn.MaxPool2d(3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        
        x = torch.flatten(x, start_dim=1)
        
        return x
    
class NeuralNet_v3(torch.nn.Module):
    
    def __init__(self):
        super(NeuralNet_v3, self).__init__()
        
        self.feature1 = C2DNN_v3()
        self.feature2 = C2DNN_v3()
        self.feature3 = C2DNN_v3()
        self.feature4 = C2DNN_v3()
        
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, 1)
        
        self.bn1 = nn.LazyBatchNorm1d()
        self.bn2 = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x1, x2, x3, x4):
        x1 = self.feature1(x1)
        x2 = self.feature2(x2)
        x3 = self.feature3(x3)
        x4 = self.feature4(x4)
        
        x = torch.cat([x1, x2, x3, x4], dim=1)
        
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x
    
    
"""x = [
    torch.randn(2, 1, 360, 4355),
    torch.randn(2, 1, 360, 4355),
    torch.randn(2, 1, 360, 4355),
    torch.randn(2, 1, 360, 4355),
]
    
model = NeuralNet_v3()
model(*x)"""