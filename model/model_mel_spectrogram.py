import torch
from torch.nn import Linear, Conv2d, MaxPool2d, ReLU, Softmax

class C2DNN_v1(torch.nn.Module):
    
    def __init__(self):
        super(C2DNN_v1, self).__init__()
        
        self.conv1 = Conv2d(1, 32, 3)
        self.conv2 = Conv2d(32, 64, 3)
        self.conv3 = Conv2d(64, 128, 3)
        self.conv4 = Conv2d(128, 256, 3)
        
        self.maxpool2d = MaxPool2d(2)
        self.relu = ReLU()
        
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
        
        x = self.conv4(x)
        x = self.maxpool2d(x)
        x = self.relu(x)
        
        x = torch.flatten(x, start_dim=1)
        
        return x

class NeuralNet_v1(torch.nn.Module):
    
    def __init__(self):
        super(NeuralNet_v1, self).__init__()
        x1_shape = C2DNN_v1()(torch.randn((1,1,128,4355))).shape[1]
        x2_shape = C2DNN_v1()(torch.randn((1,1,128,4279))).shape[1]
        
        self.feature1 = C2DNN_v1()
        self.feature2 = C2DNN_v1()
        
        self.fc1 = Linear(x1_shape + x2_shape, 256)
        self.fc2 = Linear(256, 1)
        
        self.relu = ReLU()
        self.softmax = Softmax(dim=1)
        
    def forward(self, x1, x2):
        x1 = self.feature1(x1)
        x2 = self.feature1(x2)
        
        x = torch.cat([x1, x2], dim=1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x

