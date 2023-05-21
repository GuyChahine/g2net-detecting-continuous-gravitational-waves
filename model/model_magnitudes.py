import torch
from torch import nn
from torchviz import make_dot

class C2DNN_v1(torch.nn.Module):
    
    def __init__(self):
        super(C2DNN_v1, self).__init__()
        
        self.conv1_1 = nn.Conv2d(1, 32, 9, 3)
        self.conv1_2 = nn.Conv2d(32, 32, 3, 1)
        self.conv1_3 = nn.Conv2d(32, 32, 3, 1)
        
        self.conv2_1 = nn.Conv2d(32, 64, 3, 1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, 1)
        self.conv2_3 = nn.Conv2d(64, 64, 3, 1)
        
        self.conv3_1 = nn.Conv2d(64, 128, 3, 1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, 1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, 1)
        
        self.maxpool2d = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        
        x = self.maxpool2d(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        
        x = self.maxpool2d(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        
        x = self.maxpool2d(x)
        
        
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
    
    def forward(self, x1, x2):
        x1 = self.feature1(x1)
        x2 = self.feature2(x2)
        
        x = torch.cat([x1, x2], dim=1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x


class C2DNN_v2(torch.nn.Module):
    
    def __init__(self):
        super(C2DNN_v2, self).__init__()

        self.conv1 = nn.Conv1d(360, 720, 64, 4)
        self.bn1 = nn.BatchNorm1d(720)
        self.conv2 = nn.Conv1d(720, 1440, 32, 3)
        self.bn2 = nn.BatchNorm1d(1440)
        self.conv3 = nn.Conv1d(1440, 2880, 16, 2)
        self.bn3 = nn.BatchNorm1d(2880)
        
        self.maxpool1d = nn.MaxPool1d(3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1d(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool1d(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool1d(x)
        
        x = torch.flatten(x, start_dim=1)
        
        return x
    
class NeuralNet_v2(torch.nn.Module):
    
    def __init__(self):
        super(NeuralNet_v2, self).__init__()
        
        self.feature1 = C2DNN_v2()
        self.feature2 = C2DNN_v2()
        
        self.fc1 = nn.LazyLinear(2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x1, x2):
        x1 = self.feature1(x1)
        x2 = self.feature2(x2)
        
        x = torch.cat([x1, x2], dim=1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x
    
if __name__ == "__main__":
    x = [
        torch.randn(2, 1, 360, 4355),
        torch.randn(2, 1, 360, 4355),
    ]
        
    model = NeuralNet_v1()
    model(*x)

    for name, weight in dict(model.named_parameters()).items():
        print(f"{name} {weight.numel():,}")
    print(f"\n{sum([p.numel() for p in model.parameters()]):,}")