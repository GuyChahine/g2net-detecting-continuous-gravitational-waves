import torch
from torch import nn
from torchviz import make_dot

class C2DNN_v1(torch.nn.Module):
    
    def __init__(self):
        super(C2DNN_v1, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 2, 64, 2)
        self.conv2 = nn.Conv2d(2, 4, 32, 1)
        self.conv3 = nn.Conv2d(4, 8, 16, 1)
        
        self.maxpool2d = nn.MaxPool2d(2)
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
    
class NeuralNet_v1(torch.nn.Module):
    
    def __init__(self):
        super(NeuralNet_v1, self).__init__()
        
        self.feature1 = C2DNN_v1()
        self.feature2 = C2DNN_v1()
        
        self.fc1 = nn.LazyLinear(4096)
        self.fc2 = nn.Linear(4096, 256)
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