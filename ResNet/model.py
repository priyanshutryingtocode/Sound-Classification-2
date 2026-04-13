import torch
import torch.nn as nn
import torchvision.models as models

class ResNetAudio(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetAudio, self).__init__()
        
        # Download the pre-trained ResNet18 weights
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # HACK: Replace the first layer to accept 1-channel audio instead of 3-channel RGB
        original_first_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Copy over the "Red" channel's pre-trained weights to maintain some learned edge-detection
        with torch.no_grad():
            self.resnet.conv1.weight[:] = original_first_layer.weight[:, 0:1, :, :]
            
        # Modify the final output layer for 10 sound classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)