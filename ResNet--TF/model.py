import torch
import torch.nn as nn
import torchvision.models as models

class TFAttention(nn.Module):
    """
    Time-Frequency Attention Module. 
    Independently models attention across temporal and spectral axes.
    """
    def __init__(self, channels, reduction_ratio=16):
        super(TFAttention, self).__init__()
        # Frequency Branch
        self.pool_freq = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_freq = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Temporal Branch
        self.pool_time = nn.AdaptiveAvgPool2d((1, None))
        self.conv_time = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_freq = self.pool_freq(x)
        a_freq = self.conv_freq(x_freq)
        
        x_time = self.pool_time(x)
        a_time = self.conv_time(x_time)
        
        return x * a_freq * a_time

class ResNetTF(nn.Module):
    """
    Adapted ResNet-18 architecture with injected TF-Attention modules.
    """
    def __init__(self, num_classes=10):
        super(ResNetTF, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify first conv layer to accept 1-channel spectrograms
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.backbone.conv1.weight[:] = original_conv1.weight[:, 0:1, :, :]
            
        # ResNet-18 specific channel sizes
        self.tf_attn1 = TFAttention(channels=64)
        self.tf_attn2 = TFAttention(channels=128)
        self.tf_attn3 = TFAttention(channels=256)
        self.tf_attn4 = TFAttention(channels=512)
        
        # Classification Head with strong dropout
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.tf_attn1(x)
        
        x = self.backbone.layer2(x)
        x = self.tf_attn2(x)
        
        x = self.backbone.layer3(x)
        x = self.tf_attn3(x)
        
        x = self.backbone.layer4(x)
        x = self.tf_attn4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x