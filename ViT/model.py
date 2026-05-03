import torch
import torch.nn as nn
import torch.nn.functional as F
import timm # type: ignore

class ViTAudio(nn.Module):
    def __init__(self, num_classes=10):
        super(ViTAudio, self).__init__()
        
        # 1. Load the pre-trained 'tiny224' Vision Transformer.
        # The paper specifically evaluates settings like 'tiny224'.
        # timm is amazing because we can tell it to expect 1-channel (grayscale) 
        # input instead of 3-channel RGB, and it handles the weight conversion automatically!
        self.vit = timm.create_model('vit_tiny_patch16_224', 
                                     pretrained=True, 
                                     in_chans=1, 
                                     num_classes=num_classes)

    def forward(self, x):
        # x is currently coming in as (Batch, 1, 64, 172) from your dataset
        
        # 2. Resize the spectrogram to 224x224 to match the ViT's exact expected dimensions
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 3. Pass it through the transformer
        # As the paper illustrates, this will extract patches, create sequential 
        # embeddings, and pass them through multi-head attention layers.
        return self.vit(x)