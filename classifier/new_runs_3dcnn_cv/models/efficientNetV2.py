import torch
import torch.nn as nn
import torchvision.models as models

class MRIClassifier(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(MRIClassifier, self).__init__()
        
        # Load pre-trained EfficientNetV2-L (large)
        self.backbone = models.efficientnet_v2_l(pretrained=True)
        
        # Modify first conv layer to accept 115 channels
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            115, 
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Get the number of features from the backbone (1280 for V2-L)
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with a sophisticated head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_features, 1)
        )
        
        # Enable stochastic depth
        self.apply_stochastic_depth()
        
    def apply_stochastic_depth(self, drop_path_rate=0.2):
        """Apply stochastic depth to the model"""
        import math
        
        # Get all FusedMBConv and MBConv blocks
        blocks = [m for m in self.backbone.modules() if 'Block' in str(type(m))]
        num_blocks = len(blocks)
        
        # Linearly increase drop path rate
        for i, block in enumerate(blocks):
            drop_prob = drop_path_rate * i / num_blocks
            if hasattr(block, 'drop_path'):
                block.drop_path.p = drop_prob
        
    def forward(self, x):
        logits = self.backbone(x)
        return logits, None
