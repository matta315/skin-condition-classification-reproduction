import torch
import torch.nn as nn
import torchvision.models as models

class VGG16SkinConditionClassifier(nn.Module):
    """
    VGG-16 based model for skin condition classification.
    This model is based on the architecture described in Groh et al.'s 2021 paper.
    """
    
    def __init__(self, num_classes=114, pretrained=True):
        """
        Initialize the VGG-16 based skin condition classifier.
        
        Args:
            num_classes (int): Number of skin condition classes to predict
            pretrained (bool): Whether to use pretrained weights from ImageNet
        """
        super(VGG16SkinConditionClassifier, self).__init__()
        
        # Load pretrained VGG16 model
        vgg16 = models.vgg16(pretrained=pretrained)
        
        # Extract feature extractor (all convolutional layers)
        self.features = vgg16.features
        
        # Modify classifier for our specific task
        self.avgpool = vgg16.avgpool
        
        # Create new classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        
        # Initialize weights for the new layers
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x