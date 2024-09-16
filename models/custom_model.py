import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import os

class SemsegCustom(nn.Module):
    def __init__(self, checkpoint=None, num_classes=8):
        super(SemsegCustom, self).__init__()

        self.deeplabv3 = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        
        for param in self.deeplabv3.backbone.parameters():
            param.requires_grad = False

        self.deeplabv3.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

        if checkpoint:
            checkpoint = torch.load(checkpoint)
            self.deeplabv3.classifier[4].load_state_dict(checkpoint)

    def forward(self, x):

        output = self.deeplabv3(x)['out']     
        return output