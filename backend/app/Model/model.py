import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from ultralytics import YOLO

class CNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(CNN, self).__init__()

    def forward(self, x):
        return x

class Yolov8ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(Yolov8ResNet, self).__init__()
        self.num_classes = num_classes

        self.yolo = YOLO("yolov8n.pt")
        self.yolo.model.model[-1].nc = num_classes  # Update number of classes
        self.yolo.model.model[-1].initialize_biases()

        self.yolo.model.model[:-1] = ResNetBackbone()
    
    def forward(self, x):
        features = self.yolo.model.model[:-1](x)
        x = self.yolo.model.model[-1](features)

        return x

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resNet = models.resnet50(pretrained=True)
        
        self.fastConv = nn.Sequential(
            resNet.conv1,
            resNet.bn1,
            resNet.relu,
            resNet.maxpool
        )
        self.layer1 = resNet.layer1 
        self.layer2 = resNet.layer2
        self.layer3 = resNet.layer3
        self.layer4 = resNet.layer4

        # Below are 1x1 convolutional layers to adapt the ResNet feature maps to the expected channel sizes of the YOLO bottleneck layers
        self.adapt2 = nn.Conv2d( # for ResNet layer2 output
            in_channels=512,
            out_channels=64,
            kernel_size=1,
        )

        self.adapt3 = nn.Conv2d( # for ResNet layer3 output
            in_channels=1024,
            out_channels=128,
            kernel_size=1,
        )


        self.adapt4 = nn.Conv2d( # for ResNet layer4 output
            in_channels=2048,
            out_channels=256,
            kernel_size=1,
        )

    def forward(self, x):
        
        x = self.fastConv(x)
        x = self.layer1(x)
        x2_raw = self.layer2(x)
        x2 = self.adapt2(x2_raw)

        x3_raw = self.layer3(x2_raw)
        x3 = self.adapt3(x3_raw)

        x4_raw = self.layer4(x3_raw)
        x4 = self.adapt4(x4_raw)

        x = [x2, x3, x4]
        return x