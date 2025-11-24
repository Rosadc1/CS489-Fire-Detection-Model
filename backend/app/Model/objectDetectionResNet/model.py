import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

class Yolov8ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Load YOLO model
        self.yolo = YOLO("yolov8n.pt")
        model = self.yolo.model.model
        
        model[0] = ResNetBackbone()

        # Update detection head classes
        detect = model[-1]
        detect.nc = num_classes
        detect.initialize_biases()
        
        self.model = self.yolo.model

    def forward(self, x):
        return self.model(x)

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