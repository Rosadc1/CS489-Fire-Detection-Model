import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class UNet(nn.Module):
    def __init__(self, num_classes=2, backbone='resnet50', pretrained=False):
        # super(ResNetUNet, self).__init__()
        super(UNet, self).__init__()
        """


        TO DO


        """
        # Unet encoder
        self.encBlock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1 
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )       
        self.encBlockPool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encBlock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1 
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.encBlockPool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encBlock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1 
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.encBlockPool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encBlock4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1 
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        ) 

        self.encBlockPool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck 
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                padding=1 
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
    
        # Unet decoder
        self.transConv4 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2
        )

        self.upBlock4 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),  
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.transConv3 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )
        
        self.upBlock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),  
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.transConv2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )
        
        self.upBlock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.transConv1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )
        
        self.upBlock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.outConv = nn.Conv2d(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=1
        )


    def forward(self, x):
        # Encoder
        """


        TO DO


        """
        x = self.encBlock1(x)
        skip1 = x
        x = self.encBlockPool1(x) 
        
        x = self.encBlock2(x)
        skip2 = x
        x = self.encBlockPool2(x)

        x = self.encBlock3(x)
        skip3 = x
        x = self.encBlockPool3(x)
        
        x = self.encBlock4(x)
        skip4 = x
        x = self.encBlockPool4(x)

        x = self.bottleneckBlock(x)
        # Decoder
        """


        TO DO


        """
        x = self.transConv4(x)
        x = torch.cat((x, skip4), dim=1)
        x = self.upBlock4(x)

        x = self.transConv3(x)
        x = torch.cat((x, skip3), dim=1)
        x = self.upBlock3(x)

        x = self.transConv2(x)
        x = torch.cat((x, skip2), dim=1)
        x = self.upBlock2(x)

        x = self.transConv1(x)
        x = torch.cat((x, skip1), dim=1)
        x = self.upBlock1(x)
        # 🟩 Final upsampling step
        """


        TO DO


        """
        out = self.outConv(x)
        return out
    
class ResNetUNet(nn.Module):
    def __init__(self, num_classes=2, backbone='resnet50', pretrained=True):
        super(ResNetUNet, self).__init__()
        self.ResNet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        self.encBlock1 = nn.Sequential(
            self.ResNet.conv1,
            self.ResNet.bn1,
            self.ResNet.relu,
        )
        self.encBlock1MaxPool = self.ResNet.maxpool
        self.encBlock2 = self.ResNet.layer1
        self.encBlock3 = self.ResNet.layer2
        self.encBlock4 = self.ResNet.layer3
        self.encBlock5 = self.ResNet.layer4

        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(
                in_channels=2048,
                out_channels=1024,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                padding=1 
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
    
        # Unet decoder

        self.transConv4 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2
        )
        
        self.upBlock4 = nn.Sequential(
            nn.Conv2d(
                in_channels=2560,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),  
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.transConv3 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )
        
        self.upBlock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1280,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),  
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.transConv2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )
        
        self.upBlock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=640,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.transConv1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )
        
        self.upBlock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=320,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.transConv0 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=2,
            stride=2
        )

        self.upBlock0 = nn.Sequential(
            nn.Conv2d(
                in_channels=96,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),  
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )     
        self.outConv = nn.Conv2d(
            in_channels=32,
            out_channels=num_classes,
            kernel_size=1
        )

        #""" Frozen Encoder ResUNet1
        for param in self.ResNet.parameters():
            param.requires_grad = False
        #"""

        #""" Full-fine tuning + no aug ResUNet2
        for param in self.ResNet.parameters():
            param.requires_grad = True        
        #"""

        #""" Partial-fine tuning + no aug ResUNet3
        for layer in [self.ResNet.layer1, self.ResNet.layer2, self.ResNet.layer3, self.ResNet.layer4]:
            for param in layer.parameters():
                param.requires_grad = False    
        #"""

        #""" Full-fine tuning + aug ResUNet4
        for param in self.ResNet.parameters():
            param.requires_grad = True        
        #"""
  
    def forward(self, x):
        # Encoder
        x = self.encBlock1(x)
        skip0 = x
        x = self.encBlock1MaxPool(x)

        x = self.encBlock2(x)
        skip1 = x
        
        x = self.encBlock3(x)
        skip2 = x

        x = self.encBlock4(x)
        skip3 = x

        x = self.encBlock5(x)
        skip4 = x
        # bottleNeck
        x = self.bottleneckBlock(x)

        # Decoder
        x = self.transConv4(x)
        x = torch.cat((x, skip4), dim=1)
        x = self.upBlock4(x)

        x = self.transConv3(x)
        x = torch.cat((x, skip3), dim=1)
        x = self.upBlock3(x)

        x = self.transConv2(x)
        x = torch.cat((x, skip2), dim=1)
        x = self.upBlock2(x)

        x = self.transConv1(x)
        x = torch.cat((x, skip1), dim=1)
        x = self.upBlock1(x)

        x = self.transConv0(x)
        x = torch.cat((x, skip0), dim=1)
        x = self.upBlock0(x)

        out = self.outConv(x)
        return out