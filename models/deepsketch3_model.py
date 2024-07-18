import torch
import torch.nn as nn
from torchvision import models

# DeepSketch 3 ResNet18 model model
class ResNet18WithDropout(nn.Module):
    def __init__(self, dropout_rate=0.5, pretrained=False, output_encoding=False):
        super(ResNet18WithDropout, self).__init__()
        self.output_encoding = output_encoding
        if pretrained == True:
            self.resnet = models.resnet18(weights='DEFAULT')
        else:
            self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 240)  # Assuming 250 categories in TU-Berlin
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        encoding = torch.flatten(x, 1)
        if self.output_encoding:
            return encoding
        x = self.dropout(encoding)
        x = self.resnet.fc(x)
        return x

