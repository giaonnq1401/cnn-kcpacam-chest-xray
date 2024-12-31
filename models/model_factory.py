# model_factory.py

from torchvision import models
import torch.nn as nn

class ModelFactory:
    @staticmethod
    def get_model(model_type, num_classes=1, pretrained=True):
        if model_type == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            model.fc = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
                nn.Sigmoid()
            )
            return model
            
        elif model_type == 'vgg16':
            model = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
            model.classifier[6] = nn.Sequential(
                nn.Linear(4096, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
                nn.Sigmoid()
            )
            return model
            
        elif model_type == 'vit':
            model = models.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
            model.heads = nn.Sequential(
                nn.Linear(768, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
            return model