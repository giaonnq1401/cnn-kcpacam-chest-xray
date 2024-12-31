# model_factory.py

from torchvision import models
import torch.nn as nn
from transformers import ViTForImageClassification

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
            # model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            # model.classifier = nn.Sequential(
            #     nn.Linear(768, num_classes),
            #     nn.Sigmoid()
            # )
            model = models.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
            model.heads = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
            return model