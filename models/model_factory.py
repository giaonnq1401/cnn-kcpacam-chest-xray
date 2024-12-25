import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vgg16, VGG16_Weights,
    vit_b_16, ViT_B_16_Weights
)

class ModelFactory:
    @staticmethod
    def get_model(model_name, num_classes, pretrained=True):
        model_name = model_name.lower()
        if model_name == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid()
            )
        elif model_name == 'vgg16':
            model = vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid()
            )
        elif model_name == 'vit':
            model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
            num_ftrs = model.heads.head.in_features
            model.heads.head = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        return model