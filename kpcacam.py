import os
import torch
import numpy as np
from sklearn.decomposition import KernelPCA
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from torchvision.models import resnet50, vgg16, ResNet50_Weights, VGG16_Weights
from vit_pytorch import ViT
from utils.preprocess import get_transforms

class KPCACAMVisualizer:
    def __init__(self, model_path, model_type, device='cuda'):
        self.device = device
        self.model_type = model_type
        self.model = self._load_model(model_path)
        self.model.eval()
        self.features = None
        self._register_hooks()
        _, self.transform = get_transforms()

    def _load_model(self, model_path):
        if self.model_type == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            num_ftrs = model.fc.in_features
        elif self.model_type == 'vgg16':
            model = vgg16(weights=VGG16_Weights.DEFAULT)
            num_ftrs = model.classifier[0].in_features
        elif self.model_type == 'vit':
            model = ViT(
                image_size=224,
                patch_size=16,
                num_classes=14,
                dim=768,
                depth=12,
                heads=12,
                mlp_dim=3072
            )
            num_ftrs = model.mlp_head[0].in_features
        
        checkpoint = torch.load(model_path, map_location=self.device)
        num_classes = self._get_num_classes(checkpoint)
        
        self._modify_classifier(model, num_ftrs, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(self.device)

    def _get_num_classes(self, checkpoint):
        if self.model_type == 'resnet50':
            return checkpoint['model_state_dict']['fc.0.weight'].shape[0]
        elif self.model_type == 'vgg16':
            return checkpoint['model_state_dict']['classifier.6.weight'].shape[0]
        elif self.model_type == 'vit':
            return checkpoint['model_state_dict']['mlp_head.1.weight'].shape[0]

    def _modify_classifier(self, model, num_ftrs, num_classes):
        if self.model_type == 'resnet50':
            model.fc = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, num_classes),
                torch.nn.Sigmoid()
            )
        elif self.model_type == 'vgg16':
            model.classifier = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 4096),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, num_classes),
                torch.nn.Sigmoid()
            )
        elif self.model_type == 'vit':
            model.mlp_head = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, num_classes),
                torch.nn.Sigmoid()
            )

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.features = output.detach().cpu().numpy()
        
        if self.model_type == 'resnet50':
            self.model.layer4[-1].register_forward_hook(hook_fn)
        elif self.model_type == 'vgg16':
            self.model.features[-1].register_forward_hook(hook_fn)
        elif self.model_type == 'vit':
            self.model.transformer.layers[-1].register_forward_hook(hook_fn)

    def generate_kpca_cam(self, input_tensor, target_class, n_components=3):
        with torch.no_grad():
            output = self.model(input_tensor)
        
        feature_maps = self.features[0]
        
        if self.model_type == 'vit':
            # Reshape ViT features to spatial dimensions
            feat_size = int(np.sqrt(feature_maps.shape[0] - 1))  # exclude CLS token
            feature_maps = feature_maps[1:].reshape(feat_size, feat_size, -1).transpose(2, 0, 1)
        
        reshaped_features = feature_maps.reshape(feature_maps.shape[0], -1).T
        
        kpca = KernelPCA(n_components=n_components, kernel='rbf')
        kpca_features = kpca.fit_transform(reshaped_features)
        
        cam = np.zeros(feature_maps.shape[1:])
        for comp in kpca_features.T:
            comp_map = comp.reshape(feature_maps.shape[1:])
            cam += comp_map
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        
        return cam

    def get_bounding_box(self, cam, threshold=0.5):
        binary = (cam > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return (x, y, w, h)
        return None

    def visualize(self, image_path, target_class, output_dir='outputs'):
        os.makedirs(output_dir, exist_ok=True)
        
        input_tensor = self.preprocess_image(image_path)
        original_image = cv2.imread(image_path)
        original_image = cv2.resize(original_image, (224, 224))
        
        cam = self.generate_kpca_cam(input_tensor, target_class)
        bbox = self.get_bounding_box(cam)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(superimposed, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        plt.figure(figsize=(16, 4))
        
        plt.subplot(141)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(142)
        plt.imshow(cam, cmap='jet')
        plt.title('KPCA-CAM')
        plt.axis('off')
        
        plt.subplot(143)
        plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        plt.title('Superimposed with Bounding Box')
        plt.axis('off')
        
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f'{img_name}_{self.model_type}_kpca_cam.png')
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return output_path