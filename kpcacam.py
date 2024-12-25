import os
import torch
import numpy as np
from sklearn.decomposition import KernelPCA
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from models.model_factory import ModelFactory
from utils.preprocess import get_transforms

class KPCACAMVisualizer:
    def __init__(self, model_path, model_type, device='cuda'):
        self.device = device
        self.model_type = model_type.lower()
        self.model = self._load_model(model_path)
        self.model.eval()
        self.features = None
        self._register_hooks()
        _, self.transform = get_transforms()

    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        num_classes = self._get_num_classes(checkpoint)
        model = ModelFactory.get_model(self.model_type, num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(self.device)

    def _get_num_classes(self, checkpoint):
        if self.model_type == 'resnet50':
            return checkpoint['model_state_dict']['fc.0.weight'].shape[0]
        elif self.model_type == 'vgg16':
            return checkpoint['model_state_dict']['classifier.6.0.weight'].shape[0]
        elif self.model_type == 'vit':
            return checkpoint['model_state_dict']['heads.head.0.weight'].shape[0]

    def _register_hooks(self):
        def hook_fn(module, input, output):
            if self.model_type == 'vit':
                # Get output from the last attention block
                self.features = output[0].detach().cpu().numpy()
            else:
                self.features = output.detach().cpu().numpy()
        
        if self.model_type == 'resnet50':
            self.model.layer4[-1].register_forward_hook(hook_fn)
        elif self.model_type == 'vgg16':
            self.model.features[-1].register_forward_hook(hook_fn)
        elif self.model_type == 'vit':
            # Hook into the last encoder block
            self.model.encoder.layers[-1].register_forward_hook(hook_fn)

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image)
        return input_tensor.unsqueeze(0).to(self.device)
    
    def generate_kpca_cam(self, input_tensor, target_class, n_components=3):
        with torch.no_grad():
            output = self.model(input_tensor)
        
        if self.model_type == 'vit':
            print("ViT feature shape:", self.features.shape)
            
            # ViT features have shape (197, 768) - 197 patches (14x14 + 1 cls token)
            feature_maps = self.features[1:] # Ignore cls token -> (196, 768)
            
            # Reshape to spatial format (14, 14, 768)
            h = w = int(np.sqrt(len(feature_maps)))  # h = w = 14
            feature_maps = feature_maps.reshape(h, w, -1)
            
            # Dimensionality reduction with PCA before applying KPCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=64)
            feature_maps_reduced = pca.fit_transform(feature_maps.reshape(-1, 768))
            feature_maps = feature_maps_reduced.reshape(h, w, -1).transpose(2, 0, 1)
            
        elif self.model_type == 'resnet50':
            feature_maps = self.features[0]  # [2048, 7, 7]
        
        elif self.model_type == 'vgg16':
            feature_maps = self.features[0]  # [512, 14, 14]
        
        # Print shape to check
        print(f"{self.model_type} final feature_maps shape:", feature_maps.shape)
        
        # Reshape features for KPCA [num_features, spatial_dims]
        reshaped_features = feature_maps.reshape(feature_maps.shape[0], -1).T
        
        # Apply Kernel PCA
        kpca = KernelPCA(n_components=n_components, kernel='rbf')
        kpca_features = kpca.fit_transform(reshaped_features)
        
        # Create CAM
        cam = np.zeros(feature_maps.shape[1:])  # [height, width]
        for comp in kpca_features.T:
            comp_map = comp.reshape(feature_maps.shape[1:])
            cam += np.abs(comp_map)  # Use absolute value
        
        # Normalize and resize
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--target_class', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--list_classes', action='store_true', help='List available classes')
    parser.add_argument('--num_images', type=int, default=None, help='Number of images to process')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_paths = {
        'resnet50': os.path.join(args.models_dir, 'resnet50', 'best_model.pth'),
        'vgg16': os.path.join(args.models_dir, 'vgg16', 'best_model.pth'),
        'vit': os.path.join(args.models_dir, 'vit', 'best_model.pth')
    }

    results = {}

    for model_type, model_path in model_paths.items():
        if os.path.exists(model_path):
            # List classes if requested
            if args.list_classes:
                checkpoint = torch.load(model_path, map_location=device)
                if 'classes' in checkpoint:
                    print(f"\nAvailable classes for {model_type}:")
                    for idx, class_name in enumerate(checkpoint['classes']):
                        print(f"{idx}: {class_name}")
            else:
                print(f"\nProcessing {model_type}...")
                visualizer = KPCACAMVisualizer(model_path, model_type, device)
                
                if os.path.isdir(args.image_path):
                    image_files = [f for f in os.listdir(args.image_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm'))]
                    
                    if args.num_images:
                        image_files = image_files[:args.num_images]
                        
                    for img_file in image_files:
                        img_path = os.path.join(args.image_path, img_file)
                        output_path = visualizer.visualize(img_path, args.target_class, args.output_dir)
                        results[f"{model_type}_{img_file}"] = output_path
                else:
                    output_path = visualizer.visualize(args.image_path, args.target_class, args.output_dir)
                    results[model_type] = output_path

    if not args.list_classes:
        print("\nProcessing complete!")
        for key, path in results.items():
            print(f"{key}: {path}")


if __name__ == "__main__":
    main()
