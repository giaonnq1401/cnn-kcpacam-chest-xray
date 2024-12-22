import os
import torch
import numpy as np
from sklearn.decomposition import KernelPCA
import cv2
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from utils.preprocess import get_transforms

class KPCACAMVisualizer:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
        self.features = None
        self._register_hooks()
        _, self.transform = get_transforms()

    def _load_model(self, model_path):
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        
        # Load checkpoint first to know the number of classes
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        num_classes = checkpoint['model_state_dict']['fc.0.weight'].shape[0]
        
        # Initialize the model with the correct number of classes
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, num_classes), 
            torch.nn.Sigmoid()
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(self.device)

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.features = output.detach().cpu().numpy()
        
        # Register hook on the last convolutional layer
        self.model.layer4[-1].register_forward_hook(hook_fn)

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image)
        return input_tensor.unsqueeze(0).to(self.device)

    def generate_kpca_cam(self, input_tensor, target_class, n_components=3):
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Get feature maps
        feature_maps = self.features[0]  # First batch item
        
        # Reshape feature maps
        reshaped_features = feature_maps.reshape(feature_maps.shape[0], -1).T
        
        # Apply Kernel PCA
        kpca = KernelPCA(n_components=n_components, kernel='rbf')
        kpca_features = kpca.fit_transform(reshaped_features)
        
        # Generate CAM using KPCA components
        cam = np.zeros(feature_maps.shape[1:])
        for comp in kpca_features.T:
            comp_map = comp.reshape(feature_maps.shape[1:])
            cam += comp_map
        
        # Normalize CAM
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        
        return cam

    def visualize(self, image_path, target_class, output_dir='outputs'):
        """
        Args:
            image_path: Path to input image
            target_class: Class index to visualize
            output_dir: Directory to save results
        """
        # Create an output directory if it doesn't exist yet
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare image
        input_tensor = self.preprocess_image(image_path)
        original_image = cv2.imread(image_path)
        original_image = cv2.resize(original_image, (224, 224))
        
        # Generate KPCA-CAM
        cam = self.generate_kpca_cam(input_tensor, target_class)
        
        # Apply heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        # Combine original image and heatmap
        superimposed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        
        # Create figure
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(cam, cmap='jet')
        plt.title('KPCA-CAM')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        plt.title('Superimposed')
        plt.axis('off')
        
        # Create output file name from input file name
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f'{img_name}_kpca_cam.png')
        
        # Save figure
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f'Saved visualization to {output_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--target_class', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualizer = KPCACAMVisualizer(args.model_path, device)

    # If image_path is a directory
    if os.path.isdir(args.image_path):
        # Get all image files in the folder
        image_files = [f for f in os.listdir(args.image_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm'))]
        
        print(f'Found {len(image_files)} images to process')
        
        for i, img_file in enumerate(image_files, 1):
            img_path = os.path.join(args.image_path, img_file)
            print(f'\nProcessing image {i}/{len(image_files)}: {img_file}')
            visualizer.visualize(img_path, args.target_class, args.output_dir)
            
        print(f'\nAll visualizations saved to {args.output_dir}/')
    else:
        # Process a single image
        visualizer.visualize(args.image_path, args.target_class, args.output_dir)

if __name__ == "__main__":
    main()