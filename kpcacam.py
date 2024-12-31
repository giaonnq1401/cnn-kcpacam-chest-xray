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
import pandas as pd

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
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            num_classes = 1  # Binary classification
            model = ModelFactory.get_model(self.model_type, num_classes, pretrained=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            return model.to(self.device)
        except Exception as e:
            print(f"Error loading model {model_path}: {str(e)}")
            raise

    def _register_hooks(self):
        def hook_fn(module, input, output):
            if self.model_type == 'vit':
                # For ViT, capture the whole output
                if isinstance(output, tuple):
                    self.features = output[0].detach().cpu().numpy()
                else:
                    self.features = output.detach().cpu().numpy()
            else:
                # For CNN models (ResNet, VGG)
                self.features = output.detach().cpu().numpy()
            
            # Print shape for debugging
            print("Hook captured feature shape:", self.features.shape)
        
        if self.model_type == 'resnet50':
            self.model.layer4[-1].register_forward_hook(hook_fn)
        elif self.model_type == 'vgg16':
            self.model.features[-1].register_forward_hook(hook_fn)
        elif self.model_type == 'vit':
            # For ViT from torchvision, hook into the final encoder layer
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
                self.model.encoder.layers[-1].register_forward_hook(hook_fn)
            else:
                raise AttributeError("Unable to find appropriate layer in ViT model for hook registration.")

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image)
        return input_tensor.unsqueeze(0).to(self.device)

    def _load_bbox_data(self, csv_path):
        df = pd.read_csv(csv_path)
        bbox_data = {}
        
        for _, row in df.iterrows():
            try:
                bbox = []
                for col in ['Bbox [x', 'y', 'w', 'h]']:
                    try:
                        bbox.append(float(str(row[col]).strip()))
                    except:
                        print(f"Warning: Invalid bbox value in {col} for {row['Image Index']}")
                        continue
                
                if len(bbox) == 4:  
                    bbox_data[row['Image Index']] = {
                        'label': row['Finding Label'],
                        'bbox': bbox
                    }
            except Exception as e:
                print(f"Warning: Error processing row for {row['Image Index']}: {e}")
                continue
        
        return bbox_data

    def calculate_iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        intersection_x = max(x1, x2)
        intersection_y = max(y1, y2)
        intersection_w = min(x1 + w1, x2 + w2) - intersection_x
        intersection_h = min(y1 + h1, y2 + h2) - intersection_y
        
        if intersection_w <= 0 or intersection_h <= 0:
            return 0.0
            
        intersection = intersection_w * intersection_h
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union
    
    def generate_kpca_cam(self, input_tensor, n_components=3):
        with torch.no_grad():
            output = self.model(input_tensor)
        
        if self.model_type == 'vit':
            # Handle ViT features
            feature_maps = self.features
            print("Initial feature shape:", feature_maps.shape)
            
            # Remove batch dimension
            feature_maps = feature_maps[0]  # Now shape is (197, 768)
            
            # Remove CLS token (first token)
            feature_maps = feature_maps[1:]  # Now shape is (196, 768)
            
            # Reshape to (14, 14, 768) - patch grid
            patch_grid_size = int(np.sqrt(feature_maps.shape[0]))  # Should be 14
            feature_maps = feature_maps.reshape(patch_grid_size, patch_grid_size, -1)
            
            # Transpose to (hidden_dim, grid, grid) for consistency with CNN features
            feature_maps = feature_maps.transpose(2, 0, 1)  # Now shape is (768, 14, 14)
        else:
            # Handle CNN features (ResNet, VGG)
            # Remove batch dimension
            feature_maps = self.features[0]  # Now shape should be (C, H, W)
        
        print(f"{self.model_type} feature_maps shape:", feature_maps.shape)
        
        # Ensure we're working with valid dimensions
        if len(feature_maps.shape) != 3:
            raise ValueError(f"Expected 3D feature maps, got shape {feature_maps.shape}")
        
        # Reshape features for KPCA
        reshaped_features = feature_maps.reshape(feature_maps.shape[0], -1).T
        
        # Apply KPCA
        kpca = KernelPCA(n_components=n_components, kernel='rbf')
        kpca_features = kpca.fit_transform(reshaped_features)
        
        # Create attention map
        cam = np.zeros(feature_maps.shape[1:])
        for comp in kpca_features.T:
            comp_map = comp.reshape(feature_maps.shape[1:])
            cam += np.abs(comp_map)
        
        # Normalize between 0 and 1, handling potential numerical issues
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_min != cam_max:  # Avoid division by zero
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        # Resize to match input image size
        cam = cv2.resize(cam, (224, 224))
        
        return cam, output.squeeze().item()

    def get_bounding_box(self, cam, threshold=0.5):
        binary = (cam > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(largest_contour)
        return None

    def visualize(self, image_path, output_dir='outputs', bbox_csv=None):
        os.makedirs(output_dir, exist_ok=True)
        
        input_tensor = self.preprocess_image(image_path)
        original_image = cv2.imread(image_path)
        h, w = original_image.shape[:2]
        original_image = cv2.resize(original_image, (224, 224))
        scale_x, scale_y = 224/w, 224/h
        
        cam, confidence = self.generate_kpca_cam(input_tensor)
        pred_bbox = self.get_bounding_box(cam)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        
        if pred_bbox:
            x, y, w, h = pred_bbox
            cv2.rectangle(superimposed, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        iou_score = None
        if bbox_csv:
            bbox_data = self._load_bbox_data(bbox_csv)
            img_name = os.path.basename(image_path)
            if img_name in bbox_data:
                gt_bbox = bbox_data[img_name]['bbox']
                x, y, w, h = [c * scale_x if i % 2 == 0 else c * scale_y 
                             for i, c in enumerate(gt_bbox)]
                cv2.rectangle(superimposed, (int(x), int(y)), 
                             (int(x + w), int(y + h)), (255, 0, 0), 2)
                
                if pred_bbox:
                    iou_score = self.calculate_iou(pred_bbox, (x, y, w, h))
        
        plt.figure(figsize=(16, 4))
        plt.subplot(141)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(142)
        plt.imshow(cam, cmap='jet')
        plt.title(f'KPCA-CAM\nConf: {confidence:.3f}')
        plt.axis('off')
        
        title = 'Superimposed with BBox'
        if iou_score is not None:
            title += f'\nIoU: {iou_score:.3f}'
        plt.subplot(143)
        plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f'{img_name}_{self.model_type}_kpca_cam.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return output_path, confidence, iou_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)

    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--bbox_csv', type=str, help='Path to bbox CSV file')
    parser.add_argument('--num_images', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_paths = {
        'resnet50': 'best_model_ResNet.pth',
        'vgg16': 'best_model_VGG.pth',
        'vit': 'best_model_VisionTransformer.pth'
    }

    results = {}

    for model_type, model_path in model_paths.items():
        if os.path.exists(model_path):
            print(f"\nProcessing {model_type}...")
            visualizer = KPCACAMVisualizer(model_path, model_type, device)
            
            if os.path.isdir(args.image_path):
                image_files = [f for f in os.listdir(args.image_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if args.num_images:
                    image_files = image_files[:args.num_images]
                    
                for img_file in image_files:
                    img_path = os.path.join(args.image_path, img_file)
                    output_path, conf, iou = visualizer.visualize(
                        img_path, 
                        output_dir=args.output_dir,
                        bbox_csv=args.bbox_csv
                    )
                    results[f"{model_type}_{img_file}"] = {
                        'path': output_path,
                        'confidence': conf,
                        'iou': iou
                    }
            else:
                output_path, conf, iou = visualizer.visualize(
                    args.image_path,
                    args.target_class,
                    args.output_dir,
                    args.bbox_csv
                )
                results[model_type] = {
                    'path': output_path,
                    'confidence': conf,
                    'iou': iou
                }

    print("\nProcessing complete!")
    for key, result in results.items():
        print(f"\n{key}:")
        print(f"Path: {result['path']}")
        print(f"Confidence: {result['confidence']:.3f}")
        if result['iou'] is not None:
            print(f"IoU: {result['iou']:.3f}")

if __name__ == "__main__":
    main()