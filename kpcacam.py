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
               self.features = output[0].detach().cpu().numpy()
           else:
               self.features = output.detach().cpu().numpy()
       
       if self.model_type == 'resnet50':
           self.model.layer4[-1].register_forward_hook(hook_fn)
       elif self.model_type == 'vgg16':
           self.model.features[-1].register_forward_hook(hook_fn)
       elif self.model_type == 'vit':
           self.model.encoder.layers[-1].register_forward_hook(hook_fn)

   def preprocess_image(self, image_path):
       image = Image.open(image_path).convert('RGB')
       input_tensor = self.transform(image)
       return input_tensor.unsqueeze(0).to(self.device)

   def _load_bbox_data(self, csv_path):
    df = pd.read_csv(csv_path)
    bbox_data = {}
    
    for _, row in df.iterrows():
        try:
            # Lấy các giá trị từ các cột x, y, w, h
            bbox = []
            for col in ['Bbox [x', 'y', 'w', 'h]']:
                try:
                    bbox.append(float(str(row[col]).strip()))
                except:
                    print(f"Warning: Invalid bbox value in {col} for {row['Image Index']}")
                    continue
            
            if len(bbox) == 4:  # Chỉ lưu nếu có đủ 4 giá trị
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
   
   def generate_kpca_cam(self, input_tensor, target_class, n_components=3):
       with torch.no_grad():
           output = self.model(input_tensor)
       
       if self.model_type == 'vit':
           print("ViT feature shape:", self.features.shape)
           feature_maps = self.features[1:] 
           h = w = int(np.sqrt(len(feature_maps)))
           feature_maps = feature_maps.reshape(h, w, -1)
           
           from sklearn.decomposition import PCA
           pca = PCA(n_components=64)
           feature_maps_reduced = pca.fit_transform(feature_maps.reshape(-1, 768))
           feature_maps = feature_maps_reduced.reshape(h, w, -1).transpose(2, 0, 1)
           
       elif self.model_type == 'resnet50':
           feature_maps = self.features[0]
       
       elif self.model_type == 'vgg16':
           feature_maps = self.features[0]
       
       print(f"{self.model_type} final feature_maps shape:", feature_maps.shape)
       
       reshaped_features = feature_maps.reshape(feature_maps.shape[0], -1).T
       kpca = KernelPCA(n_components=n_components, kernel='rbf')
       kpca_features = kpca.fit_transform(reshaped_features)
       
       cam = np.zeros(feature_maps.shape[1:])
       for comp in kpca_features.T:
           comp_map = comp.reshape(feature_maps.shape[1:])
           cam += np.abs(comp_map)
       
       cam = cv2.resize(cam, (224, 224))
       cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
       
       return cam, output[0, target_class].item()

   def get_bounding_box(self, cam, threshold=0.5):
       binary = (cam > threshold).astype(np.uint8)
       contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
       if contours:
           largest_contour = max(contours, key=cv2.contourArea)
           return cv2.boundingRect(largest_contour)
       return None

   def visualize(self, image_path, target_class, output_dir='outputs', bbox_csv=None):
       os.makedirs(output_dir, exist_ok=True)
       
       input_tensor = self.preprocess_image(image_path)
       original_image = cv2.imread(image_path)
       h, w = original_image.shape[:2]
       original_image = cv2.resize(original_image, (224, 224))
       scale_x, scale_y = 224/w, 224/h
       
       cam, confidence = self.generate_kpca_cam(input_tensor, target_class)
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
   parser.add_argument('--models_dir', type=str, required=True)
   parser.add_argument('--image_path', type=str, required=True)
   parser.add_argument('--target_class', type=int, required=True)
   parser.add_argument('--output_dir', type=str, default='outputs')
   parser.add_argument('--list_classes', action='store_true', help='List available classes')
   parser.add_argument('--num_images', type=int, default=None, help='Number of images to process')
   parser.add_argument('--bbox_csv', type=str, help='Path to bbox CSV file')
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
                       output_path, conf, iou = visualizer.visualize(
                           img_path, 
                           args.target_class,
                           args.output_dir,
                           args.bbox_csv
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

   if not args.list_classes:
       print("\nProcessing complete!")
       for key, result in results.items():
           print(f"\n{key}:")
           print(f"Path: {result['path']}")
           print(f"Confidence: {result['confidence']:.3f}")
           if result['iou'] is not None:
               print(f"IoU: {result['iou']:.3f}")

if __name__ == "__main__":
   main()