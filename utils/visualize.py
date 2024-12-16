import cv2
import numpy as np
import torch

def visualize_cam(original_image, grayscale_cam, output_path):
    """
    Hiển thị và lưu ảnh với CAM overlay.
    """
    # Chuyển đổi ảnh gốc về định dạng numpy (nếu chưa phải numpy)
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.permute(1, 2, 0).numpy()  # Chuyển (C, H, W) -> (H, W, C)
        original_image = (original_image * 255).astype(np.uint8)  # Chuyển từ [0, 1] -> [0, 255]

    # Đảm bảo grayscale_cam có cùng kích thước với original_image
    grayscale_cam_resized = cv2.resize(grayscale_cam, (original_image.shape[1], original_image.shape[0]))

    # Chuyển grayscale_cam thành heatmap (3 kênh màu)
    heatmap = cv2.applyColorMap((255 * grayscale_cam_resized).astype(np.uint8), cv2.COLORMAP_JET)

    # Kết hợp ảnh gốc và heatmap
    overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    # Lưu kết quả
    cv2.imwrite(output_path, overlay)
    print(f"Saved CAM overlay to: {output_path}")
