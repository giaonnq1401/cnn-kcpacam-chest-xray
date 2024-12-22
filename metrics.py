import numpy as np

def calculate_precision(cam_results, ground_truths, threshold=0.5):
    """
    Calculate the Precision measure of KPCA-CAM.
    """
    true_positive = 0
    false_positive = 0

    for cam, ground_truth in zip(cam_results, ground_truths):
        cam_binary = (cam > threshold).astype(int)
        true_positive += np.sum((cam_binary == 1) & (ground_truth == 1))
        false_positive += np.sum((cam_binary == 1) & (ground_truth == 0))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    return precision
