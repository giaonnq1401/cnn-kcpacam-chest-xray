# KPCA-CAM for Chest X-ray Analysis

## About
Apply KPCA-CAM to analyze lung X-ray images, determine the image area that the model focuses on when making predictions.

## Installation
1. Clone repo:
   ```bash
   git clone <link-repo>
   cd kpca_cam

## Setup environment
```
python -m venv env
source env/bin/activate  # Windows: .\env\Scripts\activate
```

```
pip install -r requirements.txt
or
make install
```

## Finetune
```
make fine-tun   # Default: Resnet50
```


## Visualize KPCA-CAM
```
make run
```

# References
[jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

[https://arxiv.org/abs/2410.00267](https://arxiv.org/abs/2410.00267)

KPCA-CAM: Visual Explainability of Deep Computer Vision Models using Kernel PCA Sachin Karmani, Thanushon Sivakaran, Gaurav Prasad, Mehmet Ali, Wenbo Yang, Sheyang Tang

[alkzar90/NIH-Chest-X-ray-dataset](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset)