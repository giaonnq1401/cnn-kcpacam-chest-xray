# KPCA-CAM for Chest X-ray Analysis

## About
Apply KPCA-CAM to analyze lung X-ray images, determine the image area that the model focuses on when making predictions.

## Installation
1. Clone repo:
   ```bash
   git clone <link-repo>
   cd kpca_cam

## Setup environment
```python -m venv env
source env/bin/activate  # Windows: .\env\Scripts\activate
pip install -r requirements.txt
```

Finetune (Restnet50)
```
make fine-tun
```


Visualize KPCA-CAM
```
make run
```