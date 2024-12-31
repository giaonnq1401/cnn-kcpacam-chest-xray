IMG_DIR ?= ./dataset/images/bbox
NUM_IMAGES ?= 10
BBOX_DIR ?= ./dataset/data_BBox_List_2017.csv 

fine-tune:
	python chest-xray-training.py

visualize:
	python kpcacam.py --image_path $(IMG_DIR) --num_images $(NUM_IMAGES) --bbox_csv $(BBOX_DIR)

install:
	pip install -r requirements.txt