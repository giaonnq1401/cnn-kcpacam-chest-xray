MODEL ?= resnet50
EPOCHS ?= 1
LR ?= 0.005
CSV_PATH ?= ./dataset/data_Data_Entry_2017_v2020.csv
IMG_DIR ?= ./dataset/images/example/images
TARGET_CLASS ?= 0
NUM_IMAGES ?= 5

fine-tune:
	python main.py \
		--model_name $(MODEL) \
		--csv_path $(CSV_PATH) \
		--img_dir $(IMG_DIR) \
		--num_epochs $(EPOCHS) \
		--lr $(LR)

fine-tune-all:
	make fine-tune MODEL=resnet50
	make fine-tune MODEL=vgg16
	make fine-tune MODEL=vit

visualize:
	python kpcacam.py --models_dir ./checkpoints --image_path $(IMG_DIR) --target_class $(TARGET_CLASS) --num_images $(NUM_IMAGES)

# List classes
get-classes:
	python kpcacam.py --models_dir ./checkpoints --image_path $(IMG_DIR) --target_class $(TARGET_CLASS) --list_classes

install:
	pip install -r requirements.txt
