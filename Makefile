MODEL ?= resnet50
EPOCHS ?= 1
LR ?= 0.1
CSV_PATH ?= ./dataset/data_Data_Entry_2017_v2020.csv
IMG_DIR ?= ./dataset/images/images_002/images
# IMG_DIR ?= ./dataset/images/images_002/bbox/00003333_002.png
TARGET_CLASS ?= 3
NUM_IMAGES ?= 5
TRAIN_LIST ?= ./dataset/data_train_val_list.txt 
TEST_LIST ?= ./dataset/data_test_list.txt
BATCH_SIZE ?= 32
OUTPUT_DIR ?= ./checkpoints
BBOX_DIR ?= ./dataset/data_BBox_List_2017.csv 
# MODEL_PATH ?= ./checkpoints/$(MODEL)/best_model.pth
TRAIN_SIZE ?= 100
TEST_SIZE ?= 20


fine-tune:
	python main.py \
		--model_name $(MODEL) \
		--csv_path $(CSV_PATH) \
		--img_dir $(IMG_DIR) \
		--train_list $(TRAIN_LIST) \
		--test_list $(TEST_LIST) \
		--num_epochs $(EPOCHS) \
		--lr $(LR) \
		--batch_size $(BATCH_SIZE) \
		--output_dir $(OUTPUT_DIR) \
		--train_size $(TRAIN_SIZE) --test_size $(TEST_SIZE)

fine-tune-all:
	make fine-tune MODEL=resnet50
	make fine-tune MODEL=vgg16
	make fine-tune MODEL=vit

visualize:
	python kpcacam.py --models_dir ./checkpoints --image_path $(IMG_DIR) --target_class $(TARGET_CLASS) --num_images $(NUM_IMAGES) --bbox_csv $(BBOX_DIR)

# List classes
get-classes:
	python kpcacam.py --models_dir ./checkpoints --image_path $(IMG_DIR) --target_class $(TARGET_CLASS) --list_classes

install:
	pip install -r requirements.txt

# predict:
# 	python predict.py --model_path $(MODEL_PATH) --image_path $(IMG_DIR) --threshold 0.2