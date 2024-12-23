MODEL ?= resnet50
EPOCHS ?= 1
LR ?= 0.005
CSV_PATH ?= ./dataset/data_Data_Entry_2017_v2020.csv
IMG_DIR ?= ./dataset/images/example/images

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

run:
	python kpcacam.py --model_path ./checkpoints/best_model.pth --image_path ./dataset/images/example/images --target_class 0 --output_dir outputs

install:
	pip install -r requirements.txt
