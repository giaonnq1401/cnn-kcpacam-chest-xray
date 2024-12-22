fine-tune:
	python main.py --csv_path ./dataset/data_Data_Entry_2017_v2020.csv --img_dir ./dataset/images/example/images --batch_size 32 --num_epochs 1 --lr 0.005 --output_dir ./checkpoints

run:
	python kpcacam.py --model_path ./checkpoints/best_model.pth --image_path ./dataset/images/example/images --target_class 0 --output_dir outputs