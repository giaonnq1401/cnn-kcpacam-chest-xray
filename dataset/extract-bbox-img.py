import pandas as pd
import os
import shutil

def extract_bbox_images(csv_path, source_folder, destination_folder):
    """
    Extract images with bounding boxes from the original dataset and save them to a new folder.
    
    Args:
        csv_path (str): Path to the CSV file containing bounding box information
        source_folder (str): Folder containing the original image
        destination_folder (str): Destination folder to save images with bounding box
    """
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    os.makedirs(destination_folder, exist_ok=True)
    
    # Get a list of image files with bounding boxes
    bbox_images = df['Image Index'].unique()
    
    # Copy image files to new folder
    for img_name in bbox_images:
        source_path = os.path.join(source_folder, img_name)
        dest_path = os.path.join(destination_folder, img_name)
        
        try:
            shutil.copy2(source_path, dest_path)
        except FileNotFoundError:
            print(f"Not found: {img_name}")
        except Exception as e:
            print(f"Error when copying {img_name}: {str(e)}")
    
    print(f"Copied {len(bbox_images)} images have bounding box to {destination_folder}")

csv_file = "./dataset/data_BBox_List_2017.csv"
source_dir = "./dataset/images/images_002/images"
dest_dir = "./dataset/images/bbox"

extract_bbox_images(csv_file, source_dir, dest_dir)