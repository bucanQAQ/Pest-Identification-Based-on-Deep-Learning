from rembg import remove
from tqdm import tqdm

import data_enhancement
import os


def remove_background(input_path,output_path):
    # Create directories if they don't exist
    if os.path.isdir(input_path):
        pass
    else:
        os.mkdir(input_path)
    if os.path.isdir(output_path):
        pass
    else:
        os.mkdir(output_path)

    # Get a list of class names in the input directory
    class_names=os.listdir((input_path))
    for class_name in class_names:
        img_path=os.path.join(input_path,class_name)
        save_path=os.path.join(output_path,class_name)
        img_names=os.listdir(img_path)

        # Loop through all images in the class directory
        for img_name in  tqdm(img_names,desc=class_name):
            with open(os.path.join(img_path,img_name),'rb') as i:
                with open(os.path.join(save_path,img_name),'wb') as o:
                    # Read image bytes and remove the background using the rembg library
                    input=i.read()
                    output=remove(input,
                                  alpha_matting=True,
                                  alpha_matting_background_threshold=0.9,  # Threshold for what is considered background
                                  alpha_matting_foreground_threshold=60, # Threshold for what is considered foreground
                                  alpha_matting_erode_size=10)          # The size of the erosion kernel used

                    # Write the resulting image bytes to file
                    o.write(output)


orig_dataset_path='./orig_dataset'
split_dataset_path='./split_dataset'
rembg_dataset_path='./rebg_dataset'
nore_dataset_path = './nore_dataset'
#split.data_set_split(orig_dataset_path,split_dataset_path)

# Get a list of class names in the split dataset directory
dataset_class_names=os.listdir(split_dataset_path)
rembg_dataset_path2='./rembg_dataset'
for dataset_class_name in dataset_class_names:
    split_path = os.path.join(split_dataset_path,dataset_class_name)
    rembg_path = os.path.join(rembg_dataset_path,dataset_class_name)
    nore_path = os.path.join(nore_dataset_path,dataset_class_name)
    rembg_path2 = os.path.join(rembg_dataset_path2,dataset_class_name)
    #remove_background(split_path, rembg_path)

    data_enhancement.data_enhance(split_path, nore_path)
