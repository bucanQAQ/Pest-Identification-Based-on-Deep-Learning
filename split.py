import os
import random
import shutil
from shutil import copy2

def data_set_split(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.1, test_scale=0.1):
    # Create the target dataset folders for the train, validation and test sets.
    # If the folders already exist, then skip the step.
    print("Dataset partitioning")
    class_names = os.listdir(src_data_folder)
    split_names = ['train', 'valid', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)

        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    # Randomly shuffle the data and split it into train, validation and test sets.
    for class_name in class_names:
        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)

        train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
        val_folder = os.path.join(os.path.join(target_data_folder, 'valid'), class_name)
        test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
        train_stop_flag = current_data_length * train_scale
        val_stop_flag = current_data_length * (train_scale + val_scale)
        current_idx = 0
        train_num = 0
        val_num = 0
        test_num = 0
        for i in current_data_index_list:
            src_img_path = os.path.join(current_class_data_path, current_all_data[i])
            if current_idx <= train_stop_flag:
                # Copy the image to the train folder if the current index is less than or equal to train_stop_flag.
                copy2(src_img_path, train_folder)
                train_num = train_num + 1
            elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
                # Copy the image to the validation folder if the current index is greater than train_stop_flag but less than or equal to val_stop_flag.
                copy2(src_img_path, val_folder)
                val_num = val_num + 1
            else:
                # Copy the image to the test folder if the current index is greater than val_stop_flag.
                copy2(src_img_path, test_folder)
                test_num = test_num + 1

            current_idx = current_idx + 1
        # Print the number of images in each folder for each class.
        print("*********************************{}*************************************".format(class_name))
        print(
            "The {} are divided and finished in a ratio of {}:{}:{}, number:{}".format(class_name, train_scale, val_scale, test_scale, current_data_length))
        print("Train dataset{}：{}".format(train_folder, train_num))
        print("Vaild dataset{}：{}".format(val_folder, val_num))
        print("Test dataset{}：{}".format(test_folder, test_num))


data_set_split("./orig_dataset","./split_dataset")

