    
from torch.utils.data import Dataset as BaseDataset
import cv2
import pickle
import os
import numpy as np
import albumentations as albu
import torch
import pandas as pd
from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader, Subset
import random
from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data import Dataset as BaseDataset
import cv2
import pickle
import os
import numpy as np

import torch
import pandas as pd

class SICEGradTrain(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'SICE_Grad')
        self.label_dir = os.path.join(root_dir, 'SICE_Reshape')
        self.file_names = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.jpg')]
        self.transform = transform
        self.augmentation = augmentation
        self.file_names = self.file_names[:480]

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_name = os.path.join(self.input_dir, self.file_names[idx])
        base_name = os.path.splitext(self.file_names[idx])[0]
        
        label_name_jpg_upper = os.path.join(self.label_dir, base_name + '.JPG')
        label_name_jpg_lower = os.path.join(self.label_dir, base_name + '.jpg')
        label_name_png = os.path.join(self.label_dir, base_name + '.PNG')
        
        if os.path.exists(label_name_jpg_upper):
            label_name = label_name_jpg_upper
        elif os.path.exists(label_name_jpg_lower):
            label_name = label_name_jpg_lower
        elif os.path.exists(label_name_png):
            label_name = label_name_png
        else:
            raise FileNotFoundError(f"No corresponding label image found for {input_name}")
        
        #READING IMAGE
        label_image = cv2.imread(label_name)
        input_image = cv2.imread(input_name)
        
        #CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        #ROTATING PORTRAIT IMAGES
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #RESIZE + FLIP AUGMENTATION
        if self.augmentation:
            augmented = self.augmentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']
        
        #NORMALISATION
        input_image = input_image / 255.0
        label_image = label_image / 255.0
        
        #STANDARDISATION
        # if self.transform:
        #     input_image = self.transform(image=input_image)['image']
        
        #CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)
        
        return input_image, label_image
    
class SICEGradVal(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'SICE_Grad')
        self.label_dir = os.path.join(root_dir, 'SICE_Reshape')
        self.file_names = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.jpg')]
        self.transform = transform
        self.augmentation = augmentation
        self.file_names = self.file_names[480:530]

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_name = os.path.join(self.input_dir, self.file_names[idx])
        base_name = os.path.splitext(self.file_names[idx])[0]
        
        label_name_jpg_upper = os.path.join(self.label_dir, base_name + '.JPG')
        label_name_jpg_lower = os.path.join(self.label_dir, base_name + '.jpg')
        label_name_png = os.path.join(self.label_dir, base_name + '.PNG')
        
        if os.path.exists(label_name_jpg_upper):
            label_name = label_name_jpg_upper
        elif os.path.exists(label_name_jpg_lower):
            label_name = label_name_jpg_lower
        elif os.path.exists(label_name_png):
            label_name = label_name_png
        else:
            raise FileNotFoundError(f"No corresponding label image found for {input_name}")
        
        #READING IMAGE
        label_image = cv2.imread(label_name)
        input_image = cv2.imread(input_name)
        
        #CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        #ROTATING PORTRAIT IMAGES
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #RESIZE + FLIP AUGMENTATION
        if self.augmentation:
            augmented = self.augmentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']
        
        #NORMALISATION
        input_image = input_image / 255.0
        label_image = label_image / 255.0
        
        #STANDARDISATION
        # if self.transform:
        #     input_image = self.transform(image=input_image)['image']
        
        #CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)
        
        return input_image, label_image
    
class SICEGradTest(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'SICE_Grad')
        self.label_dir = os.path.join(root_dir, 'SICE_Reshape')
        self.file_names = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.jpg')]
        self.transform = transform
        self.augmentation = augmentation
        self.file_names = self.file_names[530:]

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_name = os.path.join(self.input_dir, self.file_names[idx])
        base_name = os.path.splitext(self.file_names[idx])[0]
        
        label_name_jpg_upper = os.path.join(self.label_dir, base_name + '.JPG')
        label_name_jpg_lower = os.path.join(self.label_dir, base_name + '.jpg')
        label_name_png = os.path.join(self.label_dir, base_name + '.PNG')
        
        if os.path.exists(label_name_jpg_upper):
            label_name = label_name_jpg_upper
        elif os.path.exists(label_name_jpg_lower):
            label_name = label_name_jpg_lower
        elif os.path.exists(label_name_png):
            label_name = label_name_png
        else:
            raise FileNotFoundError(f"No corresponding label image found for {input_name}")
        
        #READING IMAGE
        label_image = cv2.imread(label_name)
        input_image = cv2.imread(input_name)
        
        #CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        #ROTATING PORTRAIT IMAGES
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #RESIZE + FLIP AUGMENTATION
        if self.augmentation:
            augmented = self.augmentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']
        
        #NORMALISATION
        input_image = input_image / 255.0
        label_image = label_image / 255.0
        
        #STANDARDISATION
        # if self.transform:
        #     input_image = self.transform(image=input_image)['image']
        
        #CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)
        
        return input_image, label_image
    
class SICEMixTrain(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'SICE_Mix')
        self.label_dir = os.path.join(root_dir, 'SICE_Reshape')
        self.file_names = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.jpg')]
        self.transform = transform
        self.augmentation = augmentation
        self.file_names = self.file_names[:480]

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_name = os.path.join(self.input_dir, self.file_names[idx])
        base_name = os.path.splitext(self.file_names[idx])[0]
        
        label_name_jpg_upper = os.path.join(self.label_dir, base_name + '.JPG')
        label_name_jpg_lower = os.path.join(self.label_dir, base_name + '.jpg')
        label_name_png = os.path.join(self.label_dir, base_name + '.PNG')
        
        if os.path.exists(label_name_jpg_upper):
            label_name = label_name_jpg_upper
        elif os.path.exists(label_name_jpg_lower):
            label_name = label_name_jpg_lower
        elif os.path.exists(label_name_png):
            label_name = label_name_png
        else:
            raise FileNotFoundError(f"No corresponding label image found for {input_name}")
        
        #READING IMAGE
        label_image = cv2.imread(label_name)
        input_image = cv2.imread(input_name)
        
        #CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        #ROTATING PORTRAIT IMAGES
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #RESIZE + FLIP AUGMENTATION
        if self.augmentation:
            augmented = self.augmentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']
        
        #NORMALISATION
        input_image = input_image / 255.0
        label_image = label_image / 255.0
        
        #STANDARDISATION
        # if self.transform:
        #     input_image = self.transform(image=input_image)['image']
        
        #CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)
        
        return input_image, label_image
    
class SICEMixVal(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'SICE_Mix')
        self.label_dir = os.path.join(root_dir, 'SICE_Reshape')
        self.file_names = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.jpg')]
        self.transform = transform
        self.augmentation = augmentation
        self.file_names = self.file_names[480:530]

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_name = os.path.join(self.input_dir, self.file_names[idx])
        base_name = os.path.splitext(self.file_names[idx])[0]
        
        label_name_jpg_upper = os.path.join(self.label_dir, base_name + '.JPG')
        label_name_jpg_lower = os.path.join(self.label_dir, base_name + '.jpg')
        label_name_png = os.path.join(self.label_dir, base_name + '.PNG')
        
        if os.path.exists(label_name_jpg_upper):
            label_name = label_name_jpg_upper
        elif os.path.exists(label_name_jpg_lower):
            label_name = label_name_jpg_lower
        elif os.path.exists(label_name_png):
            label_name = label_name_png
        else:
            raise FileNotFoundError(f"No corresponding label image found for {input_name}")
        
        #READING IMAGE
        label_image = cv2.imread(label_name)
        input_image = cv2.imread(input_name)
        
        #CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        #ROTATING PORTRAIT IMAGES
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #RESIZE + FLIP AUGMENTATION
        if self.augmentation:
            augmented = self.augmentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']
        
        #NORMALISATION
        input_image = input_image / 255.0
        label_image = label_image / 255.0
        
        #STANDARDISATION
        # if self.transform:
        #     input_image = self.transform(image=input_image)['image']
        
        #CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)
        
        return input_image, label_image
    
class SICEMixTest(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'SICE_Mix')
        self.label_dir = os.path.join(root_dir, 'SICE_Reshape')
        self.file_names = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.jpg')]
        self.transform = transform
        self.augmentation = augmentation
        self.file_names = self.file_names[530:]

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_name = os.path.join(self.input_dir, self.file_names[idx])
        base_name = os.path.splitext(self.file_names[idx])[0]
        
        label_name_jpg_upper = os.path.join(self.label_dir, base_name + '.JPG')
        label_name_jpg_lower = os.path.join(self.label_dir, base_name + '.jpg')
        label_name_png = os.path.join(self.label_dir, base_name + '.PNG')
        
        if os.path.exists(label_name_jpg_upper):
            label_name = label_name_jpg_upper
        elif os.path.exists(label_name_jpg_lower):
            label_name = label_name_jpg_lower
        elif os.path.exists(label_name_png):
            label_name = label_name_png
        else:
            raise FileNotFoundError(f"No corresponding label image found for {input_name}")
        
        #READING IMAGE
        label_image = cv2.imread(label_name)
        input_image = cv2.imread(input_name)
        
        #CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        #ROTATING PORTRAIT IMAGES
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #RESIZE + FLIP AUGMENTATION
        if self.augmentation:
            augmented = self.augmentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']
        
        #NORMALISATION
        input_image = input_image / 255.0
        label_image = label_image / 255.0
        
        #STANDARDISATION
        # if self.transform:
        #     input_image = self.transform(image=input_image)['image']
        
        #CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)
        
        return input_image, label_image
    
class LOLTrain(BaseDataset):
    def __init__(self, high_res_folder, low_res_folder, flag,  transform=None, augmentation=None):
        self.image_pairs = list_image_paths(high_res_folder, low_res_folder, flag)
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        high_res_path, low_res_path = self.image_pairs[idx]

        high_res_image = cv2.imread(high_res_path)
        low_res_image = cv2.imread(low_res_path)

        # totensor = ToTensorV2()
        
        if self.augmentation:
            augmented = self.augmentation(image=high_res_image, image1=low_res_image)
            high_res_image, low_res_image = augmented['image'], augmented['image1']

        low_res_image = low_res_image / 255.0
        high_res_image = high_res_image / 255.0

        low_res_image = torch.tensor(low_res_image, dtype=torch.float32).permute(2, 0, 1)
        high_res_image = torch.tensor(high_res_image, dtype=torch.float32).permute(2, 0, 1)
            
        return low_res_image.float(),high_res_image.float()
    
def get_training_augmentation():
    train_transform = [
        albu.Resize(608, 896, interpolation=cv2.INTER_LINEAR, always_apply=True),
        albu.VerticalFlip(p=0.5),
    ]
    return albu.Compose(train_transform, additional_targets={'image1':'image'}, is_check_shapes=False)
    
def get_validation_augmentation():
    test_transform = [
        albu.Resize(608, 896, interpolation=cv2.INTER_LINEAR, always_apply=True),
    ]   
    return albu.Compose(test_transform, additional_targets={'image1': 'image'}, is_check_shapes=False)

def get_transform(dataset):
        if dataset == 'grad':   
            mean = [0.41441402, 0.41269127, 0.37940571]
            std = [0.33492465, 0.33443474, 0.33518072]
        if dataset == 'mix':
            mean = [0.41268688, 0.41124236, 0.37886961]
            std = [0.33789958, 0.33786919, 0.33946865]
        return albu.Compose([
            albu.Normalize(mean=mean, std=std),
    ])

def list_image_paths(high_res_folder, low_res_folder,flag):
    high_res_files = sorted([f for f in os.listdir(high_res_folder) if os.path.isfile(os.path.join(high_res_folder, f))])
    
    low_res_files = sorted([f for f in os.listdir(low_res_folder) if os.path.isfile(os.path.join(low_res_folder, f))])
    
    pairs = []
    for hr_file in high_res_files:
        if hr_file in low_res_files:
            hr_path = os.path.join(high_res_folder, hr_file)
            lr_path = os.path.join(low_res_folder, hr_file)
            pairs.append((hr_path, lr_path))
    if flag == 0:
        return pairs[:440]
    if flag == 1:
        return pairs[440:]
    if flag == 2:
        return pairs 
    
class SICETrainDataset(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None, 
                 val_indices=None, test_indices=None, mode="train"):
        self.root_dir = root_dir
        self.transform = transform
        self.augmentation = augmentation
        self.data = []
        self.val_indices = set([24, 25, 26, 27, 35, 36, 40, 41, 42, 43, 44, 45, 70, 71, 72, 73, 74, 80, 81, 82, 83])
        self.test_indices = set([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 28, 31, 33, 34, 37, 38, 39, 46, 47, 48, 49, 50, 51, 52, 55, 69, 75, 76, 77, 78, 79, 100, 101, 102, 103])
        self.mode = mode

        for part in ["Dataset_Part1/Dataset_Part1", "Dataset_Part2/Dataset_Part2"]:
            part_path = os.path.join(root_dir, part)
            label_path = os.path.join(part_path, "Label")

            for folder in os.listdir(part_path):
                folder_path = os.path.join(part_path, folder)
                
                if folder.isdigit() and os.path.isdir(folder_path):
                    # Check for either .png or .jpg label file
                    label_file = None
                    for ext in [".PNG", ".JPG", ".JPEG"]:
                        potential_label = os.path.join(label_path, f"{folder}{ext}")
                        if os.path.exists(potential_label):
                            label_file = potential_label
                            break
                    
                    if not label_file:
                        continue  # Skip if no valid label file found
                    
                    # Add valid image-label pairs
                    folder_number = int(folder)
                    for img_file in os.listdir(folder_path):
                        img_path = os.path.join(folder_path, img_file)
                        if img_file.endswith(('.PNG', '.JPG', '.JPEG')):
                            # Only consider indices for Dataset_Part1
                            if part == "Dataset_Part1/Dataset_Part1":
                                self.data.append((folder_number, img_path, label_file))
                            else:
                                # Exclude indices for Dataset_Part2 in split filtering
                                self.data.append((None, img_path, label_file))

        # Filter based on mode
        if mode == "train":
            self.data = [x[1:] for x in self.data if x[0] is None or (x[0] not in self.val_indices and x[0] not in self.test_indices)]
        elif mode == "val":
            self.data = [x[1:] for x in self.data if x[0] in self.val_indices]
        elif mode == "test":
            self.data = [x[1:] for x in self.data if x[0] in self.test_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label_path = self.data[idx]

        label_image = cv2.imread(label_path)
        input_image = cv2.imread(img_path)
        
        # CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # ROTATING PORTRAIT IMAGES
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # RESIZE + FLIP AUGMENTATION
        if self.augmentation:
            augmented = self.augmentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']
        
        # NORMALIZATION
        input_image = input_image / 255.0
        label_image = label_image / 255.0
        
        # CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)

        return input_image, label_image