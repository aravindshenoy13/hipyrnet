import wandb
from .dataloader import SICEGradTrain,SICEGradTest,SICEGradVal, SICEMixTrain,SICEMixVal, LOLTrain, get_training_augmentation, get_validation_augmentation, get_transform, SICETrainDataset
import torch
from tqdm import tqdm as tqdm
import albumentations as A
import os
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from .models.lptn_model import LPTNModel
import albumentations as albu
import cv2
import pandas as pd

def get_training_augmentation_forsice():
    return albu.Compose([
        albu.Resize(608, 896, interpolation=cv2.INTER_LINEAR),  # Resize images to a specific shape
        ToTensorV2()  # Convert images to PyTorch tensors
    ], additional_targets={'mask': 'image'})

def train(epochs,
          batch_size,
          dset,
          root_dir,
          nrb_high = 5,
          nrb_low = 3,
          kernel_loss_weight=1,
          device='cuda',
          lr=1e-4,
          loss_weight = 2000,
          gan_type = 'standard',
          use_hypernet = True
          ):
    
    transform = get_transform(dataset='grad')
    if(dset=='mix'):
        train_dataset = SICEMixTrain(root_dir=root_dir, augmentation= get_training_augmentation())
        val_dataset = SICEMixVal(root_dir = root_dir, augmentation= get_validation_augmentation())

    elif(dset=='grad'):
        train_dataset = SICEGradTrain(root_dir=root_dir, augmentation= get_training_augmentation())
        val_dataset = SICEGradVal(root_dir = root_dir, augmentation= get_validation_augmentation())

    elif(dset=='lol'):
        subdirectories = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, name))]
        high = subdirectories[1]
        low = subdirectories[0]
        print(high, low)
        train_dataset = LOLTrain(high_res_folder=high, low_res_folder=low, flag=0, augmentation=get_training_augmentation())
        val_dataset = LOLTrain(high_res_folder=high, low_res_folder=low, flag=1, augmentation=get_training_augmentation())

    elif(dset == 'sice'):
        train_dataset = SICETrainDataset(root_dir = root_dir, augmentation=get_training_augmentation(), mode = 'train')
        val_dataset = SICETrainDataset(root_dir = root_dir, augmentation= get_training_augmentation(), mode ='val')
    
    elif(dset == 'sicewg'):
        train_dataset = SICETrainDataset(root_dir = root_dir, augmentation=get_training_augmentation())
        val_dataset = SICEGradVal(root_dir = '/sice-grad-and-sice-mix/SICEGM', augmentation= get_validation_augmentation())
    
    elif(dset == 'sicewm'):
        train_dataset = SICETrainDataset(root_dir = root_dir, augmentation=get_training_augmentation())
        val_dataset = SICEMixVal(root_dir = '/sice-grad-and-sice-mix/SICEGM', augmentation= get_validation_augmentation())
    
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    lptn_model = LPTNModel(loss_weight, kernel_loss_weight, device, lr, gan_type=gan_type, use_hypernet=use_hypernet, nrb_high=nrb_high, nrb_low=nrb_low)

    max_ssim = 0
    max_psnr = 0
    logger = {'epoch': 0,'train_loss': 0, 'train_psnr': 0, 'train_ssim': 0,'train_lpips': 0, 'val_ssim': 0, 'val_psnr': 0, 'val_lpips' : 0, 'test_ssim': 0, 'test_psnr': 0}    
    for i in range(0, epochs):
        total_loss = []
        kernel_loss =[]
        psnr_train,ssim_train = 0,0

        with tqdm(
            train_loader,
            desc = "Training Progress"
        ) as loader:
            for iteration,batch_data in enumerate(loader):
                x,y = batch_data
                lptn_model.feed_data(x,y)

                loss_iter,kernel_los_iter,psnr_train_iter,ssim_train_iter, lpips_train_iter = lptn_model.optimize_parameters(iteration)

                total_loss.append(loss_iter)
                kernel_loss.append(kernel_los_iter)
                psnr_train = psnr_train + psnr_train_iter
                ssim_train = ssim_train + ssim_train_iter
                lpips_train = lpips_train = lpips_train_iter
                
    
        psnr_train /= (iteration+1)
        ssim_train /= (iteration+1)
        lpips_train /= (iteration+1)
        avg_loss = sum(total_loss)/len(total_loss)
        avg_kernel_loss = sum(kernel_loss)/len(kernel_loss)
        
        print(f'TRAIN PSNR {psnr_train}')
        print(f'TRAIN SSIM {ssim_train}')
        print(f'TRAIN LPIPS {lpips_train}')
    
        psnr_val, ssim_val, lpips_val = lptn_model.nondist_validation(valid_loader)

        logger['train_loss'] = avg_loss
        logger['kernel_loss'] = avg_kernel_loss
        logger['train_psnr'] = psnr_train
        logger['train_ssim'] = ssim_train
        logger['val_psnr'] = psnr_val
        logger['val_lpips'] = lpips_val
        logger['val_ssim'] = ssim_val
        logger['epoch'] = i
        
        if max_ssim <= logger['val_ssim']:
            max_ssim = logger['val_ssim']
            max_psnr = logger['val_psnr']
            wandb.config.update({'max_ssim': max_ssim, 'max_psnr': max_psnr, 'best_epoch': i}, allow_val_change=True)
            lptn_model.save('./best_model')
         
        wandb.log(logger)
        torch.cuda.empty_cache()


def train_model(configs):
    train(
        configs['epochs'], 
        configs['batch_size'],
        configs['dset'],
        configs['root_dir'],
        configs['nrb_high'],
        configs['nrb_low'],
        configs['kernel_loss_weight'],
        configs['device'], 
        configs['lr'],
        configs['loss_weight'],
        configs['gan_type'],
        configs['use_hypernet']
        )
    
def get_training_transform():
    train_transform = transforms.Compose([
        transforms.Resize((608, 896)),             
        transforms.RandomVerticalFlip(p=0.5),     
        transforms.ToTensor(),                     
    ])
    return train_transform