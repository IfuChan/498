from __future__ import print_function

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
# import skimage.io as sk
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset  # For custom datasets



def get_data_folder():
    """
    return the path to store the data
    """
    # data_folder = '/content/data/kaggle-pneumonia-jpg/'  #org
    data_folder = 'F:\\KD project\\Datasets\\RSNA Pneumonia Detection Dataset\\kaggle-pneumonia-jpg\\'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder





class CustomDataset(Dataset):
    
    def __init__(self, root, img_paths, labels, train=True, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, index):
        image = self.img_paths[index]+ '.jpg'
        image = Image.open(image).convert('RGB')

        label = self.labels[index][5]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
    
    def __len__(self):
        
        return len(self.img_paths)


def get_rsna_pneumonia_dataloaders(batch_size=128, num_workers=8):
    data_folder = get_data_folder()
    
    csv_path = data_folder+'stage_2_train_labels.csv'
    df = pd.read_csv(csv_path)
    
    # Dividing labels for train and test set
    train_labels, test_labels = train_test_split(df.values, test_size=0.2, random_state=1)
    # print(train_labels.shape)
    # print(test_labels.shape)
    
    
    # Preparing train and validation image paths
    train_f = data_folder+'stage_2_train_images_jpg'
    test_f = data_folder+'stage_2_test_images_jpg'
    
    train_img_paths = [os.path.join(train_f, image[0]) for image in train_labels]
    test_img_paths = [os.path.join(train_f, image[0]) for image in test_labels]
    
    # print(len(train_img_paths))
    # print(train_img_paths[0])
    # print(len(test_img_paths))
     
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.ToTensor()])
        
    train_set = CustomDataset(root=data_folder, img_paths=train_img_paths, labels=train_labels, train=True, transform=transform)
        
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    n_data = len(train_set)
    test_set = CustomDataset(root=data_folder, img_paths=test_img_paths, labels=test_labels, train=False, transform=transform)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=int(num_workers/2))
        
    return train_loader, test_loader
     



