import random
from torch.utils.data import Dataset
import numpy as np
import os
import zipfile 
import gdown
import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import re
import numpy as np
import torch

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


## Create a custom Dataset class
class CelebADataset(Dataset):
    def __init__(self, root_dir=os.path.join(CUR_DIR, 'data/celeba'), transform=None, attributes=None):
        """
        Args:
          root_dir (string): Directory with all the images
          transform (callable, optional): transform to be applied to each image sample
        """
        # Read names of images in the root directory
        
        # Path to folder with the dataset
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        dataset_folder = f'{root_dir}/img_align_celeba/'
        self.dataset_folder = os.path.abspath(dataset_folder)
        if not os.path.isdir(dataset_folder):
            # URL for the CelebA dataset
            download_url = 'https://drive.google.com/file/d/1BXMkEhC-Px0M1-2dnt2-_RI4iwl-0_3-/view?usp=sharing'
            # Path to download the dataset to
            download_path = f'{root_dir}/img_align_celeba.zip'
            # Download the dataset from google drive
            gdown.download(download_url, download_path, quiet=False, fuzzy=True)

            # os.makedirs(dataset_folder)

            # Unzip the downloaded file 
            with zipfile.ZipFile(download_path, 'r') as ziphandler:
                ziphandler.extractall(root_dir)
        image_names = os.listdir(self.dataset_folder)

        self.transform = transform 
        image_names = natsorted(image_names)
        
        self.filenames = []
        self.annotations = []
        with open(f'{root_dir}/list_attr_celeba.txt') as f:
            lines = f.readlines()
            attr_names = lines[1].strip().split()
            if attributes is not None:
                index = {}
                for i, el in enumerate(attr_names):
                    try:
                        c = attributes.index(el)
                        index[i] = c
                    except ValueError:
                        continue
                self.header = attributes
                assert(len(index) == len(attributes))
            else:
                self.header = attr_names
                index = list(range(len(self.header)))
            
            for i, line in enumerate(lines[2:]):
                line = re.sub(' *\n', '', line)
                values = re.split(' +', line)
                filename = values[0]
                relevant_attributes = [-1] * len(index)
                is_ok = False
                for j, v in enumerate(values[1:]):
                    if j in index and int(v) == 1:
                        is_ok = True
                        relevant_attributes[index[j]] = 1
                if not is_ok:
                    continue
                self.filenames.append(filename)
                self.annotations.append(relevant_attributes)

                    
        self.annotations = np.array(self.annotations)    
              
    def __len__(self): 
        return len(self.filenames)

    def __getitem__(self, idx):
        # Get the path to the image 
        img_name = self.filenames[idx]
        img_path = os.path.join(self.dataset_folder, img_name)
        img_attributes = self.annotations[idx] # convert all attributes to zeros and ones
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)
        return img, {'filename': img_name, 'idx': idx, 'attributes': torch.tensor(img_attributes).long()}


class BiDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.domain2idx = [[] for i in range(len(dataset.header))]
        for did in range(len(self.domain2idx)):
            good_images = []
            for i in range(len(dataset)):
                if dataset.annotations[i, did] == 1:
                    good_images.append(i)
            self.domain2idx[did] = good_images
        self.dataset = dataset 

    def __len__(self):
        return len(self.domain2idx)
    
    def __getitem__(self, idx):
        i1, i2= random.sample(self.domain2idx[idx], 2)
        image1 = self.dataset[i1][0]
        image2 = self.dataset[i2][0]
        return image1, image2, idx