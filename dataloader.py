import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch import Tensor
import pandas as pd
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_pickle(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get path from file
        img_path = os.path.join(self.img_dir, self.img_labels['path_to_img'].iloc[idx])
        # Open image from path
        image = Image.open(img_path)
        # Get context from file (encoded type)
        context = Tensor(self.img_labels['encoded_type'].iloc[idx])
        # Apply transformations if needed
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            context = self.target_transform(context)
        return image, context