import torch
from torch.utils.data import Dataset
import os
import PIL.Image as Image

class CustomDataset(Dataset):
    def __init__(self, image_paths, label_indices, transforms):
        self.image_paths = image_paths
        self.label_indices = label_indices
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_index = self.label_indices[index]
        image = Image.open(image_path)
        image = self.transforms(image)
        return image, label_index