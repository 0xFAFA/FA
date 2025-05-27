import torch
from torch.utils.data import Dataset
import os
import PIL.Image as Image

class OodDataset(Dataset):
    def __init__(self, ood_dataset_path, transforms):
        self.transforms = transforms

        self.image_paths = []

        for filename in os.listdir(ood_dataset_path):
            full_path = os.path.join(ood_dataset_path, filename)
            self.image_paths.append(full_path)
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path)
        image = self.transforms(image)
        return image