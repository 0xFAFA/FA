import torch
from torch.utils.data import Dataset
import os
import PIL.Image as Image
import numpy as np
import random

class IdDataset(Dataset):
    def __init__(self, root, dataset_name, num_shots):
        self.image_dir = os.path.join(root, dataset_name)
        self.shots = num_shots
        self.classnames = []

        self.train_path, self.train_targets = self.get_data('train')

        if dataset_name == 'TCGA12':
            self.test_path, self.test_targets = self.get_data('val')
        else:
            self.test_path, self.test_targets = self.get_data('test')


        # 构建fewshot数据集
        self.train_zip = list(zip(self.train_path, self.train_targets))

        split_dataset_by_label = {}
        for item in self.train_zip:
            if item[1] not in split_dataset_by_label:
                split_dataset_by_label[item[1]] = []
            split_dataset_by_label[item[1]].append(item)
        
        train_fewshot = []

        for label, items in split_dataset_by_label.items():
            if len(items) >= self.shots:
                sampled_items = random.sample(items, self.shots)
            else:
                sampled_items = items

            train_fewshot.extend(sampled_items)
                
        unzipped = zip(*train_fewshot)
        self.train_data_fewshot, self.train_targets_fewshot = map(list, unzipped)
        
        self.train_path = np.array(self.train_data_fewshot)
        self.train_targets = np.array(self.train_targets_fewshot)

        self.classnames = self.classnames # * 5

        print('origin len(classnames)', len(self.classnames))
        print('len(train_data_fewshot)', len(self.train_data_fewshot))


    def get_data(self, split_dir):
        cur_split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(cur_split_dir) if f.is_dir())
        train_path = []
        train_targets = []


        for label, folder in enumerate(folders):
            imnames = self.listdir_nohidden(os.path.join(cur_split_dir, folder))

            if split_dir == 'train':
                classname = folder
                classname = classname.replace('_', ' ')
                self.classnames.append(classname)

            for imname in imnames:
                impath = os.path.join(cur_split_dir, folder, imname)
                train_path.append(impath)
                train_targets.append(label)

        return np.array(train_path), np.array(train_targets)


    @staticmethod
    def listdir_nohidden(path, sort=False):
        """List non-hidden items in a directory.

        Args:
            path (str): directory path.
            sort (bool): sort the items.
        """
        items = [f for f in os.listdir(path) if not f.startswith(".")]
        if sort:
            items.sort()
        return items
