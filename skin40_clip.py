import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

import clip
from PIL import Image
from tqdm import tqdm


from mydataset.skin40 import Skin40
from mydataset.custom_dataset import CustomDataset

from ood_utils.ood_tool import fpr_recall
from ood_utils.ood_tool import auc
from ood_utils.ood_tool import get_measures
import numpy as np

from mydataset import build_dataset
from mydataset.utils import build_data_loader

dataset_root = "/home/enroll2024/xinhua/Datasets"



# template = 'a photo of a {}.'
template = 'This is a medical pathology image of {}'
# template = 'A photo of a {}, a type of skin disease'



device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)  #ViT-B/16




# skin40 = Skin40()
# skin40.download_data(dataset_root)
# test_imgs = skin40.test_data
# test_targets = skin40.test_targets
# test_dataset = CustomDataset(test_imgs, test_targets, preprocess)



batch_size = 500
shuffle = True
num_workers = 4
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# test_dataset = datasets.ImageFolder(os.path.join(dataset_root, 'NCT-CRC-HE-100K', 'test'), transform=preprocess)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,)

test_dataset = datasets.ImageFolder(os.path.join(dataset_root, 'Skin_OOD', 'SDB198', '42+156', 'test'), transform=preprocess)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,)

# test_dataset = build_dataset('imagenet', dataset_root, 1)
# test_dataloader = build_data_loader(data_source=test_dataset.test, batch_size=batch_size, is_train=False, tfm=preprocess, shuffle=False)


labels = test_dataset.classes
# labels = test_dataset.classnames 
labels = [l.replace("_", " ") for l in labels]

print(labels)
print(len(labels))

# text = clip.tokenize([template + l for l in labels]).to(device)
text = clip.tokenize([template.format(l) for l in labels]).to(device)
print(type(text),text.shape)


# logit_result
id_conf = np.array([])
ood_conf = np.array([])


with torch.no_grad():

    correct_items = 0
    total_items = 0
    topk = 1
    T = 1.0

    for images, targets in tqdm(test_dataloader):
        images = images.to(device)
        targets = targets.to(device)
        
        
        logits_per_image, logits_per_text = model(images, text)


        logits_per_image /= 100.0
        smax_global = F.softmax(logits_per_image/T, dim=-1).cpu().numpy()
        mcm_global_score = np.max(smax_global, axis=1)
        id_conf = np.concatenate((id_conf, mcm_global_score))


        pred = logits_per_image.topk(topk, 1, True, True)[1].t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        cur_correct_items = int(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())

        correct_items += cur_correct_items
        total_items += targets.shape[0]

    acc = 100 * correct_items / total_items
    print(f'Accuracy: {acc:.2f}%')



    # OOD detection
    # testsetout = datasets.ImageFolder(root=os.path.join(dataset_root, 'LC25000', 'test'), transform=preprocess)
    testsetout = datasets.ImageFolder(root=os.path.join(dataset_root, 'Skin_OOD', 'SDB198', '42+156', 'ood'), transform=preprocess)
    # testsetout = datasets.ImageFolder(root=os.path.join(dataset_root, 'SUN'), transform=preprocess)

    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=100,
                                                shuffle=False, num_workers=4)
    
    for images, _ in tqdm(testloaderOut):
        images = images.to(device)
        logits_per_image, logits_per_text = model(images, text)
        logits_per_image /= 100.0
        smax_global = F.softmax(logits_per_image/T, dim=-1).cpu().numpy()
        mcm_global_score = np.max(smax_global, axis=1)
        ood_conf = np.concatenate((ood_conf, mcm_global_score))
    
        
    tpr = 0.95

    auroc, aupr, fpr = get_measures(id_conf, ood_conf, tpr)
    print(f'AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, FPR: {fpr:.4f}')



