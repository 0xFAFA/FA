import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

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
from mydataset import build_dataset
from mydataset.utils import build_data_loader


from ood_utils.ood_tool import fpr_recall
from ood_utils.ood_tool import auc
from ood_utils.ood_tool import get_measures
import numpy as np

dataset_root = "/home/enroll2024/xinhua/Datasets"



template = 'exactly a photo of a {}.'



device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)  #ViT-B/16  RN50



batch_size = 500
shuffle = True
num_workers = 4

few_shot_dataset_stage2 = build_dataset('imagenet', dataset_root, 1)
test_dataloader = build_data_loader(data_source=few_shot_dataset_stage2.test, batch_size=batch_size, is_train=False, tfm=preprocess, shuffle=False)


# test_dataset = datasets.ImageFolder(os.path.join(dataset_root, 'Skin_OOD', 'SDB198', '42+156', 'test'), transform=preprocess)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,)


labels = few_shot_dataset_stage2.classnames

# text = clip.tokenize([template + l for l in labels]).to(device)
text = clip.tokenize([template.format(l) for l in labels]).to(device)
print(type(text),text.shape)



with torch.no_grad():

    correct_items = 0
    total_items = 0
    topk = 1

    for images, targets in tqdm(test_dataloader):
        images = images.to(device)
        targets = targets.to(device)
        
        image_features,_ = model.encode_image(images)
        text_features = model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits =  model.logit_scale * image_features @ text_features.t()  #

        pred = logits.topk(topk, 1, True, True)[1].t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        cur_correct_items = int(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())

        correct_items += cur_correct_items
        total_items += targets.shape[0]

    acc = 100 * correct_items / total_items
    print(f'Accuracy: {acc:.2f}%')



