import json
import os
from urllib.request import urlopen
from PIL import Image
import torch
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

from mydataset.skin40 import Skin40
from mydataset.custom_dataset import CustomDataset

from ood_utils.ood_tool import fpr_recall
from ood_utils.ood_tool import auc
from ood_utils.ood_tool import get_measures
import numpy as np


dataset_root = "/home/enroll2024/xinhua/Datasets"



# # Download the model and config files
# hf_hub_download(
#     repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
#     filename="open_clip_pytorch_model.bin",
#     local_dir="checkpoints"
# )
# hf_hub_download(
#     repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
#     filename="open_clip_config.json",
#     local_dir="checkpoints"
# )


# Load the model and config files
model_name = "biomedclip_local"

with open("/home/enroll2024/xinhua/Storage/test_bio_clip/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_config.json", "r") as f:
    config = json.load(f)
    model_cfg = config["model_cfg"]
    preprocess_cfg = config["preprocess_cfg"]


if (not model_name.startswith(HF_HUB_PREFIX)
    and model_name not in _MODEL_CONFIGS
    and config is not None):
    _MODEL_CONFIGS[model_name] = model_cfg

tokenizer = get_tokenizer(model_name)
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("./")

model, _, preprocess = create_model_and_transforms(
    model_name=model_name,
    pretrained="/home/enroll2024/xinhua/Storage/test_bio_clip/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin",
    **{f"image_{k}": v for k, v in preprocess_cfg.items()},
)


print(preprocess)

# Zero-shot image classification
# template = 'this is a photo of {}'
template = 'This is a medical pathology image of {}'
# template = 'A photo of a {}, a type of skin disease'




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()


# skin40 = Skin40()
# skin40.download_data(dataset_root)
# test_imgs = skin40.test_data
# test_targets = skin40.test_targets
# test_dataset = CustomDataset(test_imgs, test_targets, preprocess)


batch_size = 40
shuffle = True
num_workers = 4
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# test_dataset = datasets.ImageFolder(os.path.join(dataset_root, 'NCT-CRC-HE-100K', 'test'), transform=preprocess)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,)


test_dataset = datasets.ImageFolder(os.path.join(dataset_root, 'Skin_OOD', 'SDB198', '102+96', 'test'), transform=preprocess)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,)


context_length = 256
labels = test_dataset.classes
labels = [l.replace("_", " ") for l in labels]
# texts = tokenizer([template.format for l in labels], context_length=context_length).to(device)
texts = tokenizer([template.format(l) for l in labels], context_length=context_length).to(device)


# logit_result
id_conf = np.array([])
ood_conf = np.array([])


with torch.no_grad():

    correct_items = 0
    total_items = 0
    topk = 1
    T = 1.0

    for images, targets in test_dataloader:
        images = images.to(device)
        targets = targets.to(device)
        image_features, text_features, logit_scale = model(images, texts)

        logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)

        logits /= 100.0
        smax_global = F.softmax(logits/T, dim=-1).cpu().numpy()
        mcm_global_score = np.max(smax_global, axis=1)
        id_conf = np.concatenate((id_conf, mcm_global_score))


        pred = logits.topk(topk, 1, True, True)[1].t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        cur_correct_items = int(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())

        correct_items += cur_correct_items
        total_items += targets.shape[0]

    acc = 100 * correct_items / total_items
    print(f'Accuracy: {acc:.2f}%')


    # OOD detection
    # testsetout = datasets.ImageFolder(root=os.path.join(dataset_root, 'LC25000', 'test'), transform=preprocess)
    testsetout = datasets.ImageFolder(root=os.path.join(dataset_root, 'Skin_OOD', 'SDB198', '102+96', 'ood'), transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=100,
                                                shuffle=False, num_workers=4)
    
    for images, _ in testloaderOut:
        images = images.to(device)
        image_features, text_features, logit_scale = model(images, texts)
        logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
        logits /= 100.0
        smax_global = F.softmax(logits/T, dim=-1).cpu().numpy()
        mcm_global_score = np.max(smax_global, axis=1)
        ood_conf = np.concatenate((ood_conf, mcm_global_score))


    
    tpr = 0.95

    auroc, aupr, fpr = get_measures(id_conf, ood_conf, tpr)
    print(f'AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, FPR: {fpr:.4f}')
