import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import shutil
import torch
import seaborn
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import clip.clip as clip
import PIL.Image as Image
import matplotlib.pyplot as plt
# from dataset.transforms import get_eval_transform
# from dataset.dataset import DataSet
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
# from utils.util import load_clip_to_cpu, get_class_dict
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
transform = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],  # clip mean
    std=[0.26862954, 0.26130258, 0.27577711],  # clip std
)
# _, classnames, _ = get_class_dict('classnames.txt')
# text_list = [f"a photo of a {text}" for text in classnames]
# text_list = clip.tokenize(text_list).cuda()

image_path = "./imagenet_test"
# save_path = "/data/runhe/storage/projects/CINO/vis-3"
image_list = os.listdir(image_path)
# val_transform = get_eval_transform()
# "ViT-B/16"
model, preprocess = clip.load('ViT-B/16')
# model = load_clip_to_cpu("ViT-B/16")
model.cuda()
# model.to(torch.float16)
model_dtype = model.dtype

score = []
abs_score = []
count = 0
for idx, file in enumerate(image_list):
    print(idx)
    image = Image.open(os.path.join(image_path, file))


    # label = file.split("-")[0]
    # label = int(label)
    image_ = preprocess(image).unsqueeze(0).cuda()
    image_ = image_.to(torch.float16)
    with torch.no_grad():
        _, featmap = model.visual(image_.type(model_dtype))
    # featmap = featmap[0, 1:]
    featmap = featmap / featmap.norm(dim=-1, keepdim=True)
    print(featmap.shape)
    featmap = featmap[0,:,:]

    featmap = featmap.detach().cpu().numpy()

    pca = PCA(n_components=3)
    pca.fit(featmap)
    pca_features = pca.transform(featmap)

    pca_features = pca_features.reshape(14,14, 3)
    plt.imshow(pca_features)
    plt.savefig(f"./imagenet_vis_res/test768_cls")
        