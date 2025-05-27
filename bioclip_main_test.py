import json
import os
from urllib.request import urlopen
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

from skin40 import Skin40

skin40 = Skin40()
skin40.download_data("/home/enroll2024/xinhua/Datasets")


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


# Zero-shot image classification
template = 'this is a photo of '
labels = [
    'adenocarcinoma histopathology',
    'brain MRI',
    'covid line chart',
    'squamous cell carcinoma histopathology',
    'immunohistochemistry histopathology',
    'bone X-ray',
    'chest X-ray',
    'pie chart',
    'hematoxylin and eosin histopathology'
]
labels_sk40 = skin40.classnames

# dataset_url = 'https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/example_data/biomed_image_classification_example_data/'
test_imgs = [
    'squamous_cell_carcinoma_histopathology.jpeg',
    'H_and_E_histopathology.jpg',
    'bone_X-ray.jpg',
    'adenocarcinoma_histopathology.jpg',
    'covid_line_chart.png',
    'IHC_histopathology.jpg',
    'chest_X-ray.jpg',
    'brain_MRI.jpg',
    'pie_chart.png'
]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

context_length = 256

# images = torch.stack([preprocess(Image.open(urlopen(dataset_url + img))) for img in test_imgs]).to(device)
images = torch.stack([preprocess(Image.open(  os.path.join( './test_bio_img', img)  )) for img in test_imgs]).to(device)



texts = tokenizer([template + l for l in labels], context_length=context_length).to(device)
with torch.no_grad():
    image_features, text_features, logit_scale = model(images, texts)

    logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)

    logits = logits.cpu().numpy()
    sorted_indices = sorted_indices.cpu().numpy()

top_k = -1

for i, img in enumerate(test_imgs):
    pred = labels[sorted_indices[i][0]]

    top_k = len(labels) if top_k == -1 else top_k
    print(img.split('/')[-1] + ':')
    for j in range(top_k):
        jth_index = sorted_indices[i][j]
        print(f'{labels[jth_index]}: {logits[i][jth_index]}')
    print('\n')
