import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import json

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch import autograd
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets

from utils import *
import clip

from mydataset.id_dataset import IdDataset
from mydataset.custom_dataset import CustomDataset
from mydataset.ood_dataset import OodDataset

from mydataset import build_dataset
from mydataset.utils import build_data_loader

from ood_utils.ood_tool import fpr_recall
from ood_utils.ood_tool import auc
from ood_utils.ood_tool import get_measures


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        # print("filename:", filename)
        self.filename = filename

    def write(self, message):
        with open(self.filename, 'a+') as log:
            self.terminal.write(message)
            log.write(message)

    def flush(self):
        pass




class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, tokenized_prompts):
        x = self.token_embedding(tokenized_prompts).type(self.dtype)  # [batch_size, n_ctx, d_model]
        # print('=============')
        # print(x.shape) #torch.Size([B, 77, 512])

        x = x + self.positional_embedding.type(self.dtype)

        # print(x.shape) #torch.Size([B, 77, 512])


        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _, _, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # print(x.shape)#torch.Size([B, 77, 512])
        # print('------')
        # print(torch.arange(x.shape[0])) #取第一维度所有样本，如tensor([0, 1, 2, 3] )
        # print(tokenized_prompts.argmax(dim=-1)) # 如tensor([12], device='cuda:0')

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]@ self.text_projection 

        # print(x.shape) #torch.Size([B, 512])

        return x

def get_base_text_features(cfg, classnames, clip_model, text_encoder):
    device = next(text_encoder.parameters()).device
    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()
    
    TEMPLATES = cfg['template']

    with torch.no_grad():
        text_embeddings = [] # [num_classes, template_num, transformer.width]
        for text in classnames:
            # print(TEMPLATES[0].format(text))
            tokens = clip.tokenize([template.format(text) for template in TEMPLATES])  # tokenized prompts are indices
            # print(tokens.shape,'tokens.shape') # torch.Size([7, 77]) tokens.shape, 7个template
            if clip_model.dtype == torch.float16:
                text_embeddings.append(text_encoder(tokens.cuda()))  # not support float16 on cpu
            else:
                text_embeddings.append(text_encoder(tokens.cuda()))
    
    text_embeddings = torch.stack(text_embeddings).mean(1) 
    text_encoder = text_encoder.to(device)
    return text_embeddings.to(device) 

def get_origin_neutral_text_features(neutral_prompt, classnames, text_encoder, clip_model):
    device = next(text_encoder.parameters()).device

    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()

    with torch.no_grad():
        prompts = [neutral_prompt for _ in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(clip_model.positional_embedding.device)

        origin_text_features = text_encoder( tokenized_prompts.cuda())

    return origin_text_features.to(device)






class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, cache_keys_forced, cache_values_forced, cache_keys_original, cache_values_original):
        super().__init__()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg
        self.classnum = len(classnames)
        self.type_num = 2

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.text_features_origin = get_base_text_features(cfg, classnames, clip_model, self.text_encoder) # [ num_classes, cfg['embed_dim'] ]
        
        self.residual_learner = nn.Parameter(torch.zeros( self.classnum , cfg['embed_dim']).half().cuda() )
        
        # self.cache_keys = cache_keys
        # self.cache_values = cache_values
        self.cache_keys_forced = cache_keys_forced
        self.cache_values_forced = cache_values_forced
        self.cache_keys_original = cache_keys_original
        self.cache_values_original = cache_values_original


        text_features_origin_neutral = get_origin_neutral_text_features(cfg['template_origin_neutral'], classnames, TextEncoder(clip_model), clip_model)  # neutral
        # text_features_origin_neutral = get_base_text_features(cfg, classnames, clip_model, TextEncoder(clip_model))  # likeforced


        origin_neutral_repeat_num = int(cfg['origin_neutral_multiply_factor'])

        if origin_neutral_repeat_num == 0:
            text_features_origin_neutral = text_features_origin_neutral.repeat(0, 1)
            self.type_num -= 1

        self.text_features_origin_neutral = text_features_origin_neutral
        self.origin_neutral_repeat_num = origin_neutral_repeat_num
        print(f"self.text_features_origin_neutral.shape: {self.text_features_origin_neutral.shape}")


    def forward(self, image):
        image_features, local_image_features = self.image_encoder(image.type(self.dtype))

        text_features_origin = self.text_features_origin.type(self.dtype).to(self.cfg['device'])
        text_features = text_features_origin + self.cfg['text_feature_alpha'] * self.residual_learner #   t + a * x


        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        
        # 计算text原型端logits
        text_features_concat = torch.cat((text_features, self.text_features_origin_neutral,), dim=0) # concat
        text_features_concat = text_features_concat / text_features_concat.norm(dim=-1, keepdim=True) # norm要在split之前进行
        
        logits = logit_scale * image_features @ text_features_concat.transpose(-1, -2)
        logits_local = logit_scale * local_image_features @ text_features_concat.T


        batch_size, token_len, sum_template_len = logits_local.shape

        logits = logits.view(batch_size, self.type_num, self.classnum)
        logits_local = logits_local.view(batch_size, token_len, self.type_num, self.classnum)

        logits_text = logits[:,0,:] # logits_forced
        logits_local_text = logits_local[:,:,0,:] # logits_local_forced


        if self.origin_neutral_repeat_num > 0:
            logits_origin_neutral = logits[:,1,:]
            logits_local_origin_neutral = logits_local[:,:,1,:]

            logits_origin_neutral = logits_origin_neutral.repeat(1, self.origin_neutral_repeat_num)
            logits_local_origin_neutral = logits_local_origin_neutral.repeat(1, 1, self.origin_neutral_repeat_num)

            logits_text = torch.cat((logits_text, logits_origin_neutral), dim=1)
            logits_local_text = torch.cat((logits_local_text, logits_local_origin_neutral), dim=2)


        # 计算cache原型端logits
        cache_beta = self.cfg['cache_beta']
        if self.cfg['shots'] != 1:
            keys_res = self.residual_learner.unsqueeze(1).repeat(1, self.cfg['shots']//2, 1).reshape(-1, self.cfg['embed_dim'])
        else:
            keys_res = self.residual_learner

        # cache_keys_forced = self.cache_keys + keys_res
        cache_keys_forced = self.cache_keys_forced + keys_res # 
        affinity = image_features @ cache_keys_forced.T
        # logits_cache_forced = ((-1) * (cache_beta - cache_beta * affinity)).exp() @ self.cache_values
        logits_cache_forced = ((-1) * (cache_beta - cache_beta * affinity)).exp() @ self.cache_values_forced

        affinity_local = local_image_features @ cache_keys_forced.T
        # logits_cache_local_forced = ((-1) * (cache_beta - cache_beta * affinity_local)).exp() @ self.cache_values
        logits_cache_local_forced = ((-1) * (cache_beta - cache_beta * affinity_local)).exp() @ self.cache_values_forced

        if self.origin_neutral_repeat_num > 0:
            # cache_keys_origin = self.cache_keys
            cache_keys_origin = self.cache_keys_original

            affinity_origin = image_features @ cache_keys_origin.T
            # logits_cache_origin = ((-1) * (cache_beta - cache_beta * affinity_origin)).exp() @ self.cache_values
            logits_cache_origin = ((-1) * (cache_beta - cache_beta * affinity_origin)).exp() @ self.cache_values_original
            logits_cache_origin = logits_cache_origin.repeat(1, self.origin_neutral_repeat_num)
            logits_cache_forced = torch.cat((logits_cache_forced, logits_cache_origin), dim=1)

            affinity_local_origin = local_image_features @ cache_keys_origin.T
            # logits_cache_local_origin = ((-1) * (cache_beta - cache_beta * affinity_local_origin)).exp() @ self.cache_values
            logits_cache_local_origin = ((-1) * (cache_beta - cache_beta * affinity_local_origin)).exp() @ self.cache_values_original
            logits_cache_local_origin = logits_cache_local_origin.repeat(1, 1, self.origin_neutral_repeat_num)
            logits_cache_local_forced = torch.cat((logits_cache_local_forced, logits_cache_local_origin), dim=2)

        logits_cache = logits_cache_forced
        logits_cache_local = logits_cache_local_forced


        logits = logits_text + self.cfg['cache_alpha'] * logits_cache
        logits_local = logits_local_text + self.cfg['cache_alpha'] * logits_cache_local #

        return logits, logits_local



def extract_cache_image_feature(cfg, clip_model, train_data_loader, text_features):
    cache_keys_forced = []
    cache_values_forced = []

    cache_keys_original = []
    cache_values_original = []

    logit_scale = clip_model.logit_scale.exp()

    with torch.no_grad():

        for i, (images, target) in enumerate(tqdm(train_data_loader)):
            images = images.cuda()
            target = target.cuda()

            image_features, image_features_local = clip_model.encode_image(images)
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * image_features_norm @ text_features.T
            # logits /= 100.0

            # 批量计算 entropy
            entropy = softmax_entropy(logits)
            # print(entropy)

            # 构造 mask：根据 type_str 决定保留哪些样本
            if cfg['shots'] != 1:

                _, indices_sorted_asc = torch.sort(entropy, descending=False)  # 升序排列
                mid_point = len(indices_sorted_asc) // 2  

                low_entropy_indices = indices_sorted_asc[:mid_point]     # 熵较小的样本索引
                high_entropy_indices = indices_sorted_asc[mid_point:]    # 熵较大的样本索引

                low_entropy_features = image_features[low_entropy_indices]
                high_entropy_features = image_features[high_entropy_indices]

                low_entropy_target = target[low_entropy_indices]
                high_entropy_target = target[high_entropy_indices]

                # normal
                # cache_keys_forced.append(low_entropy_features)
                # cache_values_forced.append(low_entropy_target)
                # cache_keys_original.append(high_entropy_features)
                # cache_values_original.append(high_entropy_target)

                # order reverse
                # cache_keys_forced.append(high_entropy_features)
                # cache_values_forced.append(high_entropy_target)
                # cache_keys_original.append(low_entropy_features)
                # cache_values_original.append(low_entropy_target)

                # just high entropy
                cache_keys_forced.append(high_entropy_features)
                cache_values_forced.append(high_entropy_target)
                cache_keys_original.append(high_entropy_features)
                cache_values_original.append(high_entropy_target)

                # just low entropy
                # cache_keys_forced.append(low_entropy_features)
                # cache_values_forced.append(low_entropy_target)
                # cache_keys_original.append(low_entropy_features)
                # cache_values_original.append(low_entropy_target)


            else:
                cache_keys_forced.append(image_features)
                cache_values_forced.append(target)

                cache_keys_original.append(image_features)
                cache_values_original.append(target)


        cache_keys_forced = torch.cat(cache_keys_forced, dim=0)
        cache_keys_forced /= cache_keys_forced.norm(dim=-1, keepdim=True)

        cache_keys_original = torch.cat(cache_keys_original, dim=0)
        cache_keys_original /= cache_keys_original.norm(dim=-1, keepdim=True)       

        cache_values_forced = F.one_hot(torch.cat(cache_values_forced, dim=0)).half()
        cache_values_original = F.one_hot(torch.cat(cache_values_original, dim=0)).half()


        # 保存结果
        torch.save(cache_keys_forced, f"{cfg['cache_dir']}/keys_{str(cfg['shots'])}shots_forced.pt")
        torch.save(cache_values_forced, f"{cfg['cache_dir']}/values_{str(cfg['shots'])}shots_forced.pt")

        torch.save(cache_keys_original, f"{cfg['cache_dir']}/keys_{str(cfg['shots'])}shots_original.pt")
        torch.save(cache_values_original, f"{cfg['cache_dir']}/values_{str(cfg['shots'])}shots_original.pt")

        print(f'forced cache_keys shape:', cache_keys_forced.shape)
        print(f'forced cache_values shape:', cache_values_forced.shape)
        print(f'original cache_keys shape:', cache_keys_original.shape)
        print(f'original cache_values shape:', cache_values_original.shape)

    return


def softmax_entropy(x): # 概率分布越均匀，熵越大
    log_prob = F.log_softmax(x, dim=-1)
    prob = log_prob.exp()
    return -(prob * log_prob).sum(dim=-1)

# def softmax_entropy(x):
#     return -(x.softmax(1) * x.log_softmax(1)).sum(1)


# def get_mid_entropy(cfg, clip_model, train_data_loader, text_features):
#     # 遍历数据集，计算每个样本的熵值，然后存入数组中
#     all_entropy_list = []

#     with torch.no_grad():
#         for i, (images, target) in enumerate(tqdm(train_data_loader)): 
#             # 假设为16shots，则batch size为16，分别计算每个类16张图的熵的中位数
#             images = images.cuda()
#             image_features,image_features_local = clip_model.encode_image(images)
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             logit_scale = clip_model.logit_scale.exp()

#             logits = logit_scale * image_features @ text_features.T
#             # logits /= 100.0 # 除100不明显

#             # 计算每个样本的熵值
#             entropy = softmax_entropy(logits)
#             # print(entropy)
#             all_entropy_list.append(entropy.cpu().numpy())
#             # print('entropy:', entropy.shape) # torch.Size([200])
#     print('all_entropy_list len:', len(all_entropy_list))
#     # 将熵值排序，然后得到中位数

#     # all_entropy_list的shape为 1000,shots，分别在每个类中排序熵值
#     all_entropy_list = np.sort(all_entropy_list, axis=1)

#     # 分别得到每个类的中位数熵值
#     mid_entropy = all_entropy_list[:, all_entropy_list.shape[1] // 2]

#     print('mid_entropy:', mid_entropy)
#     print(mid_entropy.shape)
    
#     return mid_entropy

    




def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/id_dataset_taskres_cache.yaml", help='settings in yaml format')
    args = parser.parse_args()

    return args

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备是：{device}")
    # 获取可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"当前系统上可用的GPU数量为：{num_gpus}")

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg['device'] = device

    # CLIP 
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model = torch.nn.DataParallel(clip_model).to(device)
    clip_model = clip_model.module

    model_file_name_dict = {
        "RN50": "RN50",
        "RN101": "RN101",
        "RN50x4": "RN50x4",
        "RN50x16": "RN50x16",
        "ViT-B/32": "ViT-B-32",
        "ViT-B/16": "ViT-B-16",
    }

    model_path = os.path.join("/home/enroll2024/xinhua/.cache/clip/", model_file_name_dict[cfg['backbone']]+".pt" )
    model_temp = torch.jit.load(model_path, map_location=device ).eval()
    state_dict = model_temp.state_dict()


    cfg['embed_dim'] = state_dict["text_projection"].shape[1]
    print(cfg['embed_dim'])

    origin_neutral_multiply_factor = 3
    cfg['origin_neutral_multiply_factor'] = origin_neutral_multiply_factor

    # cfg['template'] = ['a photo of a {}.',]
    cfg['template_origin_neutral'] = 'a photo of a thing that we can see in nature.'  # 


    is_train = 0
    is_extract_feature = 1

    model_name = 'fa_taskres_splitcache_entropy_justhighentropy' #
    lrname = str(cfg['lr']).replace('.','')
    cache_dir = os.path.join('./mycaches_new', cfg['id_dataset'], model_file_name_dict[cfg['backbone']], str(cfg['shots'])+'shots',model_name+'_bs'+str(cfg['fine_tune_batch_size'])+'_ep'+str(cfg['fine_tune_train_epoch']),'fcachenotrm_ocachenotrm','neutral_len'+str(origin_neutral_multiply_factor),'lr'+lrname,'seed'+str(cfg['seed'])  )  #   

    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir


    # seed
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    transform_aug_H = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random', inplace=False),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),

    ])

    transform_aug_M = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    # train_transform_aug = preprocess
    train_transform_no_aug = preprocess

    train_transform = train_transform_no_aug
    cache_transform = train_transform_no_aug



    if is_train == 1:
        sys.stdout = Logger( os.path.join(cache_dir,'log_train.txt') ,  stream=sys.stdout)
        print("\nRunning configs.")
        print(cfg, "\n")

        print("train transform:")
        for transform in train_transform.transforms:
            print(transform)


        # stage2 dataset
        few_shot_dataset_stage2 = build_dataset(cfg['id_dataset'], cfg['root_path'], cfg['shots'])
        train_batch_size = cfg['fine_tune_batch_size']
        test_batch_size = cfg['test_batch_size']

        val_loader = build_data_loader(data_source=few_shot_dataset_stage2.val, batch_size=test_batch_size, is_train=False, tfm=train_transform_no_aug, shuffle=False)
        test_loader = build_data_loader(data_source=few_shot_dataset_stage2.test, batch_size=test_batch_size, is_train=False, tfm=train_transform_no_aug, shuffle=False)

        train_loader = build_data_loader(data_source=few_shot_dataset_stage2.train_x, batch_size=train_batch_size, tfm=train_transform, is_train=True, shuffle=True)
        
        cfg['template'] = few_shot_dataset_stage2.template
        cfg['classnames'] = few_shot_dataset_stage2.classnames  #



        print(f"template: {cfg['template'] }")
        print(f"classnames: {cfg['classnames'] }")
        print(f"len of origin classnames: {len(cfg['classnames'])}")
        print(f"origin_neutral_multiply_factor: {cfg['origin_neutral_multiply_factor']}")



        if is_extract_feature == 1:
            print("cache transform:")
            for transform in cache_transform.transforms:
                print(transform)

            train_loader_extract= build_data_loader(data_source=few_shot_dataset_stage2.train_x, batch_size=cfg['split_cache_batch_size'], tfm=cache_transform, is_train=True, shuffle=False)


            text_features = get_base_text_features(cfg, cfg['classnames'], clip_model, TextEncoder(clip_model)) 
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # mid_entropy = get_mid_entropy(cfg, clip_model, train_loader_extract, text_features) # 计算中位数熵值


            extract_cache_image_feature(cfg, clip_model, train_loader_extract, text_features) #  
            print('extract cache image feature done')
            


        # load cache keys and values
        # cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_keys_forced = torch.load(f"{cfg['cache_dir']}/keys_{str(cfg['shots'])}shots_forced.pt")
        cache_keys_original = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots_original.pt")
        # cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
        cache_values_forced = torch.load(f"{cfg['cache_dir']}/values_{str(cfg['shots'])}shots_forced.pt")
        cache_values_original = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots_original.pt")

        # cache_keys = cache_keys.to(device)
        cache_keys_forced = cache_keys_forced.to(device)
        cache_keys_original = cache_keys_original.to(device)

        # cache_values = cache_values.to(device)
        cache_values_forced = cache_values_forced.to(device)
        cache_values_original = cache_values_original.to(device)

        # print('cache_keys shape:', cache_keys.shape)
        print('cache_keys_forced shape:', cache_keys_forced.shape)
        print('cache_keys_original shape:', cache_keys_original.shape)
        
        # print('cache_values shape:', cache_values.shape)
        print('cache_values_forced shape:', cache_values_forced.shape)
        print('cache_values_original shape:', cache_values_original.shape)

        # Model_stage2
        # model_stage2 = CustomCLIP(cfg,cfg['classnames'],clip_model,cache_keys, cache_values)
        model_stage2 = CustomCLIP(cfg,cfg['classnames'],clip_model,cache_keys_forced, cache_values_forced, cache_keys_original, cache_values_original)
        model_stage2 = torch.nn.DataParallel(model_stage2).to(device)
        
        

        for name, param in model_stage2.named_parameters():
            if  "residual_learner" in name : # 
                param.requires_grad_(True)
                print(name)
            else:
                param.requires_grad_(False)
                # print(name,'False')



        criterion = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model_stage2.parameters(), lr=cfg['lr'], weight_decay=5e-4,momentum=0.9,dampening=0,nesterov=False)
        # optimizer = torch.optim.Adam(model_stage2.parameters(), lr=cfg['lr'], weight_decay=5e-4)
        optimizer = torch.optim.AdamW(model_stage2.parameters(), lr=cfg['lr'], eps=0.001, weight_decay=1e-1) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,  cfg['fine_tune_train_epoch'] * len(train_loader))

        print(optimizer)
        print(scheduler)

        train_epoch = cfg['fine_tune_train_epoch']

        for train_idx in range(train_epoch):

            print('Train Epoch: {:} / {:}'.format(train_idx+1, train_epoch))
            correct_samples, all_samples = 0, 0
            loss_list = []
            model_stage2.train()
            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()

                logits,_ = model_stage2(images)
                loss = criterion(logits, target)
                acc = cls_acc(output=logits , target=target, topk=1)

                correct_samples += acc / 100 * len(logits)
                all_samples += len(logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, train_acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))



            # test
            if  train_idx == train_epoch - 1 : # True 
                model_stage2.eval()
                with torch.no_grad():
                    correct_samples, all_samples = 0, 0
                    topk = 1


                    for images, labels in tqdm(test_loader):
                        images, labels = images.to(device), labels.to(device)
                        outputs,_ = model_stage2(images)

                        outputs = outputs / 100.0
                        batch_size, repeat_len = outputs.shape
                        outputs_temp = outputs.view(batch_size, cfg['origin_neutral_multiply_factor']+1 , len(cfg['classnames']))
                        just_origin_outputs = outputs_temp[:,0,:]


                        pred = just_origin_outputs.topk(topk, 1, True, True)[1].t()
                        correct = pred.eq(labels.view(1, -1).expand_as(pred))
                        acc_num = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                        correct_samples += acc_num
                        all_samples += labels.shape[0]





                    test_acc = correct_samples / all_samples
                    print(f'test Epoch [{train_idx+1}/{train_epoch}], Test accuracy of the model on the test images: { 100 * test_acc :.2f}%')

                    # save model
                    torch.save(model_stage2.state_dict(), os.path.join(cfg['cache_dir'], 'model.pth'))




            
    else:
        # test ood

        if cfg['ood_dataset'] == 'challenging':
            test_file_name = 'log_test_ood_challenging.txt'
        elif cfg['ood_dataset'] == 'all':
            test_file_name = 'log_test_ood_all.txt'
        elif cfg['ood_dataset'] == 'nearood':
            test_file_name = 'log_test_ood_nearood.txt'
        elif cfg['ood_dataset'] == 'common':
            test_file_name = 'log_test_ood_common.txt'
        else:
            test_file_name = f'log_test_ood_{cfg["ood_dataset"]}.txt'
        sys.stdout = Logger( os.path.join(cache_dir, test_file_name) ,  stream=sys.stdout)
        print("\nRunning configs.")
        print(cfg, "\n")

        print("test transform:")
        for transform in train_transform_no_aug.transforms:
            print(transform)



        # stage2 dataset
        few_shot_dataset_stage2 = build_dataset(cfg['id_dataset'], cfg['root_path'], cfg['shots'])
        batch_size_stage2 = cfg['fine_tune_batch_size']

        test_loader_id = build_data_loader(data_source=few_shot_dataset_stage2.test, batch_size=batch_size_stage2, is_train=False, tfm=train_transform_no_aug, shuffle=False)

        cfg['template'] = few_shot_dataset_stage2.template
        cfg['classnames'] = few_shot_dataset_stage2.classnames



        print(f"template: {cfg['template'] }")
        print(f"classnames: {cfg['classnames'] }")
        print(f"len of origin classnames: {len(cfg['classnames'])}")
        print(f"origin_neutral_multiply_factor: {cfg['origin_neutral_multiply_factor']}")


        # ood dataset
        ood_dataset_name = cfg['ood_dataset']

        if ood_dataset_name == 'challenging':
            ood_dataset_name = ['OpenImage_O','NINCO','Imagenet-o' ]
        elif ood_dataset_name == 'all':
            ood_dataset_name = ['iNaturalist','SUN','Places','dtd','OpenImage_O','NINCO','Imagenet-o']
        elif ood_dataset_name == 'nearood':
            ood_dataset_name = ['ssb_hard','NINCO']
        elif ood_dataset_name == 'common':
            ood_dataset_name = ['iNaturalist','SUN','Places','dtd']
        else:
            ood_dataset_name = [ood_dataset_name]
        # else:
        #     if cfg['id_dataset'] in ['imagenet', 'fgvc', 'caltech101', 'stanford_cars', 'food101','imagenet100','ucf101','oxford_flowers','oxford_pets','eurosat']:
        #         ood_dataset_name = ['iNaturalist','SUN','Places','dtd']
        #     else:
        #         ood_dataset_name = [ood_dataset_name]

        print(f"ood_dataset_name: {ood_dataset_name}")



        # load cache keys and values
        # cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_keys_forced = torch.load(f"{cfg['cache_dir']}/keys_{str(cfg['shots'])}shots_forced.pt")
        cache_keys_original = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots_original.pt")
        # cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
        cache_values_forced = torch.load(f"{cfg['cache_dir']}/values_{str(cfg['shots'])}shots_forced.pt")
        cache_values_original = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots_original.pt")

        # cache_keys = cache_keys.to(device)
        cache_keys_forced = cache_keys_forced.to(device)
        cache_keys_original = cache_keys_original.to(device)

        # cache_values = cache_values.to(device)
        cache_values_forced = cache_values_forced.to(device)
        cache_values_original = cache_values_original.to(device)

        # print('cache_keys shape:', cache_keys.shape)
        print('cache_keys_forced shape:', cache_keys_forced.shape)
        print('cache_keys_original shape:', cache_keys_original.shape)
        
        # print('cache_values shape:', cache_values.shape)
        print('cache_values_forced shape:', cache_values_forced.shape)
        print('cache_values_original shape:', cache_values_original.shape)

        # Model_stage2
        # model_stage2 = CustomCLIP(cfg,cfg['classnames'],clip_model,cache_keys, cache_values)
        model_stage2 = CustomCLIP(cfg,cfg['classnames'],clip_model,cache_keys_forced, cache_values_forced, cache_keys_original, cache_values_original)
        model_stage2 = torch.nn.DataParallel(model_stage2).to(device)


        # load model
        model_stage2.load_state_dict(torch.load(os.path.join(cfg['cache_dir'], 'model.pth')))




        model_stage2.eval()
        with torch.no_grad():
            correct_samples, all_samples = 0, 0
            correct_samples_gl, all_samples_gl = 0, 0
            topk = 1
            T = 1.0

            # logit_result
            id_conf = np.array([])
            id_conf_gl = np.array([])

            # run id dataset
            for images, labels in tqdm(test_loader_id):
                images, labels = images.to(device), labels.to(device)

                logits_id, logits_id_local = model_stage2(images)
                logits_id /= 100.0
                logits_id_local /= 100.0
                # print(logits_id.shape) # torch.Size([200, 3000])
                # print(logits_id_local.shape) # torch.Size([200, 196, 3000])


                # ----
                batch_size, token_len, repeat_len = logits_id_local.shape
                logits_id_temp = logits_id.view(batch_size, cfg['origin_neutral_multiply_factor']+1 , len(cfg['classnames']))
                just_origin_logits_id = logits_id_temp[:,0,:]
                # print(just_origin_logits_id.shape) # torch.Size([200, 1000])
                # print(just_origin_logits_id[0,:10])

                # print(logits_id_temp[0,1,:10]) #
                # print(logits_id_temp[0,2,:10]) # the same

                logits_id_local_temp = logits_id_local.view(batch_size, token_len, cfg['origin_neutral_multiply_factor']+1 , len(cfg['classnames']))
                just_origin_logits_id_local = logits_id_local_temp[:,:,0,:]
                # ----



                smax_global = F.softmax(logits_id/T, dim=-1)
                smax_global = smax_global.cpu().numpy()
                smax_global = smax_global[:, :len(cfg['classnames'])]  # 只取forced
                mcm_global_score = np.max(smax_global, axis=1)

                smax_local = F.softmax(logits_id_local/T, dim=-1)
                smax_local = smax_local.cpu().numpy()
                smax_local = smax_local[:, :, :len(cfg['classnames'])]  # 只取forced
                mcm_local_score = np.max(smax_local, axis=(1, 2))
                # print(logits_id.size)

                id_conf = np.concatenate((id_conf, mcm_global_score))
                id_conf_gl = np.concatenate((id_conf_gl, mcm_global_score + mcm_local_score))


                # caculate accuracy
                pred = just_origin_logits_id.topk(topk, 1, True, True)[1].t()
                correct = pred.eq(labels.view(1, -1).expand_as(pred))
                acc_num = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                correct_samples += acc_num
                all_samples += labels.shape[0]

                logits_id_local_gl = just_origin_logits_id_local.max(dim=1)[0] + just_origin_logits_id
                pred_gl = logits_id_local_gl.topk(topk, 1, True, True)[1].t()
                correct_gl = pred_gl.eq(labels.view(1, -1).expand_as(pred_gl))
                acc_num_gl = float(correct_gl[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                correct_samples_gl += acc_num_gl
                all_samples_gl += labels.shape[0]

            
            test_acc = correct_samples / all_samples
            test_acc_gl = correct_samples_gl / all_samples_gl
            print(f'Test accuracy of the model on the test images: { 100 * test_acc :.2f}%')
            print(f'Test gl accuracy of the model on the test images: { 100 * test_acc_gl :.2f}%')


            # run ood dataset

            # avarage ood 
            avg_auroc, avg_aupr, avg_fpr = 0.0, 0.0, 0.0
            avg_auroc_gl, avg_aupr_gl, avg_fpr_gl = 0.0, 0.0, 0.0

            for cur_ood_dataset in ood_dataset_name:
                print(f'cur ood dataset: {cur_ood_dataset}')
                ood_conf = np.array([])
                ood_conf_gl = np.array([])

                if  cur_ood_dataset == 'ood':
                    ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset), transform=train_transform_no_aug)
                elif cur_ood_dataset in ['iNaturalist','SUN','Places','OpenImage_O','Imagenet-o','ssb_hard']:
                    ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset), transform=train_transform_no_aug)
                elif cur_ood_dataset == 'dtd':
                    ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset, 'images'), transform=train_transform_no_aug)
                # elif cur_ood_dataset == 'OpenImage_O':
                #     ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset,), transform=train_transform_no_aug)
                elif cur_ood_dataset == 'NINCO':
                    ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset, 'NINCO_OOD_classes'), transform=train_transform_no_aug)
                # elif cur_ood_dataset == 'Imagenet-o':
                #     ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset, ), transform=train_transform_no_aug)
                else:
                    ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset, 'test'), transform=train_transform_no_aug)
                
                ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=cfg['test_batch_size'],
                                                            shuffle=False, num_workers=8)


                for images, _ in tqdm(ood_loader):
                    images = images.to(device)

                    logits_ood, logits_ood_local = model_stage2(images)
                    logits_ood /= 100.0
                    logits_ood_local /= 100.0
                    # print(logits_ood.shape)
                    # print(logits_ood_local.shape)

                    

                    smax_global = F.softmax(logits_ood/T, dim=-1)
                    smax_global = smax_global.cpu().numpy()
                    smax_global = smax_global[:, :len(cfg['classnames'])]  # 只取forced
                    mcm_global_score = np.max(smax_global, axis=1)

                    smax_local = F.softmax(logits_ood_local/T, dim=-1)
                    smax_local = smax_local.cpu().numpy()
                    smax_local = smax_local[:, :, :len(cfg['classnames'])]  # 只取forced
                    mcm_local_score = np.max(smax_local, axis=(1, 2))
                    # print(logits_ood.size)

                    ood_conf = np.concatenate((ood_conf, mcm_global_score))
                    ood_conf_gl = np.concatenate((ood_conf_gl, mcm_global_score + mcm_local_score))


                print('id_conf size',id_conf.size)
                print('ood_conf size',ood_conf.size)


                # ood detection
                tpr = 0.95



                auroc, aupr, fpr = get_measures(id_conf, ood_conf, tpr)
                avg_auroc += auroc
                avg_aupr += aupr
                avg_fpr += fpr
                print(f'AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, FPR(0.95): {fpr:.4f}')


                auroc_gl, aupr_gl, fpr_gl = get_measures(id_conf_gl, ood_conf_gl, tpr)
                avg_auroc_gl += auroc_gl
                avg_aupr_gl += aupr_gl
                avg_fpr_gl += fpr_gl
                print(f'AUROC_glmcm: {auroc_gl:.4f}, AUPR_glmcm: {aupr_gl:.4f}, FPR(0.95)_glmcm: {fpr_gl:.4f}')
            
            print('--------------------------------')
            avg_auroc /= len(ood_dataset_name)
            avg_aupr /= len(ood_dataset_name)
            avg_fpr /= len(ood_dataset_name)
            print(f'Average AUROC: {avg_auroc:.4f}, Average AUPR: {avg_aupr:.4f}, Average FPR(0.95): {avg_fpr:.4f}')

            avg_auroc_gl /= len(ood_dataset_name)
            avg_aupr_gl /= len(ood_dataset_name)
            avg_fpr_gl /= len(ood_dataset_name)
            print(f'Average AUROC_glmcm: {avg_auroc_gl:.4f}, Average AUPR_glmcm: {avg_aupr_gl:.4f}, Average FPR(0.95)_glmcm: {avg_fpr_gl:.4f}')
if __name__ == '__main__':
    # python mymain.py --config configs/myconfig.yaml
    main()