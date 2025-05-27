
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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
from torchvision.transforms import Compose, Normalize
from torch import autograd
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets

from utils import *
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


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



def gen_one_hot(labels, num_class):
    labels = labels.to(torch.int64)
    n = labels.shape[0]
    one_hot_label = torch.zeros([n, num_class])
    for i in range(n):
        one_hot_label[i, labels[i]] = 1
    
    return one_hot_label
    

class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, N_CTX = 16, CTX_INIT = "a photo of a", CSC = False, CLASS_TOKEN_POSITION = "end",cfg = None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = N_CTX
        ctx_init = CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        device = cfg['device']
        

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]

            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = ctx_vectors.repeat(n_cls, 1, 1)
            prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype).to(device)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).to(device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # ----- normal optimized
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # -----

        # # ----- res optimized
        # self.ctx = ctx_vectors  # 
        # if CSC:
        #     ctx_res = torch.zeros(ctx_vectors.shape[0], 1, ctx_vectors.shape[2] ).to(device) # [1000,1,512]
        #     # ctx_res = torch.zeros_like(ctx_vectors).to(device) # likeorigin: [1000,4,512]

        #     self.ctx_res = nn.Parameter(ctx_res)
        # else:
        #     self.ctx_res = nn.Parameter(torch.zeros_like(ctx_vectors[0]).unsqueeze(0)) # [1,512]
        #     # self.ctx_res = nn.Parameter(torch.zeros_like(ctx_vectors)) # likeorigin: [4,512]
        # # ----- res


        print("------------------")
        print(self.ctx.shape)  # 当为"a photo of a"时为[4, 512]，csc为True时为[1000, 4, 512]
        # print(self.ctx_res.shape)  # 当为"a photo of a"时为[1, 512]，csc为True时为[1000, 1, 512]
        print("------------------")        

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(clip_model.positional_embedding.device)
        # print('tokenized:',tokenized_prompts.shape) #[len(classnames), 77]

        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        # ctx = self.ctx + self.ctx_res

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        # print('prompts.shape',prompts.shape)  # [len(classnames), ctx_sum, transformer.width] 如[1000, 77, 512]
        return prompts

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts.type(self.dtype) + self.positional_embedding.to(dtype = self.dtype, device = prompts.device)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _, _, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

def get_origin_text_features(prompt_learner, text_encoder, clip_model):
    device = next(text_encoder.parameters()).device

    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()

    with torch.no_grad():
        prompts = prompt_learner()
        tokenized_prompts = prompt_learner.tokenized_prompts

        origin_text_features = text_encoder(prompts.cuda(), tokenized_prompts.cuda())

    text_encoder = text_encoder.to(device)
    return origin_text_features.to(device)

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, 
                 clip_model,
                 CTX_INIT = "This is a medical pathology image of ", 
                 single = False, 
                 n_ctx = 16,
                 csc = False):
        super().__init__()
        
        self.prompt_learner_noinit = PromptLearner(classnames, clip_model, CTX_INIT='', N_CTX=n_ctx, CSC = csc, cfg = cfg)
        self.prompt_learner_init = PromptLearner(classnames, clip_model, CTX_INIT=CTX_INIT, N_CTX=n_ctx, CSC = csc, cfg = cfg)

        
        self.tokenized_prompts = self.prompt_learner_init.tokenized_prompts 
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.frozen_weights()
        self.CLIP_Transforms = Compose([
                            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                        ])

        self.single = single
        self.cfg = cfg
        self.classnum = len(classnames)

        self.text_features_classify_origin = get_origin_text_features(self.prompt_learner_noinit, self.text_encoder, clip_model)

        integer_part = int(cfg['multiply_factor'])
        repeat_sum_num = int(cfg['multiply_factor']) - 1

        if repeat_sum_num > 0:
            integer_repeated_tensor = self.text_features_classify_origin.repeat(1, 1)
        else:
            integer_repeated_tensor = self.text_features_classify_origin.repeat(0, 1)
        
        self.text_features_classify_repeat = integer_repeated_tensor
        self.repeat_sum_num = repeat_sum_num
        print(f"self.text_features_classify_repeat.shape: {self.text_features_classify_repeat.shape}")

        
    def frozen_weights(self):
        print("trainable parameters:")
        for name, param in self.named_parameters():
            if "prompt_learner_init" not in name:
                param.requires_grad_(False)
            else:
                print(name)
                param.requires_grad_(True)
        print("---------------------------")
        
    def forward(self, image):
        if self.single : image = self.CLIP_Transforms(image)
        
        # image_features = self.image_encoder(image.type(self.dtype))
        image_features, local_image_features = self.image_encoder(image.type(self.dtype))


        prompts = self.prompt_learner_init().to(image.device)   #[len(classnames), ctx_sum, transformer.width] 如[1000, 77, 512]
        tokenized_prompts = self.tokenized_prompts.to(image.device)  #[len(classnames), 77]
        tokenized_prompts = tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts).type(self.dtype)

        # confuses
        text_features_classify_final = torch.cat((text_features, self.text_features_classify_repeat), dim=0)
        # print(f"text_features_classify_final.shape: {text_features_classify_final.shape}")


        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
        text_features_classify_final = text_features_classify_final / text_features_classify_final.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features_classify_final.transpose(-1, -2)
        logits_local = logit_scale * local_image_features @ text_features_classify_final.T

        # print(f"logits.shape: {logits.shape}")
        # print(f"logits_local.shape: {logits_local.shape}")
        # print('-------')

        if self.repeat_sum_num - 1 > 0:
            
            batch_size, token_len, sum_template_len = logits_local.shape
            logits_repeat_template = logits.view(batch_size, 2, self.classnum)[:,1,:]
            # print(f"logits_repeat_template.shape: {logits_repeat_template.shape}")
            logits_repeat_context = logits_repeat_template.repeat(1, self.repeat_sum_num - 1)
            # print(f"logits_repeat_context.shape: {logits_repeat_context.shape}")

            logits = torch.cat((logits, logits_repeat_context), dim=1)
            # print(f"logits.shape: {logits.shape}")



            logits_local_repeat_template = logits_local.view(batch_size, token_len, 2, self.classnum)[:,:,1,:]
            # print(f"logits_local_repeat_template.shape: {logits_local_repeat_template.shape}")
            logits_local_repeat_context = logits_local_repeat_template.repeat(1, 1, self.repeat_sum_num - 1)
            # print(f"logits_local_repeat_context.shape: {logits_local_repeat_context.shape}")

            logits_local = torch.cat((logits_local, logits_local_repeat_context), dim=2)
            # print(f"logits_local.shape: {logits_local.shape}")



        
        return logits, logits_local


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/id_dataset.yaml", help='settings in yaml format')
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

    is_train = 0

    multiply_factor = 3 #
    cfg['multiply_factor'] = multiply_factor
    csc = False
    cfg['csc'] = csc
    cfg['template'] = 'a photo of a'  # 
    # a photo of a
    # a photo of the small  # t1
    # a bad photo of the # t2
    # a origami  # t3
    # itap of a # t4
    # art of the # t5
    # a photo of the large # t6
    

    multiply_factor_name = str(multiply_factor).replace('.','-')
    lrname = str(cfg['lr']).replace('.','')

    model_name = 'coop'
    cache_dir = os.path.join('./mycaches_id', cfg['id_dataset'], model_file_name_dict[cfg['backbone']], str(cfg['shots'])+'shots',model_name+'_efficient_batchs160_ep50_prompt_ablation','forced_init_origin_random','lenx'+str(multiply_factor_name),'lr'+lrname,'seed'+str(cfg['seed'])  )  #   
    # cache_dir = os.path.join('./mycaches_id', cfg['id_dataset'], model_file_name_dict[cfg['backbone']], str(cfg['shots'])+'shots',model_name+'_efficient_batchs160_ep50_learnnullrepeatinit','lenx'+str(multiply_factor_name),'lr0002'  )  #  new_test   seed1   

    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    
    # seed
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])



    train_transform_aug = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),

    ])
    # train_transform_aug = preprocess
    train_transform_no_aug = preprocess





    if is_train == 1:
        sys.stdout = Logger( os.path.join(cache_dir,'log_train.txt') ,  stream=sys.stdout)
        print("\nRunning configs.")
        print(cfg, "\n")

        print("augment transform:")
        for transform in train_transform_aug.transforms:
            print(transform)


        # # id dataset
        # id_dataset = IdDataset(cfg['root_path'], cfg['id_dataset'], cfg['shots'])
        # train_path = id_dataset.train_path
        # train_targets = id_dataset.train_targets
        # train_dataset = CustomDataset(train_path, train_targets, train_transform_aug)

        # test_path = id_dataset.test_path
        # test_targets = id_dataset.test_targets
        # test_dataset = CustomDataset(test_path, test_targets, train_transform_no_aug)

        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['fine_tune_batch_size'], shuffle=True, num_workers=8)
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['fine_tune_batch_size'] , shuffle=True, num_workers=8)


        # cfg['template'] = 'This is a medical pathology image of '  # ['This is a medical pathology image of {}']
        # cfg['classnames'] = id_dataset.classnames



        # stage2 dataset
        few_shot_dataset_stage2 = build_dataset(cfg['id_dataset'], cfg['root_path'], cfg['shots'])
        batch_size_stage2 = cfg['fine_tune_batch_size']

        val_loader = build_data_loader(data_source=few_shot_dataset_stage2.val, batch_size=batch_size_stage2, is_train=False, tfm=train_transform_no_aug, shuffle=False)
        test_loader = build_data_loader(data_source=few_shot_dataset_stage2.test, batch_size=batch_size_stage2, is_train=False, tfm=train_transform_no_aug, shuffle=False)

        train_loader = build_data_loader(data_source=few_shot_dataset_stage2.train_x, batch_size=batch_size_stage2, tfm=train_transform_aug, is_train=True, shuffle=True)
        
        
        cfg['classnames'] = few_shot_dataset_stage2.classnames # 



        print(f"template: {cfg['template'] }")
        print(f"classnames: {cfg['classnames'] }")
        print(f"len of origin classnames: {len(cfg['classnames'])}")
        print(f"multiply_factor: {multiply_factor}")


        # Model_stage2
        model_stage2 = CustomCLIP(cfg, cfg['classnames'], clip_model ,cfg['template'], False, 16, cfg['csc'])
        model_stage2 = torch.nn.DataParallel(model_stage2).to(device)
        

        for name, param in model_stage2.named_parameters():
            if  "prompt_learner" in name : #  
                param.requires_grad_(True)
                print(name)
            else:
                param.requires_grad_(False)



        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model_stage2.parameters(), lr=cfg['lr'], weight_decay=5e-4 ,momentum=0.9 ,dampening=0 ,nesterov=False)  # 
        # optimizer = torch.optim.Adam(model_stage2.parameters(), lr=cfg['lr'], weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,  cfg['fine_tune_train_epoch'] * len(train_loader))

        print(optimizer)
        print(scheduler)

        train_epoch = cfg['fine_tune_train_epoch']
        best_test_acc, best_epoch = 0.0, 0

        for train_idx in range(train_epoch):
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, train_epoch))

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
            if (train_idx + 1) % 100 == 0 or  train_idx == train_epoch - 1 :   #   train_idx == 0  or
                model_stage2.eval()
                with torch.no_grad():
                    correct_samples, all_samples = 0, 0
                    topk = 1

                    # correct_items = 0
                    # total_items = 0

                    for images, labels in tqdm(test_loader):
                        images, labels = images.to(device), labels.to(device)
                        outputs,_ = model_stage2(images)

                        outputs = outputs / 100.0
                        batch_size, repeat_len = outputs.shape
                        outputs_temp = outputs.view(batch_size, cfg['multiply_factor'], len(cfg['classnames']))
                        just_origin_outputs = outputs_temp[:,0,:]


                        pred = just_origin_outputs.topk(topk, 1, True, True)[1].t()
                        correct = pred.eq(labels.view(1, -1).expand_as(pred))
                        acc_num = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                        correct_samples += acc_num
                        all_samples += labels.shape[0]





                    test_acc = correct_samples / all_samples
                    print(f'test Epoch [{train_idx+1}/{train_epoch}], Test accuracy of the model on the test images: { 100 * test_acc :.2f}%')


                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        best_epoch = train_idx

                        # save model
                        torch.save(model_stage2.state_dict(), os.path.join(cfg['cache_dir'], 'model.pth'))

            
    else:
        if cfg['ood_dataset'] == 'challenging':
            test_file_name = 'log_test_ood_challenging.txt'
        else:
            test_file_name = 'log_test_ood_common.txt'
        sys.stdout = Logger( os.path.join(cache_dir, test_file_name) ,  stream=sys.stdout)
        print("\nRunning configs.")
        print(cfg, "\n")

        # test ood

        # # id dataset
        # id_dataset = IdDataset(cfg['root_path'], cfg['id_dataset'], cfg['shots'])

        # test_path = id_dataset.test_path
        # test_targets = id_dataset.test_targets
        # test_dataset = CustomDataset(test_path, test_targets, train_transform_no_aug)

        # test_loader_id = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['test_batch_size'] , shuffle=True, num_workers=8)


        # cfg['template'] = 'This is a medical pathology image of '  # ['This is a medical pathology image of {}']
        # cfg['classnames'] = id_dataset.classnames


        # id dataset
        few_shot_dataset_stage2 = build_dataset(cfg['id_dataset'], cfg['root_path'], cfg['shots'])
        batch_size_stage2 = cfg['test_batch_size']

        test_loader_id = build_data_loader(data_source=few_shot_dataset_stage2.test, batch_size=batch_size_stage2, is_train=False, tfm=train_transform_no_aug, shuffle=False)

        # cfg['template'] = 'a photo of a' # few_shot_dataset_stage2.template
        cfg['classnames'] = few_shot_dataset_stage2.classnames  #



        print(f"template: {cfg['template'] }")
        print(f"classnames: {cfg['classnames'] }")
        print(f"len of origin classnames: {len(cfg['classnames'])}")


        # ood dataset
        ood_dataset_name = cfg['ood_dataset']

        if ood_dataset_name == 'challenging':
            ood_dataset_name = ['OpenImage_O','NINCO','Imagenet-o' ]
        
        else:
            if cfg['id_dataset'] in ['imagenet', 'fgvc', 'caltech101', 'stanford_cars', 'food101','imagenet100','ucf101']  :
                ood_dataset_name = ['iNaturalist','SUN','Places','dtd']
            else:
                ood_dataset_name = [ood_dataset_name]

        print(f"ood_dataset_name: {ood_dataset_name}")

        # Model_stage2
        model_stage2 = CustomCLIP(cfg, cfg['classnames'], clip_model ,cfg['template'], False, 16, cfg['csc'])
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
            ood_conf = np.array([])
            ood_conf_gl = np.array([])

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
                logits_id_temp = logits_id.view(batch_size, cfg['multiply_factor'], len(cfg['classnames']))
                just_origin_logits_id = logits_id_temp[:,0,:]
                # print(just_origin_logits_id.shape) # torch.Size([200, 1000])
                # print(just_origin_logits_id[0,:10])

                # print(logits_id_temp[0,1,:10]) #
                # print(logits_id_temp[0,2,:10]) # the same

                logits_id_local_temp = logits_id_local.view(batch_size, token_len, cfg['multiply_factor'], len(cfg['classnames']))
                just_origin_logits_id_local = logits_id_local_temp[:,:,0,:]
                # ----



                smax_global = F.softmax(logits_id/T, dim=-1).cpu().numpy()
                mcm_global_score = np.max(smax_global, axis=1)

                smax_local = F.softmax(logits_id_local/T, dim=-1).cpu().numpy()
                mcm_local_score = np.max(smax_local, axis=(1, 2))
                # print(logits_id.size)

                id_conf = np.concatenate((id_conf, mcm_global_score))
                id_conf_gl = np.concatenate((id_conf_gl, mcm_global_score + mcm_local_score))

                # # caculate accuracy
                # pred = logits_id.topk(topk, 1, True, True)[1].t()
                # correct = pred.eq(labels.view(1, -1).expand_as(pred))
                # acc_num = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                # correct_samples += acc_num
                # all_samples += labels.shape[0]

                # logits_id_local_gl = logits_id_local.max(dim=1)[0] + logits_id
                # pred_gl = logits_id_local_gl.topk(topk, 1, True, True)[1].t()
                # correct_gl = pred_gl.eq(labels.view(1, -1).expand_as(pred_gl))
                # acc_num_gl = float(correct_gl[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                # correct_samples_gl += acc_num_gl
                # all_samples_gl += labels.shape[0]


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
                elif cur_ood_dataset in ['iNaturalist','SUN','Places','OpenImage_O','Imagenet-o']:
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

                    smax_global = F.softmax(logits_ood/T, dim=-1).cpu().numpy()
                    mcm_global_score = np.max(smax_global, axis=1)

                    smax_local = F.softmax(logits_ood_local/T, dim=-1).cpu().numpy()
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