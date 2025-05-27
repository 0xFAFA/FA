import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '5'

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


class TextResLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.alpha = cfg['text_feature_alpha']

        self.origin_classnum = len(classnames)


        # # ------其他不学习
        # self.text_feature_residuals_origin = nn.Parameter(torch.zeros( self.origin_classnum, cfg['embed_dim']))
        # self.text_feature_residuals = self.text_feature_residuals_origin

        # if cfg['multiply_factor'] > 1:
            
        #     self.text_feature_residuals_multiply = torch.zeros( int(self.origin_classnum * (cfg['multiply_factor']-1) ), cfg['embed_dim']).to(self.device)
        #     print(f"self.text_feature_residuals_multiply.shape: {self.text_feature_residuals_multiply.shape}")
        # # ------

        # ------其他学习
        self.text_feature_residuals= nn.Parameter(torch.zeros( self.origin_classnum * cfg['multiply_factor'], cfg['embed_dim']))
        # ------




    def forward(self, text_features):
        # # 其他分类头不学习
        # if self.cfg['multiply_factor'] > 1:
        #     text_feature_residuals = torch.cat([self.text_feature_residuals ,self.text_feature_residuals_multiply ], dim=0)
        # else:
        #     text_feature_residuals = self.text_feature_residuals
        # # ------

        # ------其他学习
        text_feature_residuals = self.text_feature_residuals
        # ------


        final_text_features = text_features  + self.alpha * text_feature_residuals #   t + a * x
        return final_text_features


class CustomClipTextEncoder(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.text_encoder = TextEncoder(clip_model)
        self.text_features_origin = get_base_text_features(cfg, classnames, clip_model, self.text_encoder) # [ num_classes, cfg['embed_dim'] ]


        integer_part = int(cfg['multiply_factor'])
        fractional_part = cfg['multiply_factor'] - integer_part
        
        # 整数部分的重复
        integer_repeated_tensor = self.text_features_origin.repeat(integer_part, 1)
        
        # 小数部分的处理
        if fractional_part > 0:
            # 计算需要额外添加的行数
            extra_rows = max(1, int(fractional_part * self.text_features_origin.size(0)))
            # 截取原始张量的一部分来模拟小数部分的重复
            fractional_tensor = self.text_features_origin[:extra_rows]
            # 如果有整数部分，就将两部分连接起来；如果没有，则只返回小数部分
            result_tensor = torch.cat((integer_repeated_tensor, fractional_tensor), dim=0) if integer_part > 0 else fractional_tensor
        else:
            result_tensor = integer_repeated_tensor

        # self.text_features_repeat = self.text_features_origin.repeat(cfg['multiply_factor'], 1)
        self.text_features_repeat = result_tensor
        print(f"self.text_features_repeat.shape: {self.text_features_repeat.shape}")
        
        self.text_residual_learner = TextResLearner(cfg, classnames, clip_model)
        self.alpha = cfg['text_feature_alpha']
        self.dtype = clip_model.dtype
        self.cfg = cfg
        

    def forward(self):
        self.text_features_repeat = self.text_features_repeat.type(self.dtype).to(self.cfg['device'])

        final_text_features = self.text_residual_learner(self.text_features_repeat)

        return final_text_features



class CustomClipImageEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.alpha = cfg['img_feature_alpha']
        self.dtype = clip_model.dtype

        

    def forward(self, image):
        image_features, local_image_features = self.image_encoder(image.type(self.dtype))

        final_image_features = image_features   
        return final_image_features.type(self.dtype), local_image_features.type(self.dtype)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.clip_image_encoder = CustomClipImageEncoder(cfg, clip_model)
        self.clip_text_encoder = CustomClipTextEncoder(cfg, classnames, clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features, local_image_features = self.clip_image_encoder(image.type(self.dtype))
        text_features = self.clip_text_encoder().type(self.dtype)
        # print(text_features.shape)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits =  logit_scale *image_features @ text_features.t()  #
        logits_local = logit_scale * local_image_features @ text_features.T

        return logits, logits_local



def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/id_dataset_taskres.yaml", help='settings in yaml format')
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

    multiply_factor = 3 #
    cfg['multiply_factor'] = multiply_factor
    print(f"multiply_factor: {multiply_factor}")

    model_name = 'taskres'
    cache_dir = os.path.join('./mycaches_id', cfg['id_dataset'], model_file_name_dict[cfg['backbone']], str(cfg['shots'])+'shots',model_name+'_alllearn_batchz250_ep200','lenx'+str(multiply_factor),'lr0002'  )  #  new_test   seed1
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir


    # seed
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])


    # Contrast learning stage    
    print("Contrast learning stage")
    train_transform_aug = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    train_transform_no_aug = preprocess


    is_train = 0

    if is_train == 1:
        sys.stdout = Logger( os.path.join(cache_dir,'log_train.txt') ,  stream=sys.stdout)
        print("\nRunning configs.")
        print(cfg, "\n")

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


        # cfg['template'] =  ['This is a medical pathology image of {}']
        # cfg['classnames'] = id_dataset.classnames  #



        # stage2 dataset
        few_shot_dataset_stage2 = build_dataset(cfg['id_dataset'], cfg['root_path'], cfg['shots'])
        batch_size_stage2 = cfg['fine_tune_batch_size']

        val_loader = build_data_loader(data_source=few_shot_dataset_stage2.val, batch_size=batch_size_stage2, is_train=False, tfm=train_transform_no_aug, shuffle=False)
        test_loader = build_data_loader(data_source=few_shot_dataset_stage2.test, batch_size=batch_size_stage2, is_train=False, tfm=train_transform_no_aug, shuffle=False)

        train_loader = build_data_loader(data_source=few_shot_dataset_stage2.train_x, batch_size=batch_size_stage2, tfm=train_transform_aug, is_train=True, shuffle=True)
        
        cfg['template'] = few_shot_dataset_stage2.template
        cfg['classnames'] = few_shot_dataset_stage2.classnames  #




        print(f"template: {cfg['template'] }")
        print(f"classnames: {cfg['classnames'] }")
        print(f"len of origin classnames: {len(cfg['classnames'])}")




        # Model_stage2
        model_stage2 = CustomCLIP(cfg,cfg['classnames'] , clip_model)
        model_stage2 = torch.nn.DataParallel(model_stage2).to(device)
        

        for name, param in model_stage2.named_parameters():
            if  "text_residual_learner" in name : # or"img_res_contrastive_learner" in name
                param.requires_grad_(True)
                print(name)
            else:
                param.requires_grad_(False)
                # print(name,'False')



        criterion = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model_stage2.parameters(), lr=cfg['lr'], weight_decay=5e-4,momentum=0.9,dampening=0,nesterov=False)
        optimizer = torch.optim.Adam(model_stage2.parameters(), lr=cfg['lr'], weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,  cfg['fine_tune_train_epoch'] * len(train_loader))

        train_epoch = cfg['fine_tune_train_epoch']
        best_test_acc, best_epoch = 0.0, 0

        for train_idx in range(train_epoch):
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, train_epoch))

            model_stage2.train()
            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()

                logits,_  = model_stage2(images)
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
            if (train_idx + 1) % 100 == 0 or  train_idx == train_epoch - 1     :   # or train_idx == 0 or train_idx == 1  
                model_stage2.eval()
                with torch.no_grad():
                    correct_samples, all_samples = 0, 0
                    topk = 1

                    # correct_items = 0
                    # total_items = 0

                    for images, labels in tqdm(test_loader):
                        images, labels = images.to(device), labels.to(device)
                        outputs,_  = model_stage2(images)



                        # template = 'this is a photo of '
                        # text = clip.tokenize([template + l for l in skin40.classnames]).to(device)
                        # logits_per_image, logits_per_text = clip_model(images, text)
                        # pred_t = logits_per_image.topk(topk, 1, True, True)[1].t()
                        # correct_t = pred_t.eq(labels.view(1, -1).expand_as(pred_t))
                        # cur_correct_items_t = int(correct_t[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())

                        # correct_items += cur_correct_items_t
                        # total_items += labels.shape[0]



                        pred = outputs.topk(topk, 1, True, True)[1].t()
                        correct = pred.eq(labels.view(1, -1).expand_as(pred))
                        acc_num = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                        correct_samples += acc_num
                        all_samples += labels.shape[0]
                    test_acc = correct_samples / all_samples
                    print(f'test Epoch [{train_idx+1}/{train_epoch}], Test accuracy of the model on the test images: { 100 * test_acc :.2f}%')

                    # acc = 100 * correct_items / total_items
                    # print(f'Accuracy: {acc:.2f}%')


                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        best_epoch = train_idx

                        # save model
                        torch.save(model_stage2.state_dict(), os.path.join(cfg['cache_dir'], 'model.pth'))

            
    else:
        # test ood
        sys.stdout = Logger( os.path.join(cache_dir,f'log_test_ood_GL.txt') ,  stream=sys.stdout)
        print("\nRunning configs.")
        print(cfg, "\n")

        # # id dataset
        # id_dataset = IdDataset(cfg['root_path'], cfg['id_dataset'], cfg['shots'])

        # test_path = id_dataset.test_path
        # test_targets = id_dataset.test_targets
        # test_dataset = CustomDataset(test_path, test_targets, train_transform_no_aug)

        # test_loader_id = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['fine_tune_batch_size'] , shuffle=True, num_workers=8)


        # cfg['template'] =  ['This is a medical pathology image of {}']
        # cfg['classnames'] = id_dataset.classnames 


        # stage2 dataset
        few_shot_dataset_stage2 = build_dataset(cfg['id_dataset'], cfg['root_path'], cfg['shots'])
        batch_size_stage2 = cfg['fine_tune_batch_size']

        test_loader_id = build_data_loader(data_source=few_shot_dataset_stage2.test, batch_size=batch_size_stage2, is_train=False, tfm=train_transform_no_aug, shuffle=False)

        cfg['template'] = few_shot_dataset_stage2.template
        cfg['classnames'] = few_shot_dataset_stage2.classnames


        print(f"template: {cfg['template'] }")
        print(f"classnames: {cfg['classnames'] }")
        print(f"len of origin classnames: {len(cfg['classnames'])}")
        


        # ood dataset
        ood_dataset_name = cfg['ood_dataset']
        if cfg['id_dataset'] == 'imagenet':
            ood_dataset_name = ['iNaturalist','SUN','Places','dtd']
        else:
            ood_dataset_name = [ood_dataset_name]





        # Model_stage2
        model_stage2 = CustomCLIP(cfg, cfg['classnames'], clip_model)
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
                # print(logits_id.shape)
                # print(logits_id_local.shape)

                smax_global = F.softmax(logits_id/T, dim=-1).cpu().numpy()
                mcm_global_score = np.max(smax_global, axis=1)

                smax_local = F.softmax(logits_id_local/T, dim=-1).cpu().numpy()
                mcm_local_score = np.max(smax_local, axis=(1, 2))
                # print(logits_id.size)

                id_conf = np.concatenate((id_conf, mcm_global_score))
                id_conf_gl = np.concatenate((id_conf_gl, mcm_global_score + mcm_local_score))

                # caculate accuracy
                pred = logits_id.topk(topk, 1, True, True)[1].t()
                correct = pred.eq(labels.view(1, -1).expand_as(pred))
                acc_num = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                correct_samples += acc_num
                all_samples += labels.shape[0]

                logits_id_local_gl = logits_id_local.max(dim=1)[0] + logits_id
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
                elif cur_ood_dataset in ['iNaturalist','SUN','Places']:
                    ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset), transform=train_transform_no_aug)
                elif cur_ood_dataset == 'dtd':
                    ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset, 'images'), transform=train_transform_no_aug)
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