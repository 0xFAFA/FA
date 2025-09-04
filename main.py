
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
is_train = 0

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
from model import CustomCLIP 


from mydataset import build_dataset
from mydataset.utils import build_data_loader

from ood_utils.ood_tool import get_measures

    



def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/myconfig.yaml", help='settings in yaml format')
    args = parser.parse_args()

    return args

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_gpus = torch.cuda.device_count()
    print(f"GPU num：{num_gpus}")

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

    model_path = os.path.join("/data/xinhua/.cache/clip/", model_file_name_dict[cfg['backbone']]+".pt" )
    model_temp = torch.jit.load(model_path, map_location=device ).eval()
    state_dict = model_temp.state_dict()


    cfg['embed_dim'] = state_dict["text_projection"].shape[1]
    # print(cfg['embed_dim'])

       

    K_name = str(cfg['K']).replace('.','-')
    lr_name = str(cfg['lr']).replace('.','')

    model_name = 'FA'
    cache_dir = os.path.join('./mycaches', cfg['id_dataset'], model_file_name_dict[cfg['backbone']], str(cfg['shots'])+'shots',model_name+'_efficient_batchs'+str(cfg['fine_tune_batch_size'])+'_ep'+str(cfg['fine_tune_train_epoch'])+'','K-'+str(K_name),'lr'+lr_name,'seed'+str(cfg['seed'])  )  #   

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



        # dataset
        few_shot_dataset = build_dataset(cfg['id_dataset'], cfg['root_path'], cfg['shots'])
        batch_size = cfg['fine_tune_batch_size']

        val_loader = build_data_loader(data_source=few_shot_dataset.val, batch_size=batch_size, is_train=False, tfm=train_transform_no_aug, shuffle=False)
        test_loader = build_data_loader(data_source=few_shot_dataset.test, batch_size=batch_size, is_train=False, tfm=train_transform_no_aug, shuffle=False)

        train_loader = build_data_loader(data_source=few_shot_dataset.train_x, batch_size=batch_size, tfm=train_transform_aug, is_train=True, shuffle=True)
        
        
        cfg['classnames'] = few_shot_dataset.classnames # 



        print(f"template: {cfg['template'] }")
        print(f"classnames: {cfg['classnames'] }")
        print(f"len of classnames: {len(cfg['classnames'])}")
        print(f"K: {cfg['K']}")


        # Model
        model = CustomCLIP(cfg, cfg['classnames'], clip_model ,cfg['template'], False, 16, cfg['csc'])
        model = torch.nn.DataParallel(model).to(device)
        

        for name, param in model.named_parameters():
            if  "prompt_learner" in name : #  
                param.requires_grad_(True)
                print(name)
            else:
                param.requires_grad_(False)



        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], weight_decay=5e-4 ,momentum=0.9 ,dampening=0 ,nesterov=False)  # 
        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,  cfg['fine_tune_train_epoch'] * len(train_loader))

        print(optimizer)
        print(scheduler)

        train_epoch = cfg['fine_tune_train_epoch']

        for train_idx in range(train_epoch):
            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(train_idx, train_epoch))

            model.train()
            for i, (images, target) in enumerate(tqdm(train_loader)):
                images, target = images.cuda(), target.cuda()

                logits,_ = model(images)
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
            if  train_idx == train_epoch - 1 :   #  (train_idx + 1) % 100 == 0 or  train_idx == 0  or # 待修改为取最后一个epoch
                model.eval()
                with torch.no_grad():
                    correct_samples, all_samples = 0, 0
                    topk = 1

                    for images, labels in tqdm(test_loader):
                        images, labels = images.to(device), labels.to(device)
                        outputs,_ = model(images)

                        outputs = outputs / 100.0
                        batch_size, repeat_len = outputs.shape
                        outputs_temp = outputs.view(batch_size, 1+cfg['K'], len(cfg['classnames']))
                        just_forced_outputs = outputs_temp[:,0,:]


                        pred = just_forced_outputs.topk(topk, 1, True, True)[1].t()
                        correct = pred.eq(labels.view(1, -1).expand_as(pred))
                        acc_num = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                        correct_samples += acc_num
                        all_samples += labels.shape[0]

                    test_acc = correct_samples / all_samples
                    print(f'test Epoch [{train_idx+1}/{train_epoch}], Test accuracy of the model on the test images: { 100 * test_acc :.2f}%')


                    # save model
                    torch.save(model.state_dict(), os.path.join(cfg['cache_dir'], 'model.pth'))

            
    else:
        test_file_name = f'log_test_ood_{cfg["ood_dataset"]}.txt'
        sys.stdout = Logger( os.path.join(cache_dir, test_file_name) ,  stream=sys.stdout)
        print("\nRunning configs.")
        print(cfg, "\n")

        print("augment transform:")
        for transform in train_transform_no_aug.transforms:
            print(transform)

        # test ood

        # id dataset
        few_shot_dataset = build_dataset(cfg['id_dataset'], cfg['root_path'], cfg['shots'])
        batch_size = cfg['test_batch_size']

        test_loader_id = build_data_loader(data_source=few_shot_dataset.test, batch_size=batch_size, is_train=False, tfm=train_transform_no_aug, shuffle=False)

        cfg['classnames'] = few_shot_dataset.classnames  #



        print(f"template: {cfg['template'] }")
        print(f"classnames: {cfg['classnames'] }")
        print(f"len of classnames: {len(cfg['classnames'])}")
        print(f"K: {cfg['K']}")


        # ood dataset
        ood_dataset_name = cfg['ood_dataset']

        if ood_dataset_name == 'challenging':
            ood_dataset_name = ['OpenImage_O','NINCO','imagenet-o' ]
        elif ood_dataset_name == 'all':
            ood_dataset_name = ['iNaturalist','SUN','Places','dtd','OpenImage_O','NINCO','imagenet-o']
        elif ood_dataset_name == 'nearood':
            ood_dataset_name = ['ssb_hard','NINCO']
        elif ood_dataset_name == 'common':
            ood_dataset_name = ['iNaturalist','SUN','Places','dtd']
        else:
            ood_dataset_name = [ood_dataset_name]


        print(f"ood_dataset_name: {ood_dataset_name}")

        # Model
        model = CustomCLIP(cfg, cfg['classnames'], clip_model ,cfg['template'], False, 16, cfg['csc'])
        model = torch.nn.DataParallel(model).to(device)


        # load model
        model.load_state_dict(torch.load(os.path.join(cfg['cache_dir'], 'model.pth')))

        model.eval()
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

                logits_id, logits_id_local = model(images)
                logits_id /= 100.0
                logits_id_local /= 100.0

                # ----
                batch_size, token_len, repeat_len = logits_id_local.shape
                logits_id_temp = logits_id.view(batch_size, 1+cfg['K'], len(cfg['classnames']))
                just_forced_logits_id = logits_id_temp[:,0,:]


                logits_id_local_temp = logits_id_local.view(batch_size, token_len, 1+cfg['K'], len(cfg['classnames']))
                just_forced_logits_id_local = logits_id_local_temp[:,:,0,:]
                # ----


                smax_global = F.softmax(logits_id/T, dim=-1).cpu().numpy()
                mcm_global_score = np.max(smax_global, axis=1)

                smax_local = F.softmax(logits_id_local/T, dim=-1).cpu().numpy()
                mcm_local_score = np.max(smax_local, axis=(1, 2))
                # print(logits_id.size)

                id_conf = np.concatenate((id_conf, mcm_global_score))
                id_conf_gl = np.concatenate((id_conf_gl, mcm_global_score + mcm_local_score))


                # caculate accuracy
                pred = just_forced_logits_id.topk(topk, 1, True, True)[1].t()
                correct = pred.eq(labels.view(1, -1).expand_as(pred))
                acc_num = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                correct_samples += acc_num
                all_samples += labels.shape[0]

                logits_id_local_gl = just_forced_logits_id_local.max(dim=1)[0] + just_forced_logits_id
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
                elif cur_ood_dataset in ['iNaturalist','SUN','Places','OpenImage_O','imagenet-o','ssb_hard']:
                    ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset), transform=train_transform_no_aug)
                elif cur_ood_dataset == 'dtd':
                    ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset, 'images'), transform=train_transform_no_aug)
                elif cur_ood_dataset == 'NINCO':
                    ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset, 'NINCO_OOD_classes'), transform=train_transform_no_aug)
                else:
                    ood_dataset = datasets.ImageFolder(root=os.path.join(cfg['ood_dataset_path'], cur_ood_dataset, 'test'), transform=train_transform_no_aug)
                
                ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=cfg['test_batch_size'],
                                                            shuffle=False, num_workers=8)


                for images, _ in tqdm(ood_loader):
                    images = images.to(device)

                    logits_ood, logits_ood_local = model(images)
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