import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow', default='rgb')
parser.add_argument('-load_model', type=str, default='pytorch-i3d/models/original/rgb_charades.pt', help='path to the model to load')
parser.add_argument('-root', type=str, default='Charades_v1_rgb', help='root directory of the dataset')
parser.add_argument('-gpu', type=str, default='0',)
parser.add_argument('-save_dir', type=str, default='data', help='directory to save extracted features')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

from tqdm import tqdm

import numpy as np

from pytorch_i3d import InceptionI3d

from charades_dataset_full import Charades as Dataset


def run(max_steps=64e3, mode='rgb', root='Charades_v1_rgb', split='charades.json', batch_size=1, load_model='', save_dir=''):
    # save_dir 검증 및 생성
    if not save_dir:
        save_dir = './extracted_features'
    os.makedirs(save_dir, exist_ok=True)
    print(f"Features will be saved to: {save_dir}")
    
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

    val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    for phase in ['val']:
        i3d.train(False)  # Set model to evaluate mode
                
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
                    
        # Iterate over data.
        for data in tqdm(dataloaders[phase]):
            # get the inputs
            inputs, labels, name = data
            if os.path.exists(os.path.join(save_dir, name[0]+'.npy')):
                continue

            b,c,t,h,w = inputs.shape
            print(f"Processing {name[0]}... Video shape: {inputs.shape}")
            
            with torch.no_grad():  # 메모리 효율성 향상
                if t > 1600:
                    features = []
                    for start in range(1, t-56, 1600):
                        end = min(t-1, start+1600+56)
                        start = max(1, start-48)
                        # 불필요한 변환 제거
                        ip = inputs[:,:,start:end].cuda()
                        feat = i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).cpu().numpy()
                        features.append(feat)
                        # 명시적으로 GPU 메모리 해제
                        del ip
                        torch.cuda.empty_cache()
                    np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
                    del features  # 메모리 해제
                else:
                    # 불필요한 Variable wrapper 제거
                    inputs_gpu = inputs.cuda()
                    features = i3d.extract_features(inputs_gpu)
                    np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1,2,3,0).cpu().numpy())
                    # 명시적으로 GPU 메모리 해제
                    del inputs_gpu, features
                    torch.cuda.empty_cache()
                    
                print(f"Saved features for {name[0]} to {save_dir}")


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
