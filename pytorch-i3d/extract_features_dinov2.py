import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-model_size', type=str, help='dinov2 model size', default='dinov2_vitb14', 
                    choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'])
parser.add_argument('-root', type=str, default='Charades_v1_rgb', help='root directory of the dataset')
parser.add_argument('-gpu', type=str, default='0',)
parser.add_argument('-save_dir', type=str, default='data/dinov2', help='directory to save extracted features')
parser.add_argument('-batch_size', type=int, default=1, help='batch size for processing')
parser.add_argument('-split_file', type=str, default='charades.json', help='split file path')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

from tqdm import tqdm
import numpy as np
import cv2

# DINOv2 모델 import
try:
    # 먼저 transformers 라이브러리로 시도
    from transformers import Dinov2Model, Dinov2Config
    USE_TRANSFORMERS = True
except ImportError:
    try:
        # torch hub으로 시도
        USE_TRANSFORMERS = False
    except ImportError:
        print("Error: DINOv2 not available. Please install transformers or use torch.hub")
        sys.exit(1)

from charades_dataset_full import Charades as Dataset


class DINOv2FeatureExtractor:
    def __init__(self, model_size='dinov2_vitb14', device='cuda'):
        self.device = device
        self.model_size = model_size
        
        # DINOv2 모델 로드
        if USE_TRANSFORMERS:
            self.model = Dinov2Model.from_pretrained(f"facebook/{model_size}")
        else:
            self.model = torch.hub.load('facebookresearch/dinov2', model_size)
        
        self.model.to(device)
        self.model.eval()
        
        # 이미지 전처리 (DINOv2에 맞는 normalize)
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded DINOv2 model: {model_size}")
        
    def extract_frame_features(self, frame):
        """단일 프레임에서 특징 추출"""
        # frame: numpy array (H, W, C) in range [0, 255]
        frame_tensor = self.preprocess(frame).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if USE_TRANSFORMERS:
                outputs = self.model(frame_tensor)
                features = outputs.last_hidden_state[:, 0]  # CLS token
            else:
                features = self.model(frame_tensor)
                
        return features.cpu().numpy()
    
    def extract_video_features(self, frames):
        """비디오 프레임들에서 특징 추출"""
        # frames: numpy array (T, H, W, C)
        features_list = []
        
        for i in range(frames.shape[0]):
            frame = frames[i]  # (H, W, C)
            # normalize back to [0, 255] if needed
            if frame.max() <= 1.0:
                frame = (frame + 1) * 127.5  # [-1, 1] -> [0, 255]
            frame = frame.astype(np.uint8)
            
            features = self.extract_frame_features(frame)
            features_list.append(features)
        
        return np.concatenate(features_list, axis=0)  # (T, feature_dim)


def load_rgb_frames_for_dinov2(image_dir, vid, start, num):
    """DINOv2용 RGB 프레임 로드 (정규화 없이)"""
    frames = []
    target_size = (224, 224)
    
    for i in range(start, start+num):
        img_path = os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg')
        
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            if img is None or img.size == 0:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = img[:, :, [2, 1, 0]]  # BGR to RGB
                
                # 이미지 크기 조정
                h, w, c = img.shape
                if h != target_size[0] or w != target_size[1]:
                    if h < 226 or w < 226:
                        d = 226.-min(h,w)
                        sc = 1+d/min(h,w)
                        img = cv2.resize(img, dsize=(0,0), fx=sc, fy=sc)
                    # 최종적으로 224x224로 크기 맞춤
                    img = cv2.resize(img, target_size)
                    
        except Exception as e:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        frames.append(img)
    
    return np.asarray(frames, dtype=np.uint8)  # (T, H, W, C)


def run(model_size='dinov2_vitb14', root='Charades_v1_rgb', split_file='charades.json', batch_size=1, save_dir=''):
    # save_dir 검증 및 생성
    if not save_dir:
        save_dir = './dinov2_features'
    os.makedirs(save_dir, exist_ok=True)
    print(f"DINOv2 features will be saved to: {save_dir}")
    
    # DINOv2 특징 추출기 초기화
    feature_extractor = DINOv2FeatureExtractor(model_size=model_size)
    
    # setup dataset (transform은 사용하지 않음 - DINOv2가 자체 전처리 함)
    test_transforms = transforms.Compose([])  # 빈 transform
    
    # 데이터셋 로드 (training과 testing 모두 처리)
    for split in ['training', 'testing']:
        print(f"\nProcessing {split} split...")
        
        try:
            dataset = Dataset(split_file, split, root, 'rgb', test_transforms, num=-1, save_dir=save_dir)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                                    num_workers=2, pin_memory=False)
            
            if len(dataset) == 0:
                print(f"No data found for {split} split")
                continue
                
            print(f"Found {len(dataset)} videos in {split} split")
            
            # 각 비디오 처리
            for data in tqdm(dataloader, desc=f"Extracting {split} features"):
                if len(data) == 3:
                    inputs, labels, name = data
                    video_name = name[0]
                else:
                    print("Warning: Unexpected data format")
                    continue
                
                # 이미 처리된 파일 스킵
                output_path = os.path.join(save_dir, f"{video_name}.npy")
                if os.path.exists(output_path):
                    continue
                
                try:
                    # 직접 프레임 로드 (charades_dataset_full의 transform 대신)
                    vid_data = dataset.data[dataloader.dataset.data.index(
                        next(item for item in dataset.data if item[0] == video_name)
                    )]
                    vid, label, dur, nf = vid_data
                    
                    print(f"Processing {video_name}... ({nf} frames)")
                    
                    # GPU 메모리 효율성을 위해 청크 단위로 처리
                    chunk_size = 100  # 한 번에 처리할 프레임 수
                    all_features = []
                    
                    for start_idx in range(1, nf, chunk_size):
                        end_idx = min(start_idx + chunk_size, nf)
                        
                        # 프레임 로드
                        frames = load_rgb_frames_for_dinov2(root, vid, start_idx, end_idx - start_idx)
                        
                        # DINOv2 특징 추출
                        features = feature_extractor.extract_video_features(frames)
                        all_features.append(features)
                        
                        # 메모리 정리
                        del frames
                        torch.cuda.empty_cache()
                    
                    # 모든 특징 결합
                    final_features = np.concatenate(all_features, axis=0)
                    
                    # 특징 저장
                    np.save(output_path, final_features)
                    print(f"Saved DINOv2 features for {video_name}: {final_features.shape}")
                    
                    # 메모리 정리
                    del all_features, final_features
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing video {video_name}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error processing {split} split: {e}")
            continue

    print("DINOv2 feature extraction completed!")


if __name__ == '__main__':
    run(model_size=args.model_size, root=args.root, split_file=args.split_file, 
        batch_size=args.batch_size, save_dir=args.save_dir)
