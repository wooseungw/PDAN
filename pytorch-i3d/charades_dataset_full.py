import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py

import os
import os.path

import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(image_dir, vid, start, num):
  frames = []
  target_size = (224, 224)  # 목표 크기 설정
  
  for i in range(start, start+num):
    img_path = os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg')
    img = cv2.imread(img_path)
    
    if img is None:
      print(f"Warning: Could not load image {img_path}")
      # 검은색 이미지로 대체 (224x224x3)
      img = np.zeros((224, 224, 3), dtype=np.uint8)
    else:
      img = img[:, :, [2, 1, 0]]  # BGR to RGB
    
    w,h,c = img.shape
    # 항상 목표 크기로 리사이즈
    if w != target_size[0] or h != target_size[1]:
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
            img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
        # 최종적으로 224x224로 크기 맞춤
        img = cv2.resize(img, target_size)
    
    img = (img/255.)*2 - 1
    frames.append(img)
  
  # 모든 프레임이 같은 크기인지 확인
  frames_array = np.asarray(frames, dtype=np.float32)
  print(f"Loaded {len(frames)} frames with shape: {frames_array.shape}")
  return frames_array

def load_flow_frames(image_dir, vid, start, num):
  frames = []
  target_size = (224, 224)  # 목표 크기 설정
  
  for i in range(start, start+num):
    imgx_path = os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg')
    imgy_path = os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg')
    
    imgx = cv2.imread(imgx_path, cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(imgy_path, cv2.IMREAD_GRAYSCALE)
    
    if imgx is None or imgy is None:
      print(f"Warning: Could not load flow images {imgx_path} or {imgy_path}")
      # 검은색 이미지로 대체 (224x224)
      imgx = np.zeros(target_size, dtype=np.uint8)
      imgy = np.zeros(target_size, dtype=np.uint8)
    else:
      # 항상 목표 크기로 리사이즈
      w,h = imgx.shape
      if w != target_size[0] or h != target_size[1]:
          if w < 224 or h < 224:
              d = 224.-min(w,h)
              sc = 1+d/min(w,h)
              imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
              imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
          # 최종적으로 224x224로 크기 맞춤
          imgx = cv2.resize(imgx, target_size)
          imgy = cv2.resize(imgy, target_size)
        
    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  
  # 모든 프레임이 같은 크기인지 확인
  frames_array = np.asarray(frames, dtype=np.float32)
  print(f"Loaded {len(frames)} flow frames with shape: {frames_array.shape}")
  return frames_array


def make_dataset(split_file, split, root, mode, num_classes=157):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)
    print('Found {} videos in split {}'.format(len(data), split))
    print('Loading videos from {}'.format(root))
    i = 0
    for vid in data.keys():
        print('Processing video {} ({}/{})'.format(vid, i+1, len(data)))
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid)):
            continue
        num_frames = len(os.listdir(os.path.join(root, vid)))
        if mode == 'flow':
            num_frames = num_frames//2
            
        label = np.zeros((num_classes,num_frames), np.float32)

        fps = num_frames/data[vid]['duration']
        for ann in data[vid]['actions']:
            for fr in range(0,num_frames,1):
                if fr/fps > ann[1] and fr/fps < ann[2]:
                    label[ann[0], fr] = 1 # binary classification
        dataset.append((vid, label, data[vid]['duration'], num_frames))
        i += 1
    
    return dataset


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, save_dir='', num=0):
        
        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, dur, nf = self.data[index]
        if os.path.exists(os.path.join(self.save_dir, vid+'.npy')):
            return 0, 0, vid

        try:
            if self.mode == 'rgb':
                imgs = load_rgb_frames(self.root, vid, 1, nf)
            else:
                imgs = load_flow_frames(self.root, vid, 1, nf)

            imgs = self.transforms(imgs)
            return video_to_tensor(imgs), torch.from_numpy(label), vid
        
        except Exception as e:
            print(f"Error processing video {vid}: {e}")
            # 에러 발생 시 더미 데이터 반환
            if self.mode == 'rgb':
                dummy_imgs = np.zeros((64, 224, 224, 3), dtype=np.float32)  # 64프레임 더미
            else:
                dummy_imgs = np.zeros((64, 224, 224, 2), dtype=np.float32)  # flow용 더미
            
            dummy_imgs = self.transforms(dummy_imgs)
            dummy_label = np.zeros_like(label)
            return video_to_tensor(dummy_imgs), torch.from_numpy(dummy_label), vid

    def __len__(self):
        return len(self.data)
