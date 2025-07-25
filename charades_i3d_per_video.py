import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
# import h5py

import os
import os.path
from tqdm import tqdm


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    # Handle different numpy array types more robustly
    try:
        # Always create a new numpy array to avoid type issues
        pic_copy = np.array(pic, dtype=np.float32, copy=True)
        
        # Transpose dimensions: (T, H, W, C) -> (C, T, H, W)
        transposed = pic_copy.transpose([3, 0, 1, 2])
        
        # Create tensor directly without torch.from_numpy to avoid type issues
        return torch.tensor(transposed, dtype=torch.float32)
        
    except Exception as e:
        print(f"Error in video_to_tensor: {e}")
        print(f"Input type: {type(pic)}, shape: {pic.shape if hasattr(pic, 'shape') else 'no shape'}")
        
        # Ultimate fallback: manual tensor creation
        if hasattr(pic, 'shape') and len(pic.shape) == 4:
            T, H, W, C = pic.shape
            # Create tensor with correct shape directly
            tensor_data = torch.zeros((C, T, H, W), dtype=torch.float32)
            for t in range(T):
                for h in range(H):
                    for w in range(W):
                        for c in range(C):
                            tensor_data[c, t, h, w] = float(pic[t, h, w, c])
            return tensor_data
        else:
            # Last resort
            return torch.tensor(pic, dtype=torch.float32)


def make_dataset(split_file, split, root, num_classes=157):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)
    print('split!!!!',split)
    i = 0
    for vid in tqdm(data.keys()):
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid+'.npy')):
            continue
        fts = np.load(os.path.join(root, vid+'.npy'))
        num_feat = fts.shape[0]
        label = np.zeros((num_feat,num_classes), np.float32)

        fps = num_feat/data[vid]['duration']
        for ann in data[vid]['actions']:
            for fr in range(0,num_feat,1):
                if fr/fps > ann[1] and fr/fps < ann[2]:
                    label[fr, ann[0]] = 1 # binary classification
        dataset.append((vid, label, data[vid]['duration']))
        i += 1
    
    return dataset

# make_dataset('multithumos.json', 'training', '/ssd2/thumos/val_i3d_rgb')


class MultiThumos(data_utl.Dataset):

    def __init__(self, split_file, split, root, batch_size, classes):
        
        self.data = make_dataset(split_file, split, root, classes)
        self.split_file = split_file
        self.batch_size = batch_size
        self.root = root
        self.in_mem = {}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        entry = self.data[index]
        if entry[0] in self.in_mem:
            feat = self.in_mem[entry[0]]
        else:
            # print('here')
            feat = np.load(os.path.join(self.root, entry[0]+'.npy'))
            # print(feat.shape[-1])
            feat = feat.reshape((feat.shape[0],1,1,feat.shape[-1]))
            feat = feat.astype(np.float32)

            
        label = entry[1]
        return feat, label, [entry[0], entry[2]]

    def __len__(self):
        return len(self.data)


def mt_collate_fn(batch):
    "Pads data and puts it into a tensor of same dimensions"
    max_len = 0
    for b in batch:
        if b[0].shape[0] > max_len:
            max_len = b[0].shape[0]

    new_batch = []
    for b in batch:
        # Create padded arrays with explicit numpy array creation
        f = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), dtype=np.float32)
        m = np.zeros((max_len,), dtype=np.float32)
        l = np.zeros((max_len, b[1].shape[1]), dtype=np.float32)
        
        # Fill with actual data
        f[:b[0].shape[0]] = b[0]
        m[:b[0].shape[0]] = 1.0
        l[:b[0].shape[0], :] = b[1]
        
        # Create tensors more safely
        try:
            f_tensor = video_to_tensor(f)
            m_tensor = torch.tensor(m, dtype=torch.float32)
            l_tensor = torch.tensor(l, dtype=torch.float32)
        except Exception as e:
            print(f"Error in collate_fn: {e}")
            # Fallback tensor creation
            f_tensor = torch.tensor(f.transpose([3,0,1,2]), dtype=torch.float32)
            m_tensor = torch.tensor(m, dtype=torch.float32)
            l_tensor = torch.tensor(l, dtype=torch.float32)
        
        new_batch.append([f_tensor, m_tensor, l_tensor, b[2]])

    return default_collate(new_batch)

