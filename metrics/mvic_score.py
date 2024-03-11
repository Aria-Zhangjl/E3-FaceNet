# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
from . import metric_utils
from torch.utils import data
from glob import glob
import os
import os.path as osp
from PIL import Image
import torch
import clip
import torch.nn.functional as F
def mean(list):
    all_sum = sum(list)
    cnt = len(list)
    return all_sum / cnt
#----------------------------------------------------------------------------
class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, path1, path2, transform=None):
        'Initialization'
        self.text_names, self.images_names = self.get_filenames(path1, path2)
        assert len(self.text_names) == len(self.images_names)
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images_names)

    def __getitem__(self, index):
        'Generates one sample of data'
        img = Image.open(self.images_names[index]).convert('RGB')
        with open(self.text_names[index], 'r') as f:
            txt = f.read()

        # Convert image and label to torch tensors
        if self.transform is not None:
            img = self.transform(img)
        
        return txt, img

    def get_filenames(self, text_path, image_path):
        texts = glob(os.path.join(text_path, '*.txt'))
        # print(texts[0])
        images = glob(os.path.join(image_path, '*.png')) + glob(os.path.join(image_path, '*.jpg'))
        # print(images[0])
        _ext = osp.splitext(images[0])[-1]

        text_names = [osp.splitext(osp.split(file)[-1])[0] for file in texts]
        image_names = [osp.splitext(osp.split(file)[-1])[0] for file in images]
        inter_names = list(set(text_names).intersection(set(image_names)))
        inter_names.sort()

        texts = [osp.join(text_path, filename + '.txt') for filename in inter_names]
        images = [osp.join(image_path, filename + _ext) for filename in inter_names]

        return texts, images

def calculate_clip_cosine(paths, batch_size, device,backbone):
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
    model, preprocess = clip.load(backbone, device=device)
    model.eval()
    
    dataset = Dataset(paths[0], paths[1], preprocess)
    print(dataset.__len__())
    
    dataloader = data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False, 
        num_workers=8
    )

    d0 = dataloader.__len__() * batch_size
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    sim_arr = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            start = i * batch_size
            end = start + batch_size
            text, image = batch

            text = clip.tokenize(text).to(device)
            image = image.to(device)

            text_features,_ = model.encode_text(text)
            image_features = model.encode_image(image)

            cos_sim = F.cosine_similarity(text_features,image_features,dim=1)
            sim_arr.extend(cos_sim.tolist())
    print('done')
    return mean(sim_arr)


def compute_clip_sim(opts, batch_size=16,device='cuda',backbone='ViT-B/32'):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    text_path = osp.join(opts.sample_dir,'text')
    image_path = osp.join(opts.sample_dir,'image')
    paths = ["",""]
    paths[0] = text_path
    paths[1] = image_path
    print(paths)
    clip_score = calculate_clip_cosine(paths, batch_size, device,backbone)
    return float(clip_score)

#----------------------------------------------------------------------------
