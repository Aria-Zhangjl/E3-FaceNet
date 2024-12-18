# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import imp
import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
import tqdm
import shutil
import legacy

from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from metrics import metric_main_delta
import clip
#----------------------------------------------------------------------------

def eval_loop(
    run_dir                 = '.',      # Output directory.
    eval_set_kwargs         = {},       # Options for training set.
    image_folder_kwargs     = None,     # Options for image folder
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    world_size              = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process.
    gpu                     = 0,        # Index of GPU used in training
    batch_gpu               = 4,        # Batch size for once GPU
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * world_size.
    resume_pkl             = None,     # Network pickle to resume training from.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    outdir                  = '.',
    **unused,    
):
    # Initialize.
    device = torch.device('cuda', gpu)
    np.random.seed(random_seed * world_size + rank)
    torch.manual_seed(random_seed * world_size + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.
    sample_dir = run_dir + '/sample'
    os.makedirs(sample_dir, exist_ok=True)
    clip_model, _ = clip.load("ViT-B/32", device=device)
    assert batch_gpu <= (batch_size // world_size)

    # Reload networks.
    if resume_pkl is not None:
        with dnnlib.util.open_url(resume_pkl) as f:
            network = legacy.load_network_pkl(f,init_from_origin=False)
            G = network['G_ema'].to(device) # type: ignore
            del network
    else:
        G = None
    os.makedirs(outdir, exist_ok=True)
    for metric in metrics:
        result_dict = metric_main_delta.calc_metric(metric=metric, G=G,
        dataset_kwargs=eval_set_kwargs, image_folder_kwargs = image_folder_kwargs,num_gpus=world_size, rank=rank, device=device,sample_dir=sample_dir,image_save=True,clip_model = clip_model)
        metric_main_delta.report_metric(result_dict, run_dir=outdir,)

    # Calculate the metric as well as generate the image



    
#----------------------------------------------------------------------------
