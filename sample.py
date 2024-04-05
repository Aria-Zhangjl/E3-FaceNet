# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
import time
import glob
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import imageio
import legacy
from renderer import Renderer
import clip
import time
#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------
os.environ['PYOPENGL_PLATFORM'] = 'egl'

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--render-program', default=None, show_default=True)
@click.option('--render-option', default=None, type=str, help="e.g. up_256, camera, depth")
@click.option('--n_steps', default=8, type=int, help="number of steps for each seed")
@click.option('--no-video', default=False)
@click.option('--relative_range_u_scale', default=1.5, type=float, help="relative scale on top of the original range u")
@click.option('--gen_description', default='The man is young and has blond hair.', type=str, help="The input text")
@click.option('--edit_description', default='The man has black hair.', type=str, help="The edit text")
@click.option('--each_sample', default=1, type=int, help="The number of one term")
@click.option('--alpha', default=1., type=float, help="The weight of the original text token")
@click.option('--save_idx', default=None, type=str, help="The save idx of the generated 3D face")
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    gen_description:str,
    edit_description:str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    render_program=None,
    render_option=None,
    n_steps=8,
    no_video=False,
    relative_range_u_scale=1.5,
    each_sample=1,
    alpha=1.,
    save_idx = None,
):
    device = torch.device('cuda')
    clip_model, _ = clip.load("ViT-B/32", device=device,jit=False)
    clip_model.eval()
    if os.path.isdir(network_pkl):
        network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % network_pkl)
    
    with dnnlib.util.open_url(network_pkl) as f:
        network = legacy.load_network_pkl(f)
        G = network['G_ema'].to(device) # type: ignore
        D = network['D'].to(device)
    # from fairseq import pdb;pdb.set_trace()
    if save_idx == None:
        outdir = os.path.join(outdir,gen_description.replace(' ','_')+'edit_'+edit_description.replace(' ','_'))
    else:
        outdir = os.path.join(outdir,save_idx)
    os.makedirs(outdir, exist_ok=True)

    # TODO modified to text conditioned
    # Done!
    if G.t_dim != 0:
        if edit_description == None or gen_description == None:
            ctx.fail('Must specify text description when using a conditional network')
        gen_t_ids = clip.tokenize(gen_description).to(device)
        gen_t_embds,gen_t_token = clip_model.encode_text(gen_t_ids)
        gen_t_embds = gen_t_embds.detach()
        edit_t_ids = clip.tokenize(edit_description).to(device)
        _,edit_t_token = clip_model.encode_text(edit_t_ids)
        edit_t_token = edit_t_token.detach()
    # avoid persistent classes... 
    from training.networks_4_E3_Face import Generator
    # from training.stylenerf import Discriminator
    from torch_utils import misc
    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
        # D2 = Discriminator(*D.init_args, **D.init_kwargs).to(device)
        # misc.copy_params_and_buffers(D, D2, require_all=False)
    G2 = Renderer(G2, D, program=render_program)
    
    # Generate images.
    all_imgs = []

    def stack_imgs(imgs):
        img = torch.stack(imgs, dim=2)
        return img.reshape(img.size(0) * img.size(1), img.size(2) * img.size(3), 3)

    def proc_img(img): 
        return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

    if projected_w is not None:
        ws = np.load(projected_w)
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        img = G2(styles=ws, truncation_psi=truncation_psi, noise_mode=noise_mode, render_option=render_option)
        assert isinstance(img, List)
        imgs = [proc_img(i) for i in img]
        all_imgs += [imgs]
    
    else:
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            G2.set_random_seed(seed)
            z = torch.from_numpy(np.random.RandomState(seed).randn(each_sample, G.z_dim)).to(device)
            relative_range_u = [0.5 - 0.5 * relative_range_u_scale, 0.5 + 0.5 * relative_range_u_scale]
            gen_outputs = G2(
                z=z,
                t=gen_t_embds,
                token_embeddings = gen_t_token,
                truncation_psi=truncation_psi,
                noise_mode=noise_mode,
                render_option=render_option,
                n_steps=n_steps,
                relative_range_u=relative_range_u,
                return_cameras=True)
            if isinstance(gen_outputs, tuple):
                gen_imgs, gen_cameras = gen_outputs
            else:
                gen_imgs = gen_outputs
            edit_outputs = G2(
                z=z,
                t=gen_t_embds,
                token_embeddings = edit_t_token,
                assist_token_embeddings = gen_t_token,
                assist_alpha = alpha,
                truncation_psi=truncation_psi,
                noise_mode=noise_mode,
                render_option=render_option,
                n_steps=n_steps,
                relative_range_u=relative_range_u,
                return_cameras=True)
            if isinstance(gen_outputs, tuple):
                edit_imgs, edit_cameras = edit_outputs
            else:
                edit_imgs = edit_outputs
            
            if isinstance(gen_imgs, List) and isinstance(edit_imgs,List):
                gen_imgs = [proc_img(i) for i in gen_imgs]
                edit_imgs = [proc_img(i) for i in edit_imgs]
                imgs = [torch.cat((i,j),dim=0) for i,j in zip(gen_imgs,edit_imgs)]
                if not no_video:
                    all_imgs += [imgs]
           
                curr_out_dir = os.path.join(outdir, 'seed_{:0>6d}'.format(seed))
                os.makedirs(curr_out_dir, exist_ok=True)

                if (render_option is not None) and ("gen_ibrnet_metadata" in render_option):
                    intrinsics = []
                    poses = []
                    _, H, W, _ = imgs[0].shape
                    for i, camera in enumerate(gen_cameras):
                        intri, pose, _, _ = camera
                        focal = (H - 1) * 0.5 / intri[0, 0, 0].item()
                        intri = np.diag([focal, focal, 1.0, 1.0]).astype(np.float32)
                        intri[0, 2], intri[1, 2] = (W - 1) * 0.5, (H - 1) * 0.5

                        pose = pose.squeeze().detach().cpu().numpy() @ np.diag([1, -1, -1, 1]).astype(np.float32)
                        intrinsics.append(intri)
                        poses.append(pose)

                    intrinsics = np.stack(intrinsics, axis=0)
                    poses = np.stack(poses, axis=0)

                    np.savez(os.path.join(curr_out_dir, 'cameras.npz'), intrinsics=intrinsics, poses=poses)
                    with open(os.path.join(curr_out_dir, 'meta.conf'), 'w') as f:
                        f.write('depth_range = {}\ntest_hold_out = {}\nheight = {}\nwidth = {}'.
                                format(G2.generator.synthesis.depth_range, 2, H, W))

                gen_img_dir = os.path.join(curr_out_dir, 'gen')
                os.makedirs(gen_img_dir, exist_ok=True)
                edit_img_dir = os.path.join(curr_out_dir, 'edit')
                os.makedirs(edit_img_dir, exist_ok=True)
                for step, img in enumerate(imgs):
                    if len(img) > 1:
                        PIL.Image.fromarray(img[0].detach().cpu().numpy(), 'RGB').save(f'{gen_img_dir}/{step:03d}.png')
                        PIL.Image.fromarray(img[1].detach().cpu().numpy()).save(f'{edit_img_dir}/{step:03d}.png')
                    else:
                        PIL.Image.fromarray(img[0].detach().cpu().numpy(), 'RGB').save(f'{gen_img_dir}/{step:03d}.png')
            else:
                img = proc_img(img)[0]
                PIL.Image.fromarray(img.numpy(), 'RGB').save(f'{outdir}/seed_{seed:0>6d}.png')

    if len(all_imgs) > 0 and (not no_video):
         # write to video
        timestamp = time.strftime('%Y%m%d.%H%M%S',time.localtime(time.time()))
        seeds = ','.join([str(s) for s in seeds]) if seeds is not None else 'projected'
        network_pkl = network_pkl.split('/')[-1].split('.')[0]
        all_imgs = [stack_imgs([a[k] for a in all_imgs]).numpy() for k in range(len(all_imgs[0]))]
        imageio.mimwrite(f'{outdir}/{network_pkl}_{timestamp}_{seeds}.mp4', all_imgs, fps=30, quality=8)
        outdir = f'{outdir}/{network_pkl}_{timestamp}_{seeds}'
        os.makedirs(outdir, exist_ok=True)
        for step, img in enumerate(all_imgs):
            PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/{step:04d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
