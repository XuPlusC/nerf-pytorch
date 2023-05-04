import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

import cv2
import mvs.depth_map_operate as depthOP
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, use_depth_prior=False, depth_sample_percentile=1., **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], 
                use_depth_prior=use_depth_prior, depth_sample_percentile=depth_sample_percentile, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  use_depth_prior=False,
                  depth_sample_percentile=1.,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
      use_depth_prior: [cococat exp]bool, whether the render procedure should use depth prior
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
        rays_depth = None
    else:
        # use provided ray batch
        # rays_o, rays_d = rays
        rays_o, rays_d, rays_depth = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)    # ccc : 归一化后的光线射线坐标
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    origin_delta = rays_d - rays_o
    origin_norm = torch.norm(origin_delta, p=2, dim=1, keepdim=True)
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
        new_delta = rays_d - rays_o
        new_norm = torch.norm(new_delta, p=2, dim=1, keepdim=True)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    if use_depth_prior is True:
        rays_depth = torch.reshape(rays_depth, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    # rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_depth_prior is True:
        rays = torch.cat([rays_o, rays_d, near, far, rays_depth], -1)
    else:
        rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, use_depth_prior=use_depth_prior, depth_sample_percentile=depth_sample_percentile, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    # ccc: 这个embedder是用来做位置编码的。
    # embed_fn 是编码函数，其输出大概就是把输入的x做各种傅里叶(不同频率的sin cos)变换以后的结果组成的列表
    # input_ch 就是上述输出结果——也就是位置编码结果——的维度。
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'depth_prior':args.depth_prior,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    # ccc : dists是（经过随机扰动的）采样点之间的距离。其最后一个值是1e10，作为边界。
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1) # ccc : 这是干嘛？dist乘以光线方向的世界坐标的二范数

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                use_depth_prior=False,
                depth_prior=False,
                depth_sample_percentile = 1.):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    # N_rays = ray_batch.shape[0]
    # # rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    # rays_o, rays_d, rays_depth = ray_batch[:,0:3], ray_batch[:,3:6], ray_batch[:,8] # [N_rays, 3] each
    # viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None # ccc : 这是归一化的光线射线矢量
    # bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    # near, far = bounds[...,0], bounds[...,1] # [-1,1]

    N_rays = ray_batch.shape[0]
    # rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    rays_o, rays_d, rays_depth = ray_batch[:,0:3], ray_batch[:,3:6], ray_batch[:,8:11] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None # ccc : 这是归一化的光线射线矢量
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    prior_N_samples = math.ceil(N_samples * depth_sample_percentile)
    if depth_prior and use_depth_prior and prior_N_samples > 1:
        N_samples =  math.ceil(N_samples * (1- depth_sample_percentile))

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])     # ccc : z_vals就是处于[near, far]区间内的N_samples个采样点

    if perturb > 0.:
        # get intervals between samples # ccc: 把z_vals里的N_samples个采样点加上随机扰动
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    # ccc exp：根据深度先验额外采样prior_N_samples个点
    # prior_N_samples = math.ceil(N_samples * depth_sample_percentile)
    if depth_prior and use_depth_prior and prior_N_samples > 1:
    # if depth_prior and N_samples > 0:
        prior_sample_near = rays_depth[..., 0] # [N_rays]
        prior_sample_far = rays_depth[..., 1] # [N_rays]

        prior_sample_near = prior_sample_near[:, None].expand(N_rays, prior_N_samples)
        prior_sample_far = prior_sample_far[:, None].expand(N_rays, prior_N_samples)

        prior_sample_t_vals = torch.linspace(0., 1., steps=prior_N_samples) 
        prior_sample_t_vals = prior_sample_t_vals.expand(N_rays, prior_N_samples)

        prior_sample_near = prior_sample_near * (torch.ones_like(prior_sample_t_vals) - prior_sample_t_vals)
        prior_sample_far = prior_sample_far *  prior_sample_t_vals

        p_sample_t_vals = torch.stack((prior_sample_near, prior_sample_far), dim=-1) # [N_rays, prior_N_samples, 2]
        p_sample_t_vals = torch.sum(p_sample_t_vals, dim=-1) # [N_rays, prior_N_samples, 1]
        p_sample_t_vals = p_sample_t_vals.squeeze()  # [N_rays, prior_N_samples]

        if len(p_sample_t_vals.shape) == 1: # 处理 prior_N_samples = 1的情况，被squeeze成了1维。为了拼接还得升维
            p_sample_t_vals = p_sample_t_vals[:, None]


        # new_t_vals = torch.linspace(0., 1., steps=N_samples) 
        # if not lindisp:
        #     new_zvals = extra_near * (1.-new_t_vals) + extra_far * (new_t_vals)
        # else:
        #     new_zvals = 1./(1./extra_near * (1.-new_t_vals) + 1./extra_far * (new_t_vals))

        # # new_zvals = new_zvals.expand([N_rays, N_samples]) # [N_rays, N_samples]
        # get intervals between samples # ccc: 把z_vals里的N_samples个采样点加上随机扰动
        mids = .5 * (p_sample_t_vals[...,1:] + p_sample_t_vals[...,:-1])  # [N_rays, prior_N_samples]
        upper = torch.cat([mids, p_sample_t_vals[...,-1:]], -1) # [N_rays, prior_N_samples]
        lower = torch.cat([p_sample_t_vals[...,:1], mids], -1) # [N_rays, prior_N_samples]
        # stratified samples in those intervals
        t_rand = torch.rand(p_sample_t_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(p_sample_t_vals.shape))
            t_rand = torch.Tensor(t_rand)

        p_sample_t_vals = lower + (upper - lower) * t_rand  # [N_rays, prior_N_samples]
        z_vals, _ = torch.sort(torch.cat([z_vals, p_sample_t_vals], -1), -1)
    
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)


    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map}
    if retraw:
        ret['raw'] = raw
    # ccc exp : no more fine network
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    # cococat experiment
    parser.add_argument("--depth_prior", action='store_true', 
                        help='[COCOCAT EXPERIMENT]extra fine sample using depth prior')
    parser.add_argument("--prior_dir", type=str, default='None', 
                        help='[COCOCAT EXPERIMENT]depth prior data directory')
    parser.add_argument("--prior_percentile", type=float,
                        default=.05, help='[COCOCAT EXPERIMENT] percentile of prior sampling along ray') 
    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test, depth_scale = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        # 接下来的步骤只考虑rgb，所以需要把α通道移除，将rgb通道修改为经过透明度计算的结果
        # 线性插值公式：αA + (1-α)B。在这个场景下，A即是读取到的颜色值r/g/b，B即白色背景的颜色值1（读取时对255做了归一化）
        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    # cococat: load depthmap
    if args.prior_dir == "None":
        print("No prior path set. exiting")
        return

    depth_maps, fused_depth_maps = depthOP.load_depth_map_ACMM(args.prior_dir, args.factor) # [N, H, W]
    depth_maps = depth_scale * depth_maps
    avg_depth = np.mean(depth_maps, axis=2)
    avg_depth = np.mean(avg_depth, axis=1)
    avg_depth = np.mean(avg_depth, axis=0)
    print("full depth map avg depth: " + str(avg_depth))

    fused_depth_available = True
    if fused_depth_maps is None:
        fused_depth_available = False
    
    if fused_depth_available:
        fused_depth_maps = depth_scale * fused_depth_maps
        avg_depth = np.mean(fused_depth_maps, axis=2)
        avg_depth = np.mean(avg_depth, axis=1)
        avg_depth = np.mean(avg_depth, axis=0)
        print("fused depth map avg depth: " + str(avg_depth))
    
    # cococat: project all depth map points to world coordinate
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    x_cam_coord = (i-K[0][2])/K[0][0]
    y_cam_coord = -(j-K[1][2])/K[1][1]
    x_cam_coord = np.broadcast_to(x_cam_coord[...,np.newaxis], (depth_maps.shape[0], H, W, 1)) # [N, H, W, 1]
    y_cam_coord = np.broadcast_to(y_cam_coord[...,np.newaxis], (depth_maps.shape[0], H, W, 1)) # [N, H, W, 1]
    x_cam_coord = x_cam_coord * depth_maps[...,np.newaxis]
    y_cam_coord = y_cam_coord * depth_maps[...,np.newaxis]
    depth_cam_coord = np.concatenate((x_cam_coord, y_cam_coord), -1) # [N, H, W, 2], depth points in image coordinate

    # depth_cam_coord = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1]], -1)  # [H, W, 2]
    # depth_cam_coord = np.broadcast_to(depth_cam_coord, (depth_maps.shape[0], H, W, 2)) # [N, H, W, 2]
    depth_cam_coord = np.concatenate((depth_cam_coord, -depth_maps[...,np.newaxis]), -1) # [N, H, W, 3], depth points in image coordinate
    # depth_world_coord = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rotate_pose = poses[..., :3] # [N, 3, 3] following function "get_rays_np" in run_nerf_helpers.py
    rotate_pose = np.expand_dims(rotate_pose,1).repeat(W,axis=1) # [N, W, 3, 3]
    rotate_pose = np.expand_dims(rotate_pose,1).repeat(H,axis=1) # [N, H, W, 3, 3]
    depth_world_coord = np.matmul(rotate_pose, depth_cam_coord[..., np.newaxis])
    depth_world_coord = depth_world_coord.squeeze()

    # same projection for fused depth
    if fused_depth_available:
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        x_cam_coord = (i-K[0][2])/K[0][0]
        y_cam_coord = -(j-K[1][2])/K[1][1]
        x_cam_coord = np.broadcast_to(x_cam_coord[...,np.newaxis], (fused_depth_maps.shape[0], H, W, 1)) # [N, H, W, 1]
        y_cam_coord = np.broadcast_to(y_cam_coord[...,np.newaxis], (fused_depth_maps.shape[0], H, W, 1)) # [N, H, W, 1]
        x_cam_coord = x_cam_coord * fused_depth_maps[...,np.newaxis]
        y_cam_coord = y_cam_coord * fused_depth_maps[...,np.newaxis]
        fused_depth_cam_coord = np.concatenate((x_cam_coord, y_cam_coord), -1) # [N, H, W, 2], depth points in image coordinate

        fused_depth_cam_coord = np.concatenate((fused_depth_cam_coord, -fused_depth_maps[...,np.newaxis]), -1) # [N, H, W, 3], depth points in image coordinate
        rotate_pose = poses[..., :3] # [N, 3, 3] following function "get_rays_np" in run_nerf_helpers.py
        rotate_pose = np.expand_dims(rotate_pose,1).repeat(W,axis=1) # [N, W, 3, 3]
        rotate_pose = np.expand_dims(rotate_pose,1).repeat(H,axis=1) # [N, H, W, 3, 3]
        fused_depth_world_coord = np.matmul(rotate_pose, fused_depth_cam_coord[..., np.newaxis])
        fused_depth_world_coord = fused_depth_world_coord.squeeze()

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand        # ccc: 这个就是batchsize，每一次放入网络的光线个数
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]     # ccc: 根据get_rays_np函数，pose就是c2w矩阵
        print('done, concats')

        # ccc exp: 根据深度先验世界坐标，确定深度采样的范围
        rays_o, rays_d = rays[:, 0], rays[:, 1] # [N, H, W, 3]
        vec_d2o = rays_d - rays_o
        vec_p2o = depth_world_coord - rays_o # 'p' for prior
        dist_d2o = np.linalg.norm(vec_d2o, ord=2, axis=3, keepdims=True) # [N, H, W, 1]
        dist_p2o = np.linalg.norm(vec_p2o, ord=2, axis=3, keepdims=True) # [N, H, W, 1]
        inner_product = vec_d2o * vec_p2o
        inner_product = np.sum(inner_product, -1)
        inner_product = np.expand_dims(inner_product, 3)  # [N, H, W, 1]
        prior_confidence = inner_product / (np.linalg.norm(vec_d2o, ord=2, axis=3, keepdims=True) * np.linalg.norm(vec_p2o, ord=2, axis=3, keepdims=True))

        dist_near_far_plane = args.prior_percentile * (far - near) * np.ones_like(dist_d2o)
        prior_sample_near = dist_p2o / dist_d2o - dist_near_far_plane
        prior_sample_far = dist_p2o / dist_d2o + dist_near_far_plane
        prior_sample_plane = np.concatenate([prior_sample_near, prior_sample_far], -1)  

        bad_count = 0
        floor_count = 0
        for image_index in range(dist_d2o.shape[0]):
            for row_index in range(dist_d2o.shape[1]):
                for col_index in range(dist_d2o.shape[2]):
                    if prior_confidence[image_index, row_index, col_index, 0] < 0.9659 or prior_sample_plane[image_index, row_index, col_index, 0] > far or prior_sample_plane[image_index, row_index, col_index, 1] < near :
                        prior_sample_plane[image_index, row_index, col_index, 0] = near
                        prior_sample_plane[image_index, row_index, col_index, 1] = far
                        bad_count += 1
                    if prior_sample_plane[image_index, row_index, col_index, 0] < near :
                        prior_sample_plane[image_index, row_index, col_index, 0] = near
                        floor_count += 1
                    if prior_sample_plane[image_index, row_index, col_index, 1] > far:
                        prior_sample_plane[image_index, row_index, col_index, 1] = far
                        floor_count += 1


        all_pixels = dist_d2o.shape[0] * dist_d2o.shape[1] * dist_d2o.shape[2]
        print("all prior pixels: " + str(all_pixels))
        print("bad prior pixels: " + str(bad_count))
        print("good prior rate: " + str((float(all_pixels) - bad_count) / all_pixels * 100) + "%")
        print("floor prior count: " + str(floor_count))
        prior_sample_all_info = np.concatenate([prior_sample_plane, depth_maps[..., np.newaxis]], -1) # [N, H, W, 3]
        prior_sample_all_info = np.expand_dims(prior_sample_all_info, 1)  # [N, 1, H, W, 3]

        # # original verrsion of nerf rays batch
        # rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        # rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        # rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        # rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        # rays_rgb = rays_rgb.astype(np.float32)
        # print('shuffle rays')
        # np.random.shuffle(rays_rgb)

        # ccc exp: 将融合后的深度也进行投影
        if fused_depth_available:
            # rays_o, rays_d = rays[:, 0], rays[:, 1] # [N, H, W, 3]
            # vec_d2o = rays_d - rays_o
            # dist_d2o = np.linalg.norm(vec_d2o, ord=2, axis=3, keepdims=True) # [N, H, W, 1]
            vec_fp2o = fused_depth_world_coord - rays_o # 'fp' for fused prior
            dist_fp2o = np.linalg.norm(vec_fp2o, ord=2, axis=3, keepdims=True) # [N, H, W, 1]
            fp_inner_product = vec_d2o * vec_fp2o
            fp_inner_product = np.sum(fp_inner_product, -1)
            fp_inner_product = np.expand_dims(fp_inner_product, 3)  # [N, H, W, 1]
            fused_prior_confidence = fp_inner_product / (np.linalg.norm(vec_d2o, ord=2, axis=3, keepdims=True) * np.linalg.norm(vec_fp2o, ord=2, axis=3, keepdims=True))

            # dist_near_far_plane = args.prior_percentile * (far - near) * np.ones_like(dist_d2o)
            fused_prior_sample_near = dist_fp2o / dist_d2o - dist_near_far_plane
            fused_prior_sample_far = dist_fp2o / dist_d2o + dist_near_far_plane
            fused_prior_sample_plane = np.concatenate([fused_prior_sample_near, fused_prior_sample_far], -1)  

            bad_count = 0
            floor_count = 0
            for image_index in range(dist_d2o.shape[0]):
                for row_index in range(dist_d2o.shape[1]):
                    for col_index in range(dist_d2o.shape[2]):
                        if fused_prior_confidence[image_index, row_index, col_index, 0] < 0.9659 or fused_prior_sample_plane[image_index, row_index, col_index, 0] > far or fused_prior_sample_plane[image_index, row_index, col_index, 1] < near :
                            fused_prior_sample_plane[image_index, row_index, col_index, 0] = near
                            fused_prior_sample_plane[image_index, row_index, col_index, 1] = far
                            bad_count += 1
                        if fused_prior_sample_plane[image_index, row_index, col_index, 0] < near :
                            fused_prior_sample_plane[image_index, row_index, col_index, 0] = near
                            floor_count += 1
                        if fused_prior_sample_plane[image_index, row_index, col_index, 1] > far:
                            fused_prior_sample_plane[image_index, row_index, col_index, 1] = far
                            floor_count += 1
            
            print("all prior pixels: " + str(all_pixels))
            print("bad fused prior pixels: " + str(bad_count))
            print("good fused prior rate: " + str((float(all_pixels) - bad_count) / all_pixels * 100) + "%")
            print("floor fused prior count: " + str(floor_count))
            fused_prior_sample_all_info = np.concatenate([fused_prior_sample_plane, fused_depth_maps[..., np.newaxis]], -1) # [N, H, W, 3]
            fused_prior_sample_all_info = np.expand_dims(fused_prior_sample_all_info, 1)  # [N, 1, H, W, 3]
            prior_sample_all_info = np.concatenate([prior_sample_all_info, fused_prior_sample_all_info], 1) # [N, depth+fused_depth(2), H, W, 3]

        rays_depth = np.concatenate([rays, prior_sample_all_info], 1) # [N, ro+rd+depth, H, W, 3]
        rays_depth_rgb = np.concatenate([rays_depth, images[:,None]], 1) # [N, ro+rd+depth+rgb, H, W, 3]
        rays_depth_rgb = np.transpose(rays_depth_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+depth+rgb, 3]
        rays_depth_rgb = np.stack([rays_depth_rgb[i] for i in i_train], 0) # train images only
        # rays_depth_rgb = np.reshape(rays_depth_rgb, [-1,4,3]) # [(N-1)*H*W, ro+rd+depth+rgb, 3]
        rays_depth_rgb = np.reshape(rays_depth_rgb, [-1,5,3]) # [(N-1)*H*W, ro+rd+depth+f_depth+rgb, 3]

        rays_depth_rgb = rays_depth_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_depth_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        # rays_rgb = torch.Tensor(rays_rgb).to(device)
        rays_depth_rgb = torch.Tensor(rays_depth_rgb).to(device)


    # N_iters = 50000 + 1 # N_iters = 200000 + 1
    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    timer_started = True
    time0 = time.time()
    need_prior_assist=True #TODO:
    depth_sample_percentile=1.
    depth_sample_decrease=0.005
    coarse_depth_prior_assist=True # 最初由粗糙深度指导采样，随后如果存在融合深度，则切换成融合深度
    fused_depth_prior_assist=False
    # if fused_depth_available:
    #     fused_depth_prior_assist=True

    for i in trange(start, N_iters):
        if not timer_started:
            timer_started = True
            time0 = time.time()

        depth_map = None

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_depth_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?] ccc: [B, 2+1+1(depth), 3]
            batch = torch.transpose(batch, 0, 1)
            # batch_rays, target_s = batch[:3], batch[-1]
            rays_o_d, coarse_depth, fused_depth, target_s = batch[:2], batch[2], batch[-2], batch[-1]
            if fused_depth_prior_assist: # 
                batch_rays = torch.cat([rays_o_d, fused_depth[None, ...]], dim=0)
            else:
                batch_rays = torch.cat([rays_o_d, coarse_depth[None, ...]], dim=0)

            i_batch += N_rand
            if i_batch >= rays_depth_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_depth_rgb.shape[0])
                rays_depth_rgb = rays_depth_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]
            # ccc exp: choose the corresponding depth map and send to device
            if args.depth_prior:
                if len(depth_maps) > img_i:
                    depth_map = depth_maps[img_i]
                else:
                    print("[WARNING] Cannot find corresponding depth map. Depth prior may not work.")
            if depth_map is not None:
                dpeth_map = torch.Tensor(depth_map).to(device)  # TODO:这里似乎已经不需要传入深度图了，深度信息已经被包裹于rays

            if N_rand is not None:  # ccc : 下面是世界坐标下的光心坐标和光线射线坐标
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                # ccc : 从当前图的像素坐标系下所有点中随机选择N_rand个，将他们的像素坐标存入select_coords，
                # 并将世界坐标系下的光心坐标rays_o和光线射线坐标rays_d都筛得只剩下这些随机出来的点
                # 组装成光线向量batch_rays，和这些随机点的真实rgb色值张量target_s
                # TODO: 这边的batch还没加入深度信息
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2) 
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                # depth_map=None,   # TODO:这里似乎已经不需要传入深度图了，深度信息已经被包裹于rays
                                                use_depth_prior=need_prior_assist,
                                                depth_sample_percentile=depth_sample_percentile,
                                                **render_kwargs_train)
        
        if i%args.i_print==0 and i >= 25000 and args.depth_prior and need_prior_assist:
            batch_depths = batch_rays[2, :, 2].clone().detach().cpu().numpy()
            nerf_depths = extras['depth_map'].clone().detach().cpu().numpy() # [B]
            depth_diff = 0.0
            valid_prior_count = 0
            for depth_iter in range(batch_depths.shape[0]):
                if(batch_depths[depth_iter] != 0):
                    depth_diff = depth_diff + abs(batch_depths[depth_iter] - nerf_depths[depth_iter])
                    valid_prior_count = valid_prior_count + 1

            depth_diff = depth_diff / ((far - near) * valid_prior_count)
            # depth_diff = np.absolute(np.sum(depth_diff))
            if depth_diff < 0.0675:
                depth_sample_decrease = 0.08
            elif depth_diff < 0.125:
                depth_sample_decrease = 0.04
            elif depth_diff < 0.25:
                depth_sample_decrease = 0.02
            elif depth_diff < 0.50:
                depth_sample_decrease = 0.01
            else:
                depth_sample_decrease = 0.005
            depth_sample_percentile = depth_sample_percentile - depth_sample_decrease # 消融实验，关闭退化
            print("depth_sample_percentile set to " + str(depth_sample_percentile))
            
            if depth_sample_percentile <= 0.:
                time_file = os.path.join(basedir, expname, 'time.txt')
                if fused_depth_prior_assist is True:
                    need_prior_assist = False
                    with open(time_file, 'a+') as f:
                        f.write('fused prior disabled at iter ' + str(i) +'\n')
                        f.write('end of prior guide\n')
                else :
                    if fused_depth_available is True:
                        # fused depth availble. switching to fused depth for sampling guide
                        coarse_depth_prior_assist = False
                        fused_depth_prior_assist = True
                        depth_sample_percentile=1.
                        depth_sample_decrease=0.005
                        with open(time_file, 'a+') as f:
                            f.write('fused prior disabled at iter ' + str(i) +'\n')
                            f.write('switching to fused depth\n')
                    else:
                        # no fused depth available. time to end prior guide
                        need_prior_assist = False
                        with open(time_file, 'a+') as f:
                            f.write('coarse prior disabled at iter ' + str(i) +'\n')
                            f.write('end of prior guide\n')

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if render_kwargs_train['network_fine'] is not None:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    # 'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            timer_started = False
            time_elapsed = time.time()-time0
            str_time_info = '{:d} iters done, {:.0f}m {:.0f}s passed. PSNR: {:.5f}'.format(i, time_elapsed // 60, time_elapsed % 60, psnr.item())
            time_file = os.path.join(basedir, expname, 'time.txt')
            with open(time_file, 'a+') as f:
                f.write(str_time_info +'\n')

            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
