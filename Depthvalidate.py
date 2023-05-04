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

from load_llff import load_llff_data

from run_nerf_helpers import *
import cv2
import mvs.depth_map_operate as depthOP
import math

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def visualize():
    # min_depth = colmap_depth_map.min()
    # max_depth = colmap_depth_map.max()
    # ratio = 255 / (max_depth - min_depth)
    
    # colmap_depth_map = colmap_depth_map - min_depth
    # colmap_depth_map = colmap_depth_map * ratio
    # colmap_depth_map = colmap_depth_map.astype(np.uint8)

    # falseColorsMap = cv2.applyColorMap(colmap_depth_map, cv2.COLORMAP_JET)
    # cv2.imwrite("1.png", falseColorsMap)
    print("weee")


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

if __name__=='__main__':
    
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type != 'llff':
        print("No GOD!")
        exit()

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

    # depth_maps, fused_depth_maps = depthOP.load_depth_map_ACMM(args.prior_dir, args.factor) # [N, H, W]
    # depth_maps = depth_scale * depth_maps

    # targetRows = depth_maps.shape[1]
    # targetCols = depth_maps.shape[2]

    # str_colDepthMap="/mnt/d/xjx/Experiment/ColmapTest/fern_NEW/dense/stereo/depth_maps/IMG_4026.JPG.geometric.bin"
    # str_colDepthMap="/mnt/d/xjx/Experiment/ColmapTest/trex/dense/stereo/depth_maps/"
    str_colDepthMap="/mnt/d/xjx/biyelunwen/ACMM&ACMP/trex/ACMP/"


    # colmap_depth_maps = depthOP.load_depth_map_colmap(str_colDepthMap, targetCols, targetRows, args.factor, "g")
    colmap_depth_maps, _ = depthOP.load_depth_map_ACMM(str_colDepthMap, args.factor)
    colmap_depth_maps = depth_scale * colmap_depth_maps
    # avg_depth = np.mean(colmap_depth_maps, axis=2)
    # avg_depth = np.mean(avg_depth, axis=1)
    # avg_depth = np.mean(avg_depth, axis=0)
    # print("full depth map avg depth: " + str(avg_depth))


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

    # cococat: project all depth map points to world coordinate
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    x_cam_coord = (i-K[0][2])/K[0][0]
    y_cam_coord = -(j-K[1][2])/K[1][1]
    x_cam_coord = np.broadcast_to(x_cam_coord[...,np.newaxis], (colmap_depth_maps.shape[0], H, W, 1)) # [N, H, W, 1]
    y_cam_coord = np.broadcast_to(y_cam_coord[...,np.newaxis], (colmap_depth_maps.shape[0], H, W, 1)) # [N, H, W, 1]
    x_cam_coord = x_cam_coord * colmap_depth_maps[...,np.newaxis]
    y_cam_coord = y_cam_coord * colmap_depth_maps[...,np.newaxis]
    depth_cam_coord = np.concatenate((x_cam_coord, y_cam_coord), -1) # [N, H, W, 2], depth points in image coordinate

    # depth_cam_coord = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1]], -1)  # [H, W, 2]
    # depth_cam_coord = np.broadcast_to(depth_cam_coord, (colmap_depth_maps.shape[0], H, W, 2)) # [N, H, W, 2]
    depth_cam_coord = np.concatenate((depth_cam_coord, -colmap_depth_maps[...,np.newaxis]), -1) # [N, H, W, 3], depth points in image coordinate
    # depth_world_coord = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rotate_pose = poses[..., :3] # [N, 3, 3] following function "get_rays_np" in run_nerf_helpers.py
    rotate_pose = np.expand_dims(rotate_pose,1).repeat(W,axis=1) # [N, W, 3, 3]
    rotate_pose = np.expand_dims(rotate_pose,1).repeat(H,axis=1) # [N, H, W, 3, 3]
    depth_world_coord = np.matmul(rotate_pose, depth_cam_coord[..., np.newaxis])
    depth_world_coord = depth_world_coord.squeeze()


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
        # prior_sample_all_info = np.concatenate([prior_sample_plane, depth_maps[..., np.newaxis]], -1) # [N, H, W, 3]
        # prior_sample_all_info = np.expand_dims(prior_sample_all_info, 1)  # [N, 1, H, W, 3]

