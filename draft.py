import mvs.depth_map_operate as depthOP
import numpy as np
import cv2
import struct
import torch

if __name__=="__main__":
    # # depth_map = depthOP.read_depth("/home/rec/Experiment/ColmapTest/lego/dense/0/stereo/depth_maps/r_0.png.geometric.bin", True)
    # # max_depth, min_depth = depth_map.max(), depth_map.min()

    # # depth_map = (depth_map / (max_depth - min_depth) * 255).astype(np.uint8)
    # depth_maps = depthOP.load_depth_map("/home/rec/Experiment/ColmapTest/lego/dense/0/stereo/depth_maps/", False)
    # test_map = depth_maps[5]
    # max_dpeth = max(test_map), min(test_map)
    # test_map = (test_map / (max_depth - min_depth) * 255).astype(np.uint8)
    # cv2.imwrite("/home/rec/Experiment/Test/array_geo.png", depth_maps[5])

    # test for reading acmm

    str_acmm_depth = "/home/rec/Experiment/ACMM/lego_ACMM/ACMM/2333_00000000/depths_geom.dmb"
    # depth_maps = depthOP.load_depth_map_ACMM("/home/rec/Experiment/ACMM/Fern/ACMM/")
    depthOP.read_single_depth_ACMM("/home/rec/Experiment/ACMM/Fern/ACMM/2333_00000000/depths_geom.dmb")

    print("weee")

    near = torch.Tensor()