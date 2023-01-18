import numpy as np
import os
import cv2
import struct

def read_array_colmap(path):
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
    return np.transpose(array, (1, 0, 2)).squeeze(), width, height

def read_depth_colmap(depth_map_path, half_res = False):
    # Read depth and normal maps corresponding to the same image.
    if not os.path.exists(depth_map_path):
        raise FileNotFoundError("File not found: {}".format(depth_map_path))

    depth_map, width, height = read_array_colmap(depth_map_path)

    if half_res:
        depth_map = cv2.resize(depth_map, (width//2, height//2), interpolation=cv2.INTER_AREA)
    return depth_map
    # min_depth, max_depth = np.percentile(
    #     depth_map, [args.min_depth_percentile, args.max_depth_percentile])
    # depth_map[depth_map < min_depth] = min_depth
    # depth_map[depth_map > max_depth] = max_depth

    # import pylab as plt

    # # Visualize the depth map.
    # plt.figure()
    # plt.imshow(depth_map)
    # plt.title("depth map")
    # plt.show()

def load_depth_map_colmap(depth_map_folder, half_res = False):
    depth_maps = []

    for filename in os.listdir(depth_map_folder):
        if filename.find("geometric") != -1 :
            depth_map = read_depth_colmap(depth_map_folder + "/" + filename, half_res)
            depth_maps.append(depth_map)
            # if depth_map.shape != [800, 800] :
            #     # TODO: 若要使用colmap的深度图，则存在问题：其结果可能会被裁减，长宽均不是原始图像尺寸，此时只能居中填充 
            #     # 暂时不管了，用ACMM的深度图，是原尺寸 by cococat
            #     print(depth_map.shape)
            #     print("NOOOOOOOOOOOOOOO!")

    return depth_maps


def read_single_depth_ACMM(depth_map_path, scale_factor=1) :
    with open(depth_map_path, "rb") as file:
        type = struct.unpack('i', file.read(4))[0]
        height = struct.unpack('i', file.read(4))[0]
        width = struct.unpack('i', file.read(4))[0]
        nb = struct.unpack('i', file.read(4))[0]

        if type != 1:
            print("[ERROR]reading depth map " + depth_map_path + " failed! aborting...")
            raise RuntimeError("depth map file format error!")
        
        depth_map = np.fromfile(file, np.float32)
        # depth_map = depth_map.reshape((height, width, nb), order="C")
        depth_map = np.reshape(depth_map, (height, width, nb))

        # # visualization 
        # min, max, _, _ = cv2.minMaxLoc(depth_map)
        # converted = cv2.convertScaleAbs(depth_map, _, 255 / max)
        # converted = cv2.resize(converted, (1920, 1080))
        # cv2.imshow("weee", converted)
        # cv2.waitKey(0)
        # cv2.imwrite("/home/rec/Experiment/readFromACMMtest.tiff", converted);

        depth_map = depth_map.squeeze()
        if scale_factor is not 1:
            new_size = (int(width/scale_factor), int(height/scale_factor))
            depth_map = cv2.resize(depth_map, new_size)
        return depth_map

def load_depth_map_ACMM(ACMM_root_folder, scale_factor=1):
    depth_maps = []
    depth_paths = []

    for home, dirs, files in os.walk(ACMM_root_folder):
        for subdir in sorted(dirs):
            # depth map with geometry consistency has priority
            geom_depth_path = ACMM_root_folder + subdir + "/" + "depths_geom.dmb"
            if os.path.exists(geom_depth_path) :
                depth_paths.append(geom_depth_path)
                depth_maps.append(read_single_depth_ACMM(geom_depth_path, scale_factor))
                continue

            regular_depth_path = ACMM_root_folder + subdir + "/" + "depths.dmb"
            if os.path.exists(regular_depth_path) :
                depth_paths.append(regular_depth_path)
                depth_maps.append(read_single_depth_ACMM(regular_depth_path, scale_factor))

    return depth_maps