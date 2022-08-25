from sn7_baseline_prep_funcs import map_wrapper, make_geojsons_and_masks
from solaris.utils.core import _check_gdf_load
from solaris.raster.image import create_multiband_geotiff
import solaris as sol
import multiprocessing
import pandas as pd
import numpy as np
import skimage
import gdal
import sys
import os
import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# # import from data_prep_funcs
# module_path = os.path.abspath(os.path.join('../src/'))
# if module_path not in sys.path:
#     sys.path.append(module_path)


# Dataset locatio
root_dir = '/home/liuxiangyu/dataset/spacenet7/'

# # Create Training Masks
# # Multi-thread to increase speed
# # We'll only make a 1-channel mask for now, but Solaris supports a multi-channel mask as well, see
# #     https://github.com/CosmiQ/solaris/blob/master/docs/tutorials/notebooks/api_masks_tutorial.ipynb

# aois = sorted([f for f in os.listdir(os.path.join(root_dir, 'train'))
#                if os.path.isdir(os.path.join(root_dir, 'train', f))])
# n_threads = 10
# params = []
# make_fbc = False

# input_args = []
# for i, aoi in enumerate(aois):
#     print(i, "aoi:", aoi)
#     im_dir = os.path.join(root_dir, 'train', aoi, 'images_masked/') # (training + testing) - Unusable portions of the image (usually due to cloud cover) have been masked out, in EPSG:3857 projection..
#     json_dir = os.path.join(root_dir, 'train', aoi, 'labels_match/') # This folder contains building footprints reprojected into the coordinate reference system (CRS) of the imagery (EPSG:3857 projection).  Each building footprint is assigned a unique identifier (i.e. address) that remains consistent throughout the data cube.  
#     out_dir_mask = os.path.join(root_dir, 'train', aoi, 'masks/')
#     out_dir_mask_fbc = os.path.join(root_dir, 'train', aoi, 'masks_fbc/')
#     os.makedirs(out_dir_mask, exist_ok=True)
#     if make_fbc:
#         os.makedirs(out_dir_mask_fbc, exist_ok=True)

#     json_files = sorted([f
#                          for f in os.listdir(os.path.join(json_dir))
#                          if f.endswith('Buildings.geojson') and os.path.exists(os.path.join(json_dir, f))])
#     for j, f in enumerate(json_files):
#         # print(i, j, f)
#         name_root = f.split('.')[0]
#         json_path = os.path.join(json_dir, f)
#         image_path = os.path.join(
#             im_dir, name_root + '.tif').replace('labels', 'images').replace('_Buildings', '')
#         output_path_mask = os.path.join(out_dir_mask, name_root + '.tif')
#         if make_fbc:
#             output_path_mask_fbc = os.path.join(
#                 out_dir_mask_fbc, name_root + '.tif')
#         else:
#             output_path_mask_fbc = None

#         if (os.path.exists(output_path_mask)):
#             continue
#         else:
#             input_args.append([make_geojsons_and_masks,
#                                name_root, image_path, json_path,
#                                output_path_mask, output_path_mask_fbc])

# # execute
# print("len input_args", len(input_args))
# print("Execute...\n")
# with multiprocessing.Pool(n_threads) as pool:
#     pool.map(map_wrapper, input_args)


# Make dataframe csvs for train/test
trainpath = '/home/liuxiangyu/dataset/spacenet7/csvs/sn7_baseline_train_df.csv'
valpath = '/home/liuxiangyu/dataset/spacenet7/csvs/sn7_baseline_val_df.csv'
testpath = '/home/liuxiangyu/dataset/spacenet7/csvs/sn7_baseline_test_df.csv'

d = os.path.join(root_dir, 'train')
im_list, mask_list = [], []
subdirs = sorted([f for f in os.listdir(
    d) if os.path.isdir(os.path.join(d, f))])
for subdir in subdirs:
        im_files = [os.path.join(d, subdir, 'images_masked', f)
                    for f in sorted(os.listdir(os.path.join(d, subdir, 'images_masked')))
                    if f.endswith('.tif') and os.path.exists(os.path.join(d, subdir, 'masks', f.split('.')[0] + '_Buildings.tif'))]
        mask_files = [os.path.join(d, subdir, 'masks', f.split('.')[0] + '_Buildings.tif')
                        for f in sorted(os.listdir(os.path.join(d, subdir, 'images_masked')))
                        if f.endswith('.tif') and os.path.exists(os.path.join(d, subdir, 'masks', f.split('.')[0] + '_Buildings.tif'))]
        im_list.extend(im_files)
        mask_list.extend(mask_files)

train_df = pd.DataFrame({'image': im_list, 'label': mask_list})
# 训练集中取一部分做验证集，60个AOI取10个AOI，按照AOI来取方便计算SCOT
AOIs = [z.split('mosaic_')[-1].split('.tif')[0]
        for z in train_df['image'].values]
AOIs = list(set(AOIs))
val_AOIs = AOIs[:10]
test_im_list = [i for i in list(train_df['image']) if i.split('mosaic_')[-1].split('.tif')[0] in val_AOIs]
test_df = pd.DataFrame({'image': test_im_list})
# test 不带标签
test_df.to_csv(testpath, index=False)
val_df = train_df[train_df['image'].isin(test_im_list)]
# val 带标签
val_df.to_csv(valpath, index=False)
# remove the validation samples from the training df
train_df = train_df[~train_df['image'].isin(test_im_list)]
train_df.to_csv(trainpath, index=False)


# 验证集label的geojson转csv

import os
from sn7_baseline_postproc_funcs import sn7_convert_geojsons_to_csv

truth_csv = '/home/liuxiangyu/CosmiQ_SN7_Baseline/sn7_baseline_labels.csv'

aoi_dirs = sorted([os.path.join('/home/liuxiangyu/dataset/spacenet7/train/', aoi, 'labels_match_pix') \
                   for aoi in val_AOIs])

# Execute
net_df = sn7_convert_geojsons_to_csv(aoi_dirs, truth_csv, 'ground')