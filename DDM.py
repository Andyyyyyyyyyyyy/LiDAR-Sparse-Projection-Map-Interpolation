import os
import os.path as osp
import sys
from collections import defaultdict
import glob
import collections
import time
import numpy as np
import PIL.Image
import cv2
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

#range
min_range = 5
max_range = 80


img_dir ="./Images/"
lidar_dir = "./LiDAR/"
calib_file = "calibration.txt"   
ouput_dir ="./DenseDepthMap/"

if osp.exists(ouput_dir):
	print('Output directory already exists:', ouput_dir)
else:
	os.makedirs( ouput_dir );
	print('Creating DenseIMG:', ouput_dir)

#read files
img_files = glob.glob(osp.join(img_dir, '*.png'))
lidar_files = glob.glob(osp.join(lidar_dir, '*.bin'))
img_files.sort()
lidar_files.sort()

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

# read calibration.txt #another cali_matrix form
calib = read_calib_file(calib_file)
R_rect_00 = np.zeros((4, 4))
R_rect_00[:3,:3] = np.array(calib['R0_rect']).reshape(-1, 3)
R_rect_00[3,3] = 1.

P_rect_02 = np.zeros((4, 4))
P_rect_02[:3, :] = np.array(calib['P2']).reshape(-1, 4)
P_rect_02[3, 3] = 1.

vel2cam0 = np.zeros((4, 4))
vel2cam0[:3,:] = np.array(calib['Tr_velo_to_cam']).reshape(-1, 4)
vel2cam0[3,3] = 1.

# compute transform matrix 
T_lidar_camera = np.dot(R_rect_00, vel2cam0)
P_lidar_image = np.dot(P_rect_02, T_lidar_camera)

#load
for  i, lidar_file  in enumerate(lidar_files):

    print(i,'Generating dense depth image from:', osp.basename(lidar_file))
    start_time = time.time()
    
    with open(lidar_file, 'r') as f:
        pointcloud = np.fromfile(f, dtype=np.float32).reshape(-1, 4)

    image = cv2.imread(img_files[i]) 
    h, w, _= image.shape
   
    pointcloud = pointcloud[np.logical_and((pointcloud[:, 1] < pointcloud[:, 0] - 0.27), (-pointcloud[:, 1] < pointcloud[:, 0] - 0.27))]   
    
    pixel = np.dot(P_lidar_image, pointcloud.T).T

    pixel[:,0] = np.divide(pixel[:,0], pixel[:, 2])
    pixel[:,1] = np.divide(pixel[:,1], pixel[:, 2])
    pixel[:,:2] = np.round(pixel[:,:2]).astype(np.uint16)
    
    pixel = pixel[(pixel[:,0] >= 0) & (pixel[:,0] < w) & (pixel[:,1] >= 0) & (pixel[:,1] < h)]

    # post process: unique pixel with closest distances 
    _, ia = np.unique(pixel[:, :2], axis=0, return_index=True)              # unique value (keeps closest points)  
    pixel = pixel[ia,:]                                                     # [r c depth reflectance], 

    #sparse maps 
    dms = np.zeros((h, w))                        # reflectance map
    dms[pixel[:, 1].astype(np.uint16), pixel[:, 0].astype(np.uint16)] = pixel[:, 3]   


    # interpolation reflectance-map 
    row, col = np.meshgrid(np.linspace(0, w-1, num=w), np.linspace(0, h-1, num=h))  
    dmd = griddata(pixel[:, :2], pixel[:, 2], (row, col), method='nearest')    #try 'nearest','linear'

    dmd = (dmd - min_range) / (max_range - min_range)
    dmd = np.where(dmd < 0, 0, dmd)
    dmd = np.where(dmd > 1, 1, dmd)

    # mask the result to the ROI
    mask = np.zeros((h, w))                                # initialize mask 
    index = (dms != 0).argmax(axis=0)
    for i in range(w):
        mask[index[i]:, i] = 1

    kernel = np.ones((2, 10), np.uint8)
    mask   = cv2.dilate(mask,kernel,iterations = 1)                         # by Dilation
    
    ## mask the reflectance map
    dm  = mask*(255*dmd).astype(np.uint8)
    
    #Save
    plt.imsave(ouput_dir+osp.splitext(osp.basename(lidar_file))[0]+".jpg", dm, cmap="jet")
#    cv2.imwrite(ouput_dir+osp.splitext(osp.basename(lidar_file))[0]+".jpg", dm)
    
    print("Saving as :",ouput_dir+osp.splitext(osp.basename(lidar_file))[0]+".jpg")
    end_time = time.time()
    print("Time Consuming :", end_time - start_time)
    print()