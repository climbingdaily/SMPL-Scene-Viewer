# -*- coding: utf-8 -*-
# @Author  : Zhang.Jingyi

"""
This file contains definitions of useful data structures and the paths 
for the datasets and data files necessary to run the code.
"""

import os
import sys
from os.path import join

ROOT_DATASET = '/hdd/dyd'
PEDX_ROOT = os.path.join(ROOT_DATASET, 'pedx')
LIDARCAP_ROOT = join(ROOT_DATASET, 'lidarcap')
DATASET_FILES = [
    {'pedx': join(PEDX_ROOT, 'extras_py3', 'pedx_test.npz'), 'lidarcap': join(
        LIDARCAP_ROOT, 'extras', 'lidarcap_test.npz'), 'lidarcap_seq': join(
        LIDARCAP_ROOT, 'extras', 'lidarcap_seq_test.hdf5'), 'lidarcap_seq_reduced': join(
        LIDARCAP_ROOT, 'extras', 'lidarcap_seq_reduced_test.hdf5')},
    {'pedx': join(PEDX_ROOT, 'extras_py3', 'pedx_train.npz'), 'lidarcap': join(
        LIDARCAP_ROOT, 'extras', 'lidarcap_train.npz'), 'lidarcap_seq': join(
        LIDARCAP_ROOT, 'extras', 'lidarcap_seq_train.hdf5'), 'lidarcap_seq_reduced': join(
        LIDARCAP_ROOT, 'extras', 'lidarcap_seq_reduced_train.hdf5')},
]

VISUAL_DATASET_FILES = {'lidarcap_seq': join(
    LIDARCAP_ROOT, 'extras', 'lidarcap_seq_whole.hdf5')}
SMPL_FILE = 'C:\\Users\\DAI\\Documents\\GitHub\\ASC_Lidar_human\\smpl\\basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
# seems all zeros.
JOINT_REGRESSOR_TRAIN_EXTRA = 'C:\\Users\\DAI\\Documents\\GitHub\\ASC_Lidar_human\\smpl\\J_regressor_extra.npy'

"""
Each dataset uses different sets of joints.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are:
0 - Right Ankle
1 - Right Knee
2 - Right Hip
3 - Left Hip
4 - Left Knee
5 - Left Ankle
6 - Right Wrist
7 - Right Elbow
8 - Right Shoulder
9 - Left Shoulder
10 - Left Elbow
11 - Left Wrist
12 - Neck (LSP definition)
13 - Top of Head (LSP definition)
14 - Pelvis (MPII definition)
15 - Thorax (MPII definition)
16 - Spine (Human3.6M definition)
17 - Jaw (Human3.6M definition)
18 - Head (Human3.6M definition)
19 - Nose
20 - Left Eye
21 - Right Eye
22 - Left Ear
23 - Right Ear
"""

JOINTS_IDX = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18,
              20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]


'''
some parameters are related to dataset
'''

IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
