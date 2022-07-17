# -*- coding: utf-8 -*-
# @Author  : jingyi
'''
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
'''
from .smpl import SMPL

import numpy as np
import argparse
from .generate_ply import save_ply
import math
import os
import pandas as pd
import torch
import sys
from pathlib import Path

'''
pose --> rotation of 24 skelentons
beta --> shape of human

pose can be:
    1. (B, 24, 3, 3)
    or
    2. (B, 72)
beta should be:
    (B, 10)
'''

'''
SMPL
'Root', 'Left_Hip', 'Right_Hip', 'Waist', 'Left_Knee', 'Right_Knee',
'Upper_Waist', 'Left_Ankle', 'Right_Ankle', 'Chest',
'Left_Toe', 'Right_Toe', 'Base_Neck', 'Left_Shoulder',
'Right_Shoulder', 'Upper_Neck', 'Left_Arm', 'Right_Arm',
'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist',
'Left_Finger', 'Right_Finger'
'''


def get_x_rot_mat(theta):
    res = np.eye(3)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    res[1, 1] = cos_theta
    res[1, 2] = -sin_theta
    res[2, 1] = sin_theta
    res[2, 2] = cos_theta
    return res


def get_y_rot_mat(theta):
    res = np.eye(3)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    res[0, 0] = cos_theta
    res[0, 2] = sin_theta
    res[2, 0] = -sin_theta
    res[2, 2] = cos_theta
    return res


def get_z_rot_mat(theta):
    res = np.eye(3)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    res[0, 0] = cos_theta
    res[0, 1] = -sin_theta
    res[1, 0] = sin_theta
    res[1, 1] = cos_theta
    return res


def rotmat_to_axis_angle(rotmat, return_angle=False):
    angle = math.acos((rotmat[0, 0] + rotmat[1, 1] + rotmat[2, 2] - 1) / 2)
    vec = [rotmat[2, 1] - rotmat[1, 2], rotmat[0, 2] -
           rotmat[2, 0], rotmat[1, 0] - rotmat[0, 1]]
    norm = np.linalg.norm(vec)
    if abs(norm) < 1e-8:
        norm = 1.0
    for i in range(3):
        vec[i] /= norm
    if return_angle:
        return np.array([vec, angle])
    for i in range(3):
        vec[i] *= angle
    return np.array(vec)


def get_pose_from_bvh(rotation_df, idx=0, converter_version=True):
    smpl_to_imu = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', 'LeftLeg',
                   'RightLeg', 'Spine1', 'LeftFoot', 'RightFoot', 'Spine2',
                   'LeftFootEnd', 'RightFootEnd', 'Neck', 'LeftShoulder',
                   'RightShoulder', 'Head', 'LeftArm', 'RightArm',
                   'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand',
                   'LeftHandThumb2', 'RightHandThumb2']
    pose = []
    import pandas as pd
    for each in smpl_to_imu:
        if converter_version:
            xrot = math.radians(rotation_df.at[idx, each + '.X'])
            yrot = math.radians(rotation_df.at[idx, each + '.Y'])
            zrot = math.radians(rotation_df.at[idx, each + '.Z'])
        else:
            xrot = 0
            yrot = 0
            zrot = 0
            if each + '.x' in rotation_df.columns:
                xrot = math.radians(rotation_df.at[idx, each + '.x'])
                yrot = math.radians(rotation_df.at[idx, each + '.y'])
                zrot = math.radians(rotation_df.at[idx, each + '.z'])
        if each == 'LeftShoulder':
            zrot -= 0.3 
        elif each == 'RightShoulder':
            zrot += 0.3
        elif each == 'LeftArm':
            zrot += 0.3 
        elif each == 'RightArm':
            zrot -= 0.3
        rotmat = np.eye(3)
        rotmat = np.dot(rotmat, get_y_rot_mat(yrot))
        rotmat = np.dot(rotmat, get_x_rot_mat(xrot))
        rotmat = np.dot(rotmat, get_z_rot_mat(zrot))
        pose.append(rotmat)
    for i in range(len(pose)):
        pose[i] = rotmat_to_axis_angle(pose[i])
    pose = np.stack(pose).flatten()
    return pose  # return rotation matrix


def main():
    rotation_df = pd.read_csv(args.bvh_name + '_rotations.csv')
    worldpos_df = pd.read_csv(args.bvh_name + '_worldpos.csv')
    with open('pose.txt', 'w') as f:
        for i in range(1059, 1658, 3):
            s = str((i - 1056) // 3)
            s += ' ' + str(worldpos_df.at[i, 'Hips.X'])
            s += ' ' + str(worldpos_df.at[i, 'Hips.Y'])
            s += ' ' + str(worldpos_df.at[i, 'Hips.Z'])
            cur_pose = get_pose_from_bvh(rotation_df, i)
            for each in cur_pose:
                s += ' ' + str(each)
            s += '\n'
            f.write(s)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        rotpath = 'E:\\Daiyudi\\Documents\\OneDrive - stu.xmu.edu.cn\\I_2021_HumanMotion\数据采集\\0719\\mocap_csv\\02_with_lidar_rot_trans_T.csv'
        lidar_file = "e:\\Daiyudi\Documents\\OneDrive - stu.xmu.edu.cn\\I_2021_HumanMotion\\数据采集\\0719\\02lidar\\traj_with_timestamp_变换后的轨迹_与mocap重叠部分.txt"
        out_dir = 'E:\\Daiyudi\\Documents\\OneDrive - stu.xmu.edu.cn\\I_2021_HumanMotion\数据采集\\0719\\mocap_csv\\SMPL'
    elif len(sys.argv) == 3:
        rotpath = sys.argv[1]
        lidar_file = sys.argv[2]
        out_dir = os.path.join(os.path.dirname(rotpath), 'SMPL')
    elif len(sys.argv) == 4:
        print(len(sys.argv))

        rotpath = sys.argv[1]
        lidar_file = sys.argv[2]
        out_dir = sys.argv[3]
    else:
        print('请输入正确参数! *rot.csv *lidar_traj.txt [out_dir] ')
        exit()

    os.makedirs(out_dir, exist_ok=True)
    smpl_out_dir = os.path.join(out_dir, Path(rotpath).stem)
    os.makedirs(smpl_out_dir, exist_ok=True)

    print('rotpath: ', rotpath)
    print('lidar_file: ', lidar_file)
    print('out_dir: ', out_dir)

    rotation_df = pd.read_csv(rotpath)
    lidar = np.loadtxt(lidar_file, dtype=float)
    with open(lidar_file) as f:
        lines = f.readlines()
    n = min(len(rotation_df), len(lines))
    smpl = SMPL()
    
    # mocap->lidar坐标系
    mocap_init = np.array([
        [-1, 0, 0],
        [0, 0, 1], 
        [0, 1, 0]])

    for i in range(min(lidar.shape[0], len(rotation_df))):
        vertices = smpl(torch.from_numpy(get_pose_from_bvh(
            rotation_df, i, False)).unsqueeze(0).float(), torch.zeros((1, 10)))
        vertices = vertices.squeeze().cpu().numpy()
        translation = lidar[i, 1:4]
        vertices = np.matmul(mocap_init, vertices.T).T + translation
        save_ply(vertices, os.path.join(smpl_out_dir, str(i) + '_smpl.ply'))
    print('SMPL saved in: ', smpl_out_dir)