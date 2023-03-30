################################################################################
# File: \tool_func.py                                                          #
# Created Date: Monday July 18th 2022                                          #
# Author: climbingdaily                                                        #
# -----                                                                        #
# Modified By: the developer climbingdaily at yudidai@stu.xmu.edu.cn           #
# https://github.com/climbingdaily                                             #
# -----                                                                        #
# Copyright (c) 2022 yudidai                                                   #
# -----                                                                        #
# HISTORY:                                                                     #
################################################################################

from logging import exception
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from subprocess import run
import time
import json

def read_json_file(file_name):
    """
    Reads a json file
    Args:
        file_name:
    Returns:
    """
    with open(file_name) as f:
        try:
            data = json.load(f)
        except:
            data = {}
    return data

def extrinsic_to_cam(extrinsic):
    cam = np.eye(4)
    cam[:3, :3] = extrinsic[:3, :3].T @ np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    cam[:3, 3] = -(extrinsic[:3, :3].T @ extrinsic[:3, 3])
    return cam

def cam_to_extrinsic(cam):
    """
    It takes a camera matrix and returns the extrinsic matrix
    
    Args:
      cam: the camera matrix
    
    Returns:
      The extrinsic matrix
    """
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) @ cam[:3, :3].T
    extrinsic[:3, 3] = -(extrinsic[:3, :3] @ cam[:3, 3])
    return extrinsic

def plot_kpt_on_img(img, kpts):
    # @wzj
    if kpts is not None:
        # @wzj
        return 
    else:
        return img

def images_to_video(img_dir, filename=None, delete=False, inpu_fps=20):
    if filename is None:
        filename = os.path.basename(img_dir) + time.strftime("-%Y-%m-%d_%H-%M", time.localtime())

    video_path = os.path.join(os.path.dirname(img_dir), 'vis_data', f'{filename}.mp4')
    # video_path2 = os.path.join(os.path.dirname(img_dir), 'vis_data', f'{filename}.avi')
    os.makedirs(os.path.join(os.path.dirname(img_dir), 'vis_data'), exist_ok=True)
    # command = f"ffmpeg -f image2 -i {path}\\{filename}_%4d.jpg -b:v 10M -c:v h264 -r 20  {video_path}"
    command = f"ffmpeg -f image2 -threads 8 -r {inpu_fps} -start_number 1 -i \"{img_dir}\\%5d.jpg\" -b:v 10M -c:v h264 -r 20  \"{video_path}\""
    if os.path.exists(video_path) or os.path.exists(video_path):
        return False, f"'{video_path}' existed."
    elif not os.path.exists(img_dir):
        return False, f"'{img_dir}' not existed."
    else:
        try:
            run(command, shell=True)
        except:
            print(f'[command error] \"{command}\"')

        if delete :
            try:
            # ! Danger
                img_list = os.listdir(img_dir)
                for img in img_list:
                    os.remove(os.path.join(img_dir, img))
                    time.sleep(0.001)
                os.removedirs(img_dir)
            except Exception as e:
                print(e)
        return True, f'Video saved in {video_path}'

def filterTraj(traj_xyz, fit_time=None, segment=20, frame_time=0.05, keep_data = False):

    if fit_time is None:
        fit_time = np.arange(len(traj_xyz)) * frame_time
    
    times = fit_time.copy()

    # 2. Spherical Linear Interpolation of Rotations.
    from scipy.spatial.transform import RotationSpline

    rotation = False
    if rotation:
        R_quats = R.from_quat(traj_xyz[:, 4: 8])
        spline = RotationSpline(times, R_quats)
        quats_plot = spline(fit_time).as_quat()

    trajs_plot = []  # 拟合后轨迹

    length = traj_xyz.shape[0]
    for i in range(0, length, segment):
        s = max(0, i-1)   # start index
        e = i+segment   # end index
        if length < e:
            s = length - segment
            e = length

        ps = s - segment//2  # filter start index
        pe = e + segment//2  # # filter end index

        if ps < 0:
            ps = 0
            pe += segment//2
        if pe > length:
            ps -= segment//2
            pe = length

        fp = np.polyfit(times[ps:pe], traj_xyz[ps:pe], 3)  # 分段拟合轨迹

        fs = 0 if s == 0 else np.where(fit_time == times[i - 1])[0][0] # 拟合轨迹到起始坐标
        fe = np.where(fit_time == times[e-1])[0][0]  # 拟合轨迹到结束坐标
        fe = fe+1 if e == length else fe

        for j in fit_time[fs: fe]:
            trajs_plot.append(np.polyval(fp, j))


    frame_id = -1 * np.ones(len(trajs_plot))
    old_id = [np.where(fit_time==t)[0][0] for t in times]
    frame_id[old_id] = old_id
    interp_idx = np.where(frame_id == -1)[0]

    frame_id = np.arange(len(trajs_plot))

    # fitLidar = np.concatenate(
    #     (frame_id.reshape(-1, 1), np.asarray(trajs_plot), quats_plot, fit_time.reshape(-1, 1)), axis=1)
    fit_traj = trajs_plot

    if keep_data:
        fit_traj[old_id] = traj_xyz

    return fit_traj
