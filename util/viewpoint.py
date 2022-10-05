################################################################################
# File: \viewpoint.py                                                          #
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

import numpy as np
from pkg_resources import ExtractionError
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
from .tool_func import filterTraj

def make_cloud_in_vis_center(point_cloud):
    center = point_cloud.get_center()
    yaw = np.arctan2(center[1], center[0])

    # rot the points, make them on the X-axis
    rot = R.from_rotvec(np.array([0, 0, -yaw])).as_matrix()

    # put points in 5 meters distance
    trans_x = 5 - (rot @ center)[0]

    rt = np.concatenate((rot.T, np.array([[trans_x, 0, -center[-1]]]))).T
    rt = np.concatenate((rt, np.array([[0, 0, 0, 1]])))

    point_cloud.transform(rt)
    # point_cloud.traslate(rt)

    return rt, center

def get_head_global_rots(pose, parents=[0, 3, 6, 9, 12, 15]):
    """
    It takes a pose (a 3D array of shape (n_frames, 24, 3)) and returns a 3D array of shape (n_frames,
    3, 3) where each 3x3 matrix is the rotation matrix of the head in the global coordinate system
    
    Args:
      pose: the pose to be transformed
      parents: The indices of the parent joints.
    
    Returns:
      The global rotation matrix of the head.
    """
    if pose.shape[1] == 72:
        pose = pose.reshape(-1, 24, 3)

    rots = np.eye(3)
    for r in parents[::-1]:
        rots = R.from_rotvec(pose[:, r]).as_matrix() @ rots
    rots = rots @ np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]).T
    return rots

def generate_views(position, direction, filter=True, rad=np.deg2rad(10), dist = 0.2):
    """
    > Given a list of positions and directions, generate a list of views and extrinsics
    
    Args:
      position: the position of the camera
      direction: the direction of the camera. This is a list of 3x3 matrices, where each matrix is the
    rotation matrix of the camera.
      filter: whether to filter the trajectory or not. Defaults to True
      rad: the angle of the camera from the ground
      dist: distance from the camera to the lookat point
    
    Returns:
      view_list is a list of dictionaries, each dictionary contains the trajectory of the camera.
    """
    assert len(position) == len(direction) 

    mocap_init = np.array([[-1, 0, 0, ], [0, 0, 1], [0, 1, 0]])
    base_view = {
        "trajectory" : 
        [
            {
                "field_of_view" : 80.0,
                "front" : [ 0, -np.cos(rad), np.sin(rad)],
                "lookat" : [ 0, 1, 1.7],
                "up" : [ 0, np.sin(rad), np.cos(rad)],
                "zoom" : 0.0065
            }
        ],
    }

    if direction[0].shape[0] == 4:
        func = R.from_quat
    else:
        func = R.from_matrix

    if filter:
       position = filterTraj(position)

    view_list = []
    extrinsic_list = []
    for t, r in zip(position, direction):
        view = deepcopy(base_view)
        rot = func(r).as_matrix()
        rot = R.from_rotvec(-rad * rot[:, 0]).as_matrix() @ rot

        view['trajectory'][0]['front'] = -rot[:, 1]
        view['trajectory'][0]['up'] = rot[:, 2] 
        view['trajectory'][0]['lookat'] = t + rot @ np.array([0, -dist, 0])
        # view_list.append(view)
        
        front = view['trajectory'][0]['front']
        up = view['trajectory'][0]['up']
        origin = view['trajectory'][0]['lookat']

        extrinsic_list.append(view_to_extrinsic(origin, up, front))
    
    return position, extrinsic_list

def view_to_extrinsic(lookat, up, front):
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.stack([-np.cross(front, up), -up, -front])
    extrinsic[:3, 3] = - (extrinsic[:3, :3] @ lookat)   # t = - R @ p
    return extrinsic

# def direction_to_view(r, delta_r):
    
#     if r.shape[0] == 4:
#         func = R.from_quat
#     else:
#         func = R.from_matrix

#     rot = func(r).as_matrix()
#     rot = R.from_rotvec(-rad * rot[:, 0]).as_matrix() @ rot

#     lookat = t + rot @ np.array([0, -dist, 0])
#     up = rot[:, 2] 
#     front = -rot[:, 1]

def extrinsic_to_view(extrinsic):
    up = -extrinsic[1, :3]
    front = -extrinsic[2, :3]
    lookat = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
    return lookat, up, front