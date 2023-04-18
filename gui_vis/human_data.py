################################################################################
# File: \human_data.py                                                         #
# Created Date: Saturday August 13th 2022                                      #
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
import torch
from scipy.spatial.transform import Rotation as R

from util import Data_loader, generate_views, get_head_global_rots
from smpl import SMPL, poses_to_vertices

def vertices_to_joints(vertices, index = 15):
    # default index is head index
    smpl = SMPL()
    return smpl.get_full_joints(torch.FloatTensor(vertices))[..., index, :].numpy()

def make_3rd_view(positions, rots, rotz=0, lookdown=12, move_back = 1, move_up = 1.0, move_right = 0.5, filter=True):
    """
    It takes the positions and rotations of the camera, and returns the positions and rotations of the
    camera, but with the camera rotated by `rotz` degrees around the z-axis, and moved to a new position
    
    Args:
      positions: the position of the camera in the world
      rots: the rotation of the camera
      rotz: rotation around the z axis. Defaults to 0
      lookdown: the angle of the camera, in degrees. Defaults to 32
    """
    lookdown = R.from_rotvec(np.deg2rad(-lookdown) * np.array([1, 0, 0])).as_matrix()
    rotz = R.from_rotvec(np.deg2rad(rotz) * np.array([0, 0, 1])).as_matrix()
    offset = np.array([move_right, -move_back, move_up])

    # rots = np.zeros_like(rots)
    for i in range(rots.shape[0]):
        vv = rots[i][:2,1]
        rot = np.eye(3)
        rot[:2,1] = vv / np.linalg.norm(vv)
        rot[0,0] = rot[1,1]
        rot[1,0] = -rot[0,1]
        rots[i] =  rot @ rotz @ lookdown

    cam_pos = positions + (rots @ lookdown.T @ offset).squeeze()
    _, extrinsics = generate_views(cam_pos, rots, dist=0, rad=np.deg2rad(0), filter=filter)
    return positions, extrinsics

def load_human_mesh(verts_list, human_data, start, end, pose_str='pose', tran_str='trans', trans_str2=None, rot=None, info='First'):
    if pose_str not in human_data:
        pose_str = 'pose'
    if pose_str in human_data:
        pose = human_data[pose_str].copy()
        if rot is not None:
            if human_data[rot].shape[1] == 72:
                pose[:, :3] = human_data[rot][:, :3].copy()
            elif human_data[rot].shape[1] == 24:
                pose[:, :1] = human_data[rot][:, :1].copy()
        
        if 'beta' not in human_data:
            beta = [0] * 10
        else:
            beta = human_data['beta'].copy()

        if 'gender' not in human_data:
            gender = 'male'
        else:
            gender = human_data['gender']

        if tran_str in human_data:
            trans = human_data[tran_str].copy()
        elif trans_str2 is not None and trans_str2 in human_data:
            trans = human_data[trans_str2].copy()
        else:
            return
        vert = poses_to_vertices(pose, trans, beta=beta, gender=gender)
        verts_list[f'{info}'] = {'verts': vert[start:end], 'trans': trans[start:end], 'pose': pose[start:end]}
        print(f'[SMPL MODEL] {info} ({pose_str}) loaded')

def load_vis_data(humans, start=0, end=-1, data_format=None):
    """
    > This function loads the SMPL model and the point cloud data into the `vis_data` dictionary
    
    Args:
      humans: the dictionary containing the data
      start: the start frame of the video. Defaults to 0
      end: the end frame of the video
    """
    import os
    vis_data = {}
    vis_data['humans'] = {}

    for person in data_format:
        if person in humans and 'pose' in humans[person]:
            end = humans[person]['pose'].shape[0] if end <= 0 else end 
            if 'beta' not in humans[person]:
                humans[person]['beta'] = [0] * 10
            if 'gender' not in humans[person]:
                humans[person]['gender'] = 'male'

            if 'point_clouds' in humans[person]:
                global_frame_id = humans[person]['point_frame']
                global_frame_id = set(global_frame_id).intersection(humans['frame_num'][start: end])
                valid_idx = [humans['frame_num'].index(l) for l in global_frame_id]
                vis_data['point cloud'] = [humans[person]['point_clouds'], valid_idx]
                print(f'[PointCloud] {person} point cloud loaded')

            for info, values in data_format[person].items():
                
                if values['pose'] == 'pred_pose' and 'pred_pose' in humans[person]:
                    # load predicted pose
                    pose = humans[person]['pred_pose'].copy()
                    if 'opt_trans' in humans[person]:
                        trans = humans[person]['opt_trans'].copy()
                    else:
                        trans = humans[person]['trans'].copy()
                    local_id = [humans[person]['point_frame'].tolist().index(i) for i in global_frame_id]
                    verts = poses_to_vertices(pose[local_id], 
                                              trans[valid_idx], 
                                              beta = humans[person]['beta'], 
                                              gender=humans[person]['gender'])
                    vis_data['humans']['Pred(S)'] = {
                        'verts': verts, 
                        'trans': trans[valid_idx], 
                        'pose': pose[local_id]}
                    print(f'[SMPL MODEL] Predicted person loaded')

                elif values['trans'] == 'lidar_traj' and 'lidar_traj' in humans[person]:
                    # load lidar_traj
                    f_vert = vis_data['humans']['Baseline1(F)']['verts']
                    pose = vis_data['humans']['Baseline1(F)']['pose']
                    trans = vis_data['humans']['Baseline1(F)']['trans']
                    lidar_traj = humans[person]['lidar_traj'][:, 1:4]
                    head = vertices_to_joints(f_vert, 15)
                    root = vertices_to_joints(f_vert, 0)
                    head_rots = get_head_global_rots(pose)

                    def lidar2head(i):
                        return head_rots[i].T @ (head[i] - lidar_traj[i])

                    lidar_to_head = head_rots @ lidar2head(0)
                    head_to_root = root - head
                    smpl_offset = trans[0] - root[0]

                    lidar_trans = lidar_traj[start: end] + lidar_to_head + head_to_root + smpl_offset

                    vis_data['humans'][info] = {'verts': f_vert-trans[:, None, :]+lidar_trans[:, None, :], 
                                                'trans': lidar_trans.squeeze(),
                                                'pose': pose}
                else:
                    rot = values['rot'] if 'rot' in values else None
                    load_human_mesh(vis_data['humans'], 
                                    humans[person], 
                                    start, 
                                    end, 
                                    values['pose'],
                                    values['trans'], 
                                    values['trans_bak'], 
                                    rot,
                                    info=info)

    print(f'[SMPL LOADED] ==============')

    return vis_data

class HUMAN_DATA:
    FOV = 'first'

    def __init__(self, is_remote=False, data_format=None):
        self.is_remote = is_remote
        self.cameras = {}
        self.humans = {}
        self.data_format = data_format

    def load(self, filename):
        data_loader = Data_loader(self.is_remote)
        try:
            self.humans = data_loader.load_pkl(filename)
        except Exception as e:
            print(e)
            """
            implement your function here
            self.humans = 
              {
                'first_person': {'pose':[N, 72], 'trans':[N, 3], 'beta':[10] }
                'second_person': {'pred_pose':[optional], 'trans':[optional], 'point_clouds':[N, x, 3], 'point_frame'[range(N)], 'beta':[10] }
                'frame_num': {[range(N)] }
              }
            # second_person is optional
            """
        # if 'first_person' not in self.humans or 'second_person' not in self.humans:
            # self.humans = {'first_person': self.humans}
        self.vis_data_list = load_vis_data(self.humans, data_format = self.data_format)
        # self.set_cameras()

    def load_hdf5(self, filename):
        data_loader = Data_loader(self.is_remote)

        self.vis_data_list = load_vis_data(self.humans)

    def set_cameras(self, offset_center=-0.2):
        if 'cameras' not in self.data_format:
            print("The camera information is not define in 'smpl_key.json'")
            cameras = {self.vis_data_list['humans'].keys()[0]: {"abbr": "(RANDOM)"}}
        else:
            cameras = self.data_format['cameras']

        for name, camera in cameras.items():
            if name in self.vis_data_list['humans']:
                try:
                    abbr = camera['abbr']
                    verts = self.vis_data_list['humans'][name]['verts']
                    position = vertices_to_joints(verts, 0)
                    head_rotation = get_head_global_rots(self.vis_data_list['humans'][name]['pose'])
                    self.cameras[f'{abbr} View'] = generate_views(position + np.array([0, 0, 0.2]), head_rotation, dist=offset_center)

                    rotation = get_head_global_rots(self.vis_data_list['humans'][name]['pose'], parents=[0])

                    self.cameras[f'{abbr} 3rd View back +3m'] = make_3rd_view(position, rotation, rotz=0, lookdown=10, move_back=3, move_up=0.3, move_right=0)
                    self.cameras[f'{abbr} 3rd View front +3m'] = make_3rd_view(position, rotation, rotz=180, lookdown=10, move_back=3, move_up=0.3, move_right=0)
                    self.cameras[f'{abbr} 3rd View left +3m'] = make_3rd_view(position, rotation, rotz=90, lookdown=10, move_back=3, move_up=0.3, move_right=0)
                    self.cameras[f'{abbr} 3rd View right +3m'] = make_3rd_view(position, rotation, rotz=-90, lookdown=10, move_back=3, move_up=0.3, move_right=0)
                    self.cameras[f'{abbr} 3rd View +Z'] = make_3rd_view(position, rotation, rotz=0, lookdown=90, move_up=6)
                except Exception as e:
                    print(f'[WARNING] No {name}. Some error occured in {e.args[0]}')
            elif 'trans' in camera:
                # lidar trajectory setting
                try:
                    position = self.humans[camera['person']][camera['trans']][:, 1:4]
                    head_rots = R.from_quat(self.humans[camera['person']][camera['direction']][:, 4:8]).as_matrix()
                    rotation = head_rots @ head_rots[0].T
                    self.cameras[name] = generate_views(position, rotation, filter=False, dist=0, rad=0)
                except Exception as e:
                    print(f'[WARNING] No {name}. Some error occured in {e.args[0]}')
            
        views = list(self.cameras.keys())
        for view in views:
            print(f'[Camera]: {view}')
        print(f'[Camera loaded]')
        
        return views

    def get_extrinsic(self, FOV):
        return self.cameras[FOV]
