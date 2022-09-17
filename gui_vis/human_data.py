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

from util import load_data_remote, generate_views, get_head_global_rots
from smpl import SMPL, poses_to_vertices

def vertices_to_joints(vertices, index = 15):
    # default index is head index
    smpl = SMPL()
    return smpl.get_full_joints(torch.FloatTensor(vertices))[..., index, :]

def make_3rd_view(positions, rots, rotz=0, lookdown=32):
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
    move = rotz @ np.array([0.5, -1, 1.2])

    rots = np.zeros_like(rots)
    for i in range(rots.shape[0]):
        rots[i] =  rotz @ lookdown
    views = generate_views(positions + move, rots, dist=0, rad=np.deg2rad(0))
    return views

def load_human_mesh(verts_list, human_data, start, end, pose_str='pose', tran_str='trans', trans_str2=None, info='First'):
    if pose_str in human_data:
        pose = human_data[pose_str].copy()
        beta = human_data['beta'].copy()
        if tran_str in human_data:
            trans = human_data[tran_str].copy()
        else:
            trans = human_data[trans_str2].copy()
        vert = poses_to_vertices(pose, trans, beta=beta)
        verts_list[f'{info}'] = vert[start:end]
        print(f'[SMPL MODEL] {info} ({pose_str}) loaded')

def load_vis_data(humans, start=0, end=-1):
    """
    It loads the data from the pickle file into a dictionary
    
    Args:
      humans: the dictionary containing the data
      start: the first frame to load. Defaults to 0
      end: the last frame to be visualized
    
    Returns:
      vis_data
    """

    vis_data = {}
    vis_data['humans'] = {}

    first_person = humans['first_person']
    pose = first_person['pose'].copy()
    beta = first_person['beta'].copy()
    end = pose.shape[0] if end <= 0 else end 
    if 'mocap_trans' in first_person:
        trans = first_person['mocap_trans'].copy()
    else:
        trans = first_person['trans'].copy()

    f_vert = poses_to_vertices(pose, beta=beta)

    # pose + trans
    vis_data['humans']['Baseline1(F)'] = f_vert[start: end] + \
        np.expand_dims(trans.astype(np.float32), 1)[start: end]

    # load first person
    if 'lidar_traj' in first_person:
        lidar_traj = first_person['lidar_traj'][:, 1:4]
        head = vertices_to_joints(f_vert, 15).numpy()
        root = vertices_to_joints(f_vert, 0).numpy()
        head_rots = get_head_global_rots(pose)

        def lidar2head(i):
            return head_rots[i].T @ (head[i] - lidar_traj[i] + trans[i])

        lidar_to_head = head_rots @ lidar2head(0)

        ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ### !!!   It's very important: -root[0]   !!
        ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        trans = lidar_traj + lidar_to_head - head + root - root[0]  
    
        vis_data['humans']['Baseline2(F)'] = f_vert[start: end] + \
            np.expand_dims(trans.astype(np.float32), 1)[start: end]
    
    print(f'[SMPL MODEL] First pose loaded')

    load_human_mesh(vis_data['humans'], 
                    first_person, 
                    start, 
                    end, 
                    'opt_pose', 
                    'opt_trans', 
                    info='Ours(F)')

    load_human_mesh(vis_data['humans'], 
                    first_person, 
                    start, 
                    end, 
                    'full_pose', 
                    'mocap_trans', 
                    info='Ours_opt(F)')

    if 'second_person' in humans:
        second_person = humans['second_person']    

        load_human_mesh(vis_data['humans'], 
                        second_person, 
                        start, 
                        end, 
                        'pose',
                        'mocap_trans', 
                        info='Baseline1(S)')

        load_human_mesh(vis_data['humans'], 
                        second_person, 
                        start, 
                        end, 
                        'pose',
                        'trans', 
                        info='Baseline2(S)')

        load_human_mesh(vis_data['humans'], 
                        second_person, 
                        start, 
                        end, 
                        'opt_pose',
                        'opt_trans', 
                        info='Ours(S)')

        if 'point_clouds' in second_person:
            global_frame_id = second_person['point_frame']
            global_frame_id = set(global_frame_id).intersection(humans['frame_num'][start: end])
            valid_idx = [humans['frame_num'].index(l) for l in global_frame_id]

            vis_data['point cloud'] = [second_person['point_clouds'], valid_idx]
            print(f'[PointCloud] Second point cloud loaded')

            if 'pred_pose' in second_person:
                pose = second_person['pred_pose'].copy()
                if 'opt_trans' in second_person:
                    trans = second_person['opt_trans'].copy()
                else:
                    trans = second_person['trans'].copy()
                local_id = [second_person['point_frame'].tolist().index(i) for i in global_frame_id]
                vis_data['humans']['Second pred'] = poses_to_vertices(pose[local_id], trans[valid_idx], beta = second_person['beta'])
                print(f'[SMPL MODEL] Predicted person loaded')

    return vis_data

class HUMAN_DATA:
    FOV = 'first'
    FREE_VIEW = False

    def __init__(self, is_remote=False):
        self.is_remote = is_remote
        self.cameras = {}

    def load(self, filename):
        load_data_class = load_data_remote(self.is_remote)
        try:
            self.humans = load_data_class.load_pkl(filename)
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
        if 'first_person' not in self.humans:
            self.humans = {'first_person': self.humans}
        self.vis_data_list = load_vis_data(self.humans)
        # self.set_cameras()

    def load_hdf5(self, filename):
        load_data_class = load_data_remote(self.is_remote)

        self.vis_data_list = load_vis_data(self.humans)

    def set_cameras(self, offset_center=-0.2):
        humans_verts = self.humans

        try:
            lidar_position = humans_verts['first_person']['lidar_traj'][:, 1:4]
            head_rots = get_head_global_rots(humans_verts['first_person']['pose'])
            self.cameras['First Lidar View'] = generate_views(lidar_position, head_rots, dist=offset_center)
        except Exception as e:
            print(e)
            print(f'No First Lidar View')

        try:
            verts = self.vis_data_list['humans']['Baseline2(F)']
            root_position = vertices_to_joints(verts, 0)
            root_rots = get_head_global_rots(humans_verts['first_person']['pose'], parents=[0])

            self.cameras['First root View'] = generate_views(root_position, root_rots, rad=np.deg2rad(-10), dist=-0.3)
            self.cameras['3rd View +Y'] = make_3rd_view(root_position, root_rots, rotz=0)
            self.cameras['3rd View -X'] = make_3rd_view(root_position, root_rots, rotz=90)
            self.cameras['3rd View -Y'] = make_3rd_view(root_position, root_rots, rotz=180)
            self.cameras['3rd View +X'] = make_3rd_view(root_position, root_rots, rotz=270)
            self.cameras['3rd View +Z'] = make_3rd_view(root_position, root_rots, rotz=0, lookdown=90)

        except Exception as e:
            print(e)
            print(f'No First root View')

        try:
            try:
                second_verts = self.vis_data_list['humans']['Ours(S)']
                second_pose = humans_verts['second_person']['opt_pose']
            except:
                try:
                    second_verts = self.vis_data_list['humans']['Baseline2(S)']
                    second_pose = humans_verts['second_person']['pose']
                except Exception as e:
                    print(e)
                    print(f'There is no second pose in the data')
            position = vertices_to_joints(second_verts) + np.array([0, 0, 0.2])
            rotation = get_head_global_rots(second_pose)
            self.cameras['Second View'] = generate_views(position, rotation, dist=offset_center)

            position = vertices_to_joints(second_verts, 0)
            rotation = get_head_global_rots(second_pose, parents=[0])
            self.cameras['(p2) 3rd View +Y'] = make_3rd_view(position, rotation, rotz=0)
            self.cameras['(p2) 3rd View -X'] = make_3rd_view(position, rotation, rotz=90)
            self.cameras['(p2) 3rd View -Y'] = make_3rd_view(position, rotation, rotz=180)
            self.cameras['(p2) 3rd View +X'] = make_3rd_view(position, rotation, rotz=270)
        except Exception as e:
            print(e)

        views = list(self.cameras.keys())
        for view in views:
            print(f'[Camera]: {view}')
        return views

    def get_extrinsic(self, FOV):
        return self.cameras[FOV]
