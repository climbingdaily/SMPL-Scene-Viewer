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
from util.viewpoint import extrinsic_to_view

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

def load_human_mesh(verts_list, human_data, start, end, pose_str='pose', tran_str='trans', trans_str2=None, info='First'):
    if pose_str in human_data:
        pose = human_data[pose_str].copy()
        
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

    if 'pose' in first_person:
        end = first_person['pose'].shape[0] if end <= 0 else end 
        pose = first_person['pose'][start:end].copy()
        if 'beta' not in first_person:
            first_person['beta'] = [0] * 10
        if 'gender' not in first_person:
            first_person['gender'] = 'male'

        beta = first_person['beta'].copy()
        gender = first_person['gender']

        if 'mocap_trans' in first_person:
            trans = first_person['mocap_trans'].copy()
        else:
            trans = first_person['trans'].copy()
        f_vert = poses_to_vertices(pose, beta=beta, gender=gender)

        # pose + trans
        save_trans = np.expand_dims(trans.astype(np.float32), 1)[start: end]
        vis_data['humans']['Baseline1(F)'] = {
            'verts': f_vert + save_trans, 
            'trans': save_trans,
            'pose': pose}

        # load first person
        if 'lidar_traj' in first_person:
            lidar_traj = first_person['lidar_traj'][:, 1:4]
            head = vertices_to_joints(f_vert, 15)
            root = vertices_to_joints(f_vert, 0)
            head_rots = get_head_global_rots(pose)

            def lidar2head(i):
                return head_rots[i].T @ (head[i] - lidar_traj[i] + trans[i])

            lidar_to_head = head_rots @ lidar2head(0)

            ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            ### !!!   It's very important: -root[0]   !!
            ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            trans = lidar_traj + lidar_to_head - head + root - root[0]  
            trans = np.expand_dims(trans.astype(np.float32), 1)[start: end]

            vis_data['humans']['Baseline2(F)'] = {'verts': f_vert + trans, 
                                                  'trans': trans.squeeze(),
                                                  'pose': pose}
            
    
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
        if 'pose' in second_person and end <= 0:
            end = second_person['pose'].shape[0]

        if 'gender' not in second_person:
            second_person['gender'] = 'male'

        gender = second_person['gender']

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
                        'glamr_pose',
                        'glamr_tran', 
                        info='GLAMR(S)')

        load_human_mesh(vis_data['humans'], 
                        second_person, 
                        start, 
                        end, 
                        'glamr_pose',
                        'opt_trans', 
                        info='GLAMR_our_trans(S)')

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
                verts = poses_to_vertices(pose[local_id], trans[valid_idx], beta = second_person['beta'], gender=gender)
                vis_data['humans']['Pred(S)'] = {
                    'verts': verts, 
                    'trans': trans[valid_idx], 
                    'pose': pose[local_id]}
                print(f'[SMPL MODEL] Predicted person loaded')

    print(f'[Data loading end] ==============')

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

        # first lidar view generation
        try:
            lidar_pos = humans_verts['first_person']['lidar_traj'][:, 1:4]
            if 'pose' not in humans_verts['first_person']:
                head_rots = R.from_quat(humans_verts['first_person']['lidar_traj'][:, 4:8]).as_matrix()
                head_rots = head_rots @ head_rots[0].T
            else:    
                head_rots = get_head_global_rots(humans_verts['first_person']['pose'])
            self.cameras['First Lidar View'] = generate_views(lidar_pos, head_rots, dist=offset_center)
        except Exception as e:
            print(e)
            print(f'No First Lidar View')
            
            try:
                verts = self.vis_data_list['humans']['Baseline1(F)']['verts']
                root_position = vertices_to_joints(verts, 0)
                root_rots = get_head_global_rots(humans_verts['first_person']['pose'], parents=[0])
                self.cameras['First root View'] = generate_views(root_position, root_rots, rad=np.deg2rad(-10), dist=-0.3)
                
            except Exception as e:
                print(e)
                print(f'No First root View')
        
        # first person view generation
        try:
            try:
                verts = self.vis_data_list['humans']['Baseline2(F)']['verts']
                root_position = vertices_to_joints(verts, 0)
                root_rots = get_head_global_rots(humans_verts['first_person']['pose'], parents=[0])
            except Exception as e:
                print(e)
                try:
                    verts = self.vis_data_list['humans']['Baseline1(F)']['verts']
                    root_position = vertices_to_joints(verts, 0)
                    root_rots = get_head_global_rots(humans_verts['first_person']['pose'], parents=[0])
                except Exception as e: 
                    print(e)
            self.cameras['First root View'] = generate_views(root_position, root_rots, rad=np.deg2rad(-10), dist=-0.3)
            self.cameras['3rd View back +3m'] = make_3rd_view(root_position, root_rots, rotz=0, lookdown=10, move_back=3, move_up=0.3, move_right=0, filter=False)
            self.cameras['3rd View front +3m'] = make_3rd_view(root_position, root_rots, rotz=180, lookdown=10, move_back=3, move_up=0.3, move_right=0, filter=False)
            # self.cameras['3rd View +Y'] = make_3rd_view(root_position, root_rots, rotz=0)
            # self.cameras['3rd View -X'] = make_3rd_view(root_position, root_rots, rotz=90)
            # self.cameras['3rd View -Y'] = make_3rd_view(root_position, root_rots, rotz=180)
            # self.cameras['3rd View +X'] = make_3rd_view(root_position, root_rots, rotz=270)
            self.cameras['3rd View +Z'] = make_3rd_view(root_position, root_rots, rotz=0, lookdown=90, move_up=6)

        except Exception as e:
            print(e)
            print(f'No First person View')

        # second person view generation
        try:
            try:
                second_verts = self.vis_data_list['humans']['Ours(S)']['verts']
                second_pose = humans_verts['second_person']['opt_pose']
            except:
                try:
                    second_verts = self.vis_data_list['humans']['Baseline2(S)']['verts']
                    second_pose = humans_verts['second_person']['pose']
                except Exception as e:
                    print(e)
                    print(f'There is no second pose in the data')
            rotation = get_head_global_rots(second_pose)
            position = vertices_to_joints(second_verts) + np.array([0, 0, 0.2])
            self.cameras['Second View'] = generate_views(position, rotation, dist=offset_center)

            position = vertices_to_joints(second_verts, 0)
            rotation = get_head_global_rots(second_pose, parents=[0])
            self.cameras['(p2) 3rd View back +3m'] = make_3rd_view(position, rotation, rotz=0, lookdown=10, move_back=3, move_up=0.3, move_right=0)
            self.cameras['(p2) 3rd View front +3m'] = make_3rd_view(position, rotation, rotz=180, lookdown=10, move_back=3, move_up=0.3, move_right=0)
            self.cameras['(p2) 3rd View left +3m'] = make_3rd_view(position, rotation, rotz=90, lookdown=10, move_back=3, move_up=0.3, move_right=0)
            self.cameras['(p2) 3rd View right +3m'] = make_3rd_view(position, rotation, rotz=-90, lookdown=10, move_back=3, move_up=0.3, move_right=0)
            # self.cameras['(p2) 3rd View +Y'] = make_3rd_view(position, rotation, rotz=0)
            # self.cameras['(p2) 3rd View -X'] = make_3rd_view(position, rotation, rotz=90)
            # self.cameras['(p2) 3rd View -Y'] = make_3rd_view(position, rotation, rotz=180)
            # self.cameras['(p2) 3rd View +X'] = make_3rd_view(position, rotation, rotz=270)
            self.cameras['(p2) 3rd View +Z'] = make_3rd_view(position, rotation, rotz=0, lookdown=90, move_up=6)
        except Exception as e:
            print(e)

        views = list(self.cameras.keys())
        for view in views:
            print(f'[Camera]: {view}')
        print(f'[Camera loaded]')
        
        return views

    def get_extrinsic(self, FOV):
        return self.cameras[FOV]
