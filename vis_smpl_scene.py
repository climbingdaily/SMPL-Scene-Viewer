################################################################################
# File: \vis_double.py                                                         #
# Created Date: Sunday July 17th 2022                                          #
# Author: climbingdaily                                                        #
# -----                                                                        #
# Modified By: the developer climbingdaily at yudidai@stu.xmu.edu.cn           #
# https://github.com/climbingdaily                                             #
# -----                                                                        #
# Copyright (c) 2022 yudidai                                                   #
# -----                                                                        #
# HISTORY:                                                                     #
# 2022-08-10	ABC	
################################################################################

import numpy as np
import configargparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from util import o3dvis
import matplotlib.pyplot as plt
import torch
import os
import time

from smpl import SMPL, poses_to_vertices
from util import load_data_remote, generate_views, load_scene, images_to_video

view = {
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 68.419929504394531, 39.271018981933594, 11.569537162780762 ],
			"boundingbox_min" : [ -11.513210296630859, -35.915927886962891, -2.4593989849090576 ],
			"field_of_view" : 60.0,
			"front" : [ 0.28886465410454343, -0.85891896928352873, 0.42286571841896009 ],
			"lookat" : [ 0.76326815774101275, 3.2896492351216851, 0.040108816664781548 ],
			"up" : [ -0.12866047345544837, 0.40286011796513765, 0.90617338734004726 ],
			"zoom" : 0.039999999999999994
		}
	],
}

POSE_KEY = ['First opt_pose', 'First pose', 'Second opt_pose', 'Second pose', 'Second pred']
POSE_COLOR = {'points': plt.get_cmap("tab20")(1)[:3]}
for i, color in enumerate(POSE_KEY):
    POSE_COLOR[color] = plt.get_cmap("Pastel1")(i)[:3]
    
def vertices_to_head(vertices, index = 15):
    smpl = SMPL()
    return smpl.get_full_joints(torch.FloatTensor(vertices))[..., index, :]

def get_head_global_rots(pose):
    if pose.shape[1] == 72:
        pose = pose.reshape(-1, 24, 3)

    head_parents = [0, 3, 6, 9, 12, 15]
    head_rots = np.eye(3)
    for r in head_parents[::-1]:
        head_rots = R.from_rotvec(pose[:, r]).as_matrix() @ head_rots
    head_rots = head_rots @ np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]).T
    return head_rots

def load_human_mesh(verts_list, human_data, start, end, pose_str='pose', tran_str='trans', trans_str2=None, info='First'):
    if pose_str in human_data:
        pose = human_data[pose_str].copy()
        beta = human_data['beta'].copy()
        if tran_str in human_data:
            trans = human_data[tran_str].copy()
        else:
            trans = human_data[trans_str2].copy()
        vert = poses_to_vertices(pose, trans, beta=beta)
        verts_list[f'{info} {pose_str}'] = vert[start:end]
        print(f'[SMPL MODEL] {info} {pose_str} loaded')

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

    # load first person
    if 'lidar_traj' in first_person:
        lidar_traj = first_person['lidar_traj'][:, 1:4]
        head = vertices_to_head(f_vert, 15)
        root = vertices_to_head(f_vert, 0)
        head_to_root = (root - head).numpy()
        
        head_rots = get_head_global_rots(pose)

        lidar_to_head =  head[0] - lidar_traj[0] + trans[0] 
        lidar_to_head = head_rots @ head_rots[0].T @ lidar_to_head.numpy()

        trans = lidar_traj + lidar_to_head + head_to_root
        # trans = lidar_traj[:, 1:4]
    
    f_vert += np.expand_dims(trans.astype(np.float32), 1)

    vis_data['humans']['First pose'] = f_vert[start: end]
    print(f'[SMPL MODEL] First person loaded')

    load_human_mesh(vis_data['humans'], first_person, start, end, 'opt_pose', 'opt_trans')

    if 'second_person' in humans:
        second_person = humans['second_person']    

        load_human_mesh(vis_data['humans'], second_person, start, end, 'pose',
                        'trans', 'mocap_trans', info='Second')

        load_human_mesh(vis_data['humans'], second_person, start, end, 'opt_pose',
                        'opt_trans', info='Second')

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

def vis_pt_and_smpl(vis, vis_data, extrinsics=None, video_name=None, freeviewpoint=False):
    """
    > This function takes in a point cloud and a SMPL mesh, and visualizes them in a 3D viewer
    
    Args:
      vis: the visualization object
      vis_data: a dictionary containing the point cloud and the SMPL meshes.
      extrinsics: the camera extrinsics for each frame.
      video_name: the name of the video you want to visualize
      freeviewpoint: if True, the camera will move relative to the previous frame. If False, the camera
    will be fixed. Defaults to False
    """

    pointcloud = o3d.geometry.PointCloud()
    smpl_geometries = []
    smpl_materials = []
    human_data = vis_data['humans']
    points = vis_data['point cloud'][0]
    indexes = vis_data['point cloud'][1]

    mat_ground = o3d.visualization.Material("defaultLitSSR")
    mat_ground.scalar_properties['roughness'] = 0.15
    mat_ground.scalar_properties['reflectance'] = 0.72
    mat_ground.scalar_properties['transmission'] = 0.6
    mat_ground.scalar_properties['thickness'] = 0.3
    mat_ground.scalar_properties['absorption_distance'] = 0.1
    mat_ground.vector_properties['absorption_color'] = np.array(
        [0.82, 0.98, 0.972, 1.0])

    for i in human_data:
        smpl = o3d.io.read_triangle_mesh('.\\smpl\\sample.ply')
        # smpl_m = o3d.t.geometry.TriangleMesh.from_legacy(smpl)
        # smpl_m.material = mat_ground
        # smpl_materials.append(smpl_m)
        smpl_geometries.append(smpl) # a ramdon SMPL mesh

    init_param = False

    vis.img_save_count = 0
    video_name += time.strftime("-%Y-%m-%d_%H-%M", time.localtime())
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'temp_{video_name}')
    keys = list(human_data.keys())

    for i in range(human_data[keys[0]].shape[0]):
        if i in indexes:
            index = indexes.index(i)
            pointcloud.points = o3d.utility.Vector3dVector(points[index])
        else:
            index = -1
            pointcloud.points = o3d.utility.Vector3dVector(np.array([[0,0,0]]))

        pointcloud.normals = o3d.utility.Vector3dVector()
        pointcloud.paint_uniform_color(POSE_COLOR['points'])

        for idx, smpl in enumerate(smpl_geometries):
            key = keys[idx]
            if 'pred' in key.lower():
                if index >= 0:
                    smpl.vertices = o3d.utility.Vector3dVector(human_data[key][index])
                    smpl.paint_uniform_color(POSE_COLOR[key])
                else:
                    smpl.vertices = o3d.utility.Vector3dVector(np.asarray(smpl.vertices) * 0)
            elif 'first' in key.lower():
                smpl.vertices = o3d.utility.Vector3dVector(human_data[key][i])
                smpl.paint_uniform_color(POSE_COLOR[key])
            elif 'second' in key.lower():
                smpl.vertices = o3d.utility.Vector3dVector(human_data[key][i])
                smpl.paint_uniform_color(POSE_COLOR[key])
            
            smpl.vertex_normals = o3d.utility.Vector3dVector()
            smpl.compute_vertex_normals()

        if extrinsics is not None:
            # vis.set_view(view_list[i])
            if i > 0 and freeviewpoint:
                camera_pose = vis.get_camera()
                # relative_pose = extrinsics[i] @ np.linalg.inv(extrinsics[i-1])
                relative_trans = -extrinsics[i][:3, :3].T @ extrinsics[i][:3, 3] + extrinsics[i-1][:3, :3].T @ extrinsics[i-1][:3, 3]
                
                camera_positon = -(camera_pose[:3, :3].T @ camera_pose[:3, 3])
                camera_pose[:3, 3] = -(camera_pose[:3, :3] @ (camera_positon + relative_trans))
                vis.init_camera(camera_pose)
            else:
                vis.init_camera(extrinsics[i])   
                
        # add to visualization
        if not init_param:
            vis.add_geometry(pointcloud, reset_bounding_box = False, name='human points')  
            for si, smpl in enumerate(smpl_geometries):
                vis.add_geometry(smpl, reset_bounding_box = False, name=keys[si])  
            vis.change_pause_status()
            init_param = True

        else:
            vis.update_geometry(pointcloud,  name='human points') 
            
            for si, smpl in enumerate(smpl_geometries):
                vis.update_geometry(smpl, name=keys[si])  
 
        vis.save_imgs(image_dir)
        
        vis.waitKey(20, helps=False)
            
    images_to_video(image_dir, video_name, delete=True)

    for g in smpl_geometries:
        vis.remove_geometry(g)

if __name__ == '__main__':    
    import config
    parser = configargparse.ArgumentParser()
    parser.add_argument("--start", '-S', type=int, default=-2)
    parser.add_argument("--end", '-e', type=int, default=-2)
    parser.add_argument("--scene_path", '-s', type=str,default=None)
    parser.add_argument("--smpl_file_path", '-F', type=str, default=None)
    parser.add_argument("--viewpoint_type", '-V', type=str, default=None)

    args, opts = parser.parse_known_args()

    start = config.start if args.start == -2 else args.start
    end = config.end if args.end == -2 else args.end
    scene_path = config.scene_path if args.scene_path is None else args.scene_path
    is_remote = True if '--remote' in opts else config.remote

    smpl_file_path = config.smpl_file_path if args.smpl_file_path is None else args.smpl_file_path
    viewpoint_type = config.viewpoint_type if args.viewpoint_type is None else args.viewpoint_type

    vis = o3dvis("Humans and scen vis", width=1280, height=720)
    load_data_class = load_data_remote(is_remote)

    scene = load_scene(vis, scene_path)

    if not scene.has_points():
        scene = load_scene(vis, scene_path, load_data_class = load_data_class)

    print(f'Load pkl in {smpl_file_path}')

    humans = load_data_class.load_pkl(smpl_file_path)

    vis_data_list = load_vis_data(humans, start, end)

    scene_name = os.path.basename(scene_path).split('.')[0]

    freeview = False

    POVs, extrinsics = generate_views(humans['first_person']['lidar_traj']
                        [:, 1:4], get_head_global_rots(humans['first_person']['pose']))

    if viewpoint_type == 'second':
        if 'Second pose' in vis_data_list:         
            POVs, extrinsics = generate_views(vertices_to_head(
                vis_data_list['Second pose']) + np.array([0, 0, 0.2]), get_head_global_rots(humans['second_person']['pose']))
        else:
            print(f'There is no second pose in the data')

    elif viewpoint_type == 'third':        
        freeview = True

    vis_pt_and_smpl(vis, vis_data_list, extrinsics=extrinsics, 
                    video_name = scene_name + f'_{viewpoint_type}', freeviewpoint=freeview)
