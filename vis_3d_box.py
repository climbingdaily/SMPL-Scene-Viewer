################################################################################
# File: \vis_3d_box.py                                                         #
# Created Date: Sunday July 17th 2022                                          #
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
import os
import configargparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from util.o3dvis import o3dvis
import matplotlib.pyplot as plt
import copy
from util import load_data_remote, make_cloud_in_vis_center, hidden_point_removal, select_points_on_the_scan_line

mat_box = o3d.visualization.rendering.MaterialRecord()
# mat_box.shader = 'defaultUnlit'
mat_box.shader = 'defaultLitTransparency'

cmap = np.array([
    [245, 150, 100, 255],
    [245, 230, 100, 255],
    [150, 60, 30, 255],
    [180, 30, 80, 255],
    [255, 0, 0, 255],
    [30, 30, 255, 255],
    [200, 40, 255, 255],
    [90, 30, 150, 255],
    [255, 0, 255, 255],
    [255, 150, 255, 255],
    [75, 0, 75, 255],
    [75, 0, 175, 255],
    [0, 200, 255, 255],
    [50, 120, 255, 255],
    [0, 175, 0, 255],
    [0, 60, 135, 255],
    [80, 240, 150, 255],
    [150, 240, 255, 255],
    [0, 0, 255, 255],
])
cmap = cmap[:, [2, 1, 0]]  # convert bgra to rgba

def load_all_files_id(folder):
    results = os.listdir(folder)
    files_by_framid = {}
    files_by_humanid = {}
    for f in results:
        if not f.endswith('.pcd'):
            continue
        basename = f.split('.')[0]
        humanid = basename.split('_')[0]
        frameid = basename.split('_')[1]
        if frameid in files_by_framid:
            files_by_framid[frameid].append(humanid)
        else:
            files_by_framid[frameid] = [humanid]

        if humanid in files_by_humanid:
            files_by_humanid[humanid].append(frameid)
        else:
            files_by_humanid[humanid] = [frameid]

    files_by_humanid = dict(
        sorted(files_by_humanid.items(), key=lambda x: int(x[0])))
    files_by_framid = dict(
        sorted(files_by_framid.items(), key=lambda x: int(x[0])))

    for key, frame in files_by_humanid.items():
        files_by_humanid[key] = sorted(frame, key=lambda x: int(x))
    for key, frame in files_by_framid.items():
        files_by_framid[key] = sorted(frame, key=lambda x: int(x))

    return files_by_framid, files_by_humanid

def segment_ransac(pointcloud, return_seg = False):
    # pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.20, max_nn=20))
    pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    colors = np.asarray(pointcloud.colors)
    points = np.asarray(pointcloud.points)
    # normals = np.asarray(pointcloud.normals)


    rest_idx = set(np.arange(len(pointcloud.points)))
    # plane_idx = set()
    temp_idx = set()
    
    temp_cloud = o3d.geometry.PointCloud()

    for i in range(4):
        if i == 0:
            plane_model, inliers = pointcloud.segment_plane(distance_threshold=0.10, ransac_n=3, num_iterations=1200)
        elif len(temp_cloud.points) > 300:
            plane_model, inliers = temp_cloud.segment_plane(distance_threshold=0.10, ransac_n=3, num_iterations=1200)
        else:
            break
            
        # plane_inds += rest_idx[inliers]
        origin_inline = np.array(list(rest_idx))[inliers]
        colors[origin_inline] = plt.get_cmap('tab10')(i)[:3]
        rest_idx -= set(origin_inline)
        # plane_idx.union(set(origin_inline))

        if i == 0:
            temp_cloud = pointcloud.select_by_index(inliers, invert=True)
        else:
            temp_cloud = temp_cloud.select_by_index(inliers, invert=True)

        equation = plane_model[:3] ** 2
        if equation[2]/(equation[0] + equation[1]) < 130.6460956439: 
            # 如果平面与地面的夹角大于5°
            colors[origin_inline] = [1, 0, 0]
            temp_idx.union(set(origin_inline))

    if return_seg:
        non_ground_idx = np.array(list(rest_idx.union(temp_idx)))
        pointcloud.points = o3d.utility.Vector3dVector(points[non_ground_idx])
        pointcloud.colors = o3d.utility.Vector3dVector(colors[non_ground_idx])
        # pointcloud.normals = o3d.utility.Vector3dVector(normals[non_ground_idx])
    else:
        pointcloud.colors = o3d.utility.Vector3dVector(colors)

def read_frame_from_dets(frame_info):
    name = frame_info['name']
    score = frame_info['score']
    boxes_lidar = frame_info['boxes_lidar']
    frame_name = frame_info['frame_id']
    obj_id = frame_info['ids']
    # seq_id = frame_info['seq_id']

    name_m = name!='Car' #
    name = name[name_m]
    score = score[name_m]
    boxes_lidar = boxes_lidar[name_m][score>0.5]
    obj_id = obj_id[name_m][score>0.5]

    return obj_id, frame_name, boxes_lidar

def add_boxes_to_vis(boxes_lidar, vis, human_id, boxes_list):

    for i, box in enumerate(boxes_lidar):
        transform = R.from_rotvec(
            box[6] * np.array([0, 0, 1])).as_matrix()
        center = box[:3]
        extend = box[3:6]

        bbox = o3d.geometry.OrientedBoundingBox(center, transform, extend)

        bbox.color = cmap[int(human_id[i]) % len(cmap)] / 255
        boxes_list.append(bbox)
        vis.add_geometry(bbox, reset_bounding_box = False, waitKey=0)

def icp_mesh_and_point_cloud(mesh, point_cloud, reg_mesh = None, threshold = 0.2, vis = None):
    """_summary_

    Args:
        mesh (_type_): _description_
        point_cloud (_type_): _description_
    """
    smpl = o3d.geometry.PointCloud()
    smpl.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    # smpl = mesh.sample_points_poisson_disk(6890)
    geometries = []
    smpl = hidden_point_removal(smpl)
    init_transform = np.asarray([[1, 0, 0, 0], 
                                 [0, 1, 0, 0], 
                                 [0, 0, 1, 0], 
                                 [0, 0, 0, 1]])
    smpl = select_points_on_the_scan_line(np.asarray(smpl.points))
    # smpl.translate(np.asarray([0, -0.8, 0.]))
    if vis is not None:
        vis.add_geometry(smpl, reset_bounding_box = False)
    else:
        geometries.append(smpl)

    reg_p2l = o3d.pipelines.registration.registration_icp(
        smpl, point_cloud, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # print(reg_p2l)
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    # draw_registration_result(source, target, reg_p2l.transformation)
    if reg_mesh is None:
        reg_mesh = copy.deepcopy(mesh)
    else:
        reg_mesh.vertices = mesh.vertices
        reg_mesh.vertex_colors = mesh.vertex_colors
        reg_mesh.vertex_normals = mesh.vertex_normals

    reg_mesh.paint_uniform_color([1, 0.706, 0])
    reg_mesh.transform(reg_p2l.transformation)

    geometries.append(reg_mesh)

    return geometries

def display_by_human(load_data, file_path, skip = 0):
    """_summary_

    Args:
        load_data (_type_): _description_
        files_by_humanid (_type_): _description_
        file_path (_type_): _description_
        poses (_type_): _description_
        join (str, optional): _description_. Defaults to '\'.
        skip (int, optional): _description_. Defaults to 0.
    """
    join = '/' if load_data.remote else '\\'

    view = {
            "trajectory":
            [
                {
                    "boundingbox_max" : [ 5.1677, 1.3001, 0.163910],
                    "boundingbox_min" : [ 4.8066, 0.7946, -0.64518],
                    "field_of_view" : 60.0,
                    "front" : [ -0.9701, 0.1998, 0.1373 ],
                    "lookat" : [ 4.9432, 0.5004, -0.7150],
                    "up" : [ 0.15808, 0.09163, 0.9831 ],
                    "zoom" : 2.0
                }
            ]
    }
    vis = o3dvis(width=600, height=600)
    pointcloud = o3d.geometry.PointCloud()
    geometries = []
    reg_mesh = []
    _, files_by_humanid = load_all_files_id(file_path)
    mesh_path = file_path + "_predict_smpl"

    first_frame = False
    for humanid in files_by_humanid:
        frame_ids = files_by_humanid[humanid]
        vis.remove_geometries(geometries, reset_bounding_box = False)
        # vis.remove_geometries(reg_mesh, reset_bounding_box = False)
        for frame_id in frame_ids:

            cloud_file = join.join([file_path, f'{humanid}_{frame_id}.pcd'])
            pointcloud = load_data.load_point_cloud(cloud_file, pointcloud)
            
            transformation = make_cloud_in_vis_center(pointcloud)

            smpl = f'{humanid}_{frame_id}.ply'
            # smpl = join.join([frame_id, f'{humanid}.ply'])
            color = plt.get_cmap("tab20")(int(humanid) % 20)[:3]
            vis.add_mesh_together(mesh_path, [smpl], [color], geometries, [transformation])

            if not first_frame:
                vis.change_pause_status()
                # reg_mesh += icp_mesh_and_point_cloud(geometries[0], pointcloud)
                vis.add_geometry(pointcloud.translate(np.array([0, 1, 0])), reset_bounding_box = True)
                vis.set_view(view)
                # vis.add_geometry(reg_mesh[1], reset_bounding_box = False)
                first_frame = True
            else:
                # reg_mesh = icp_mesh_and_point_cloud(geometries[0], pointcloud, reg_mesh[1])
                # vis.vis.update_geometry(reg_mesh[1])
                vis.vis.update_geometry(pointcloud.translate(np.array([0, 1, 0])))
                

            # geometries += registered

            vis.waitKey(50, helps=False)
            
            vis.save_imgs(os.path.join(file_path, f'imgs'))
            
    # imges_to_video(os.path.join(file_path, f'imgs'), delete=True)

def display_by_box_frame(load_data, dets, poses, data_root_path = None, mesh_dir=None, skip = 0):
    """ 根据检测的bbox, 判断每帧的是否有对应的mesh, 有则显示
    Args:
        dets (dict): _description_
        poses (floats): (n, 4, 4) array
        join (str, optional): _description_. Defaults to '\\'.
        data_root_path (str, optional): _description_. Defaults to None.
        files_by_frameid (str, optional): _description_. Defaults to None.
        skip (int, optional): skip [skip] to vis. Defaults to 0.
    """
    join = '/' if load_data.remote else '\\'
    files_by_frameid, _ = load_all_files_id(mesh_dir)

    # 从dets中读取boundingbox
    boxes_list = []
    first_frame = True
    mesh_list = []

    vis = o3dvis()
    pointcloud = o3d.geometry.PointCloud()
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[
                                                                  0, 0, 0])
    vis.add_geometry(axis_pcd, reset_bounding_box = False)

    for frame_idx, frame_info in enumerate(dets):
        if frame_idx < 800 or data_root_path is None:
            continue
        transformation = poses[frame_idx]
        # transformation = np.eye(4)

        human_ids, frame_name, boxes_lidar = read_frame_from_dets(frame_info)
        # print(boxes_lidar.shape)
        
        pcd_file = join.join([data_root_path, 'human_semantic', frame_name+'.pcd'])
        pointcloud = load_data.load_point_cloud(pcd_file, pointcloud, position=transformation[:3, 3])

        # update axis
        vis.remove_geometry(axis_pcd, reset_bounding_box=False)
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[
            0, 0, 0]).transform(transformation)
        vis.add_geometry(axis_pcd, reset_bounding_box=False)

        # remove mesh_list and boxes_list 
        vis.remove_geometries(boxes_list, reset_bounding_box = False)
        vis.remove_geometries(mesh_list, reset_bounding_box = False)

        if f'{frame_idx:04d}' in files_by_frameid:
            fid = f'{frame_idx:04d}'
            # smpl = [join.join(['predict_smpl', f'{idx}_{id}.ply']) for id in files_by_frameid[idx]]
            smpl_list = [join.join([fid, f'{id}.ply']) for id in files_by_frameid[fid]]

            colors = [plt.get_cmap("tab20")(int(id) % 20)[:3] for id in human_ids]
            mesh_list += vis.add_mesh_together(mesh_dir, smpl_list, colors = colors)

        if len(pointcloud.points) > 0 and len(boxes_lidar) >0:

            add_boxes_to_vis(boxes_lidar, vis, human_ids, boxes_list)

            # update point cloud
            if first_frame:
                vis.add_geometry(pointcloud)
                first_frame = False
                vis.change_pause_status()
            else:
                vis.vis.update_geometry(pointcloud)

            vis.waitKey(1, helps=False)
        else:
            print(f'Skip frame {frame_idx}, {frame_name}')       

        vis.save_imgs(os.path.join(data_root_path, 'imgs'),
                      '{:04d}.jpg'.format(frame_idx))

if __name__ == '__main__':

    parser = configargparse.ArgumentParser()
    parser.add_argument("--remote", '-R', action='store_true')
    parser.add_argument("--tracking_file", '-B', type=str, default='C:\\Users\\DAI\\Desktop\\temp\\0417-03_tracking.pkl')
    parser.add_argument("--mesh_dir", '-M', type=str, default='New Folder')
    parser.add_argument("--type", '-T', type=int, default=2)
    args = parser.parse_args() 

    # load_boxes(dets, 'C:\\Users\\Yudi Dai\\Desktop\\segment\\velodyne')
    # load_boxes(dets, 'C:\\Users\\DAI\\Desktop\\temp\\velodyne')
    load_data = load_data_remote(args.remote)
    data_root_path = os.path.dirname(args.tracking_file)
    join = '/' if load_data.remote else '\\'

    # ============================================================================
    # 生成数据集的可视化，需要的数据包括
    # ============================================================================
    # 1. dets:      detection 的结果
    if os.path.exists(args.tracking_file):
        dets = load_data.load_pkl(args.tracking_file)
        
    # 2. poses:     slam的结果
    poses = load_data.read_poses(data_root_path)

    # 3. meshes:    lidarcap的结果
    if args.mesh_dir is None:
        mesh_dir = join.join([data_root_path, 'segment_by_tracking_03_rot'])
    else:
        mesh_dir = args.mesh_dir

    # 4. 摆正的每帧数据
    if args.type == 1:
        display_by_box_frame(load_data, dets, poses, data_root_path, mesh_dir, skip = 800)

    elif args.type == 2:
        display_by_human(load_data, mesh_dir, skip = 500)
