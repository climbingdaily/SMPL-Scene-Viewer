import numpy as np
import h5py
import configargparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from o3dvis import o3dvis
import matplotlib.pyplot as plt
import torch
from smpl.smpl import SMPL
from vis_3d_box import load_data_remote
from copy import deepcopy

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

pt_color = plt.get_cmap("tab20")(1)[:3]
smpl_color = plt.get_cmap("tab20")(3)[:3]
gt_smpl_color = plt.get_cmap("tab20")(5)[:3]
pred_smpl_color = plt.get_cmap("tab20")(7)[:3]


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


def vertices_to_head(vertices, index = 15):
    smpl = SMPL()
    return smpl.get_full_joints(torch.FloatTensor(vertices))[..., index, :]

def poses_to_vertices(poses, trans=None, batch_size = 1024):
    poses = poses.astype(np.float32)
    vertices = np.zeros((0, 6890, 3))

    n = len(poses)
    smpl = SMPL()
    n_batch = (n + batch_size - 1) // batch_size

    for i in range(n_batch):
        lb = i * batch_size
        ub = (i + 1) * batch_size

        cur_n = min(ub - lb, n - lb)
        cur_vertices = smpl(torch.from_numpy(
            poses[lb:ub]), torch.zeros((cur_n, 10)))
        vertices = np.concatenate((vertices, cur_vertices.cpu().numpy()))

    if trans is not None:
        trans = trans.astype(np.float32)
        vertices += np.expand_dims(trans, 1)
    return vertices

def load_pred_smpl(file_path, start=0, end=-1, pose='pred_rotmats', trans=None, remote=False):
    import pickle

    pred_vertices = np.zeros((0, 6890, 3))
    load_data_class = load_data_remote(remote)
    humans = load_data_class.load_pkl(file_path)

    for k,v in humans.items():    
        pred_pose = v[pose]
        if end == -1:
            end = pred_pose.shape[0]
        pred_pose = pred_pose[start:end]
        pred_vertices = np.concatenate((pred_vertices, poses_to_vertices(pred_pose, trans)))

    return pred_vertices

def get_head_global_rots(pose):
    if pose.shape[1] == 72:
        pose = pose.reshape(-1, 24, 3)

    head_parents = [0, 3, 6, 9, 12, 15]
    head_rots = np.eye(3)
    for r in head_parents[::-1]:
        head_rots = R.from_rotvec(pose[:, r]).as_matrix() @ head_rots
    head_rots = head_rots @ np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]).T
    return head_rots

def load_pkl_vis(humans, start=0, end=-1, pred_file_path=None, remote=False):
    """
    It loads the pickle file, converts the poses to vertices, and then visualizes the vertices and point
    clouds
    
    :param file_path: the path to the pickle file
    :param start: the first frame to visualize, defaults to 0 (optional)
    :param end: the last frame to be visualized
    :param remote: whether to load the data from a remote server, defaults to False (optional)
    """

    first_person = humans['first_person']
    pose = first_person['pose'].copy()
    trans = first_person['mocap_trans'].copy()
    f_vert = poses_to_vertices(pose)
    if 'lidar_traj' in first_person:
        lidar_traj = first_person['lidar_traj'][:, 1:4]
        head = vertices_to_head(f_vert, 15)
        root = vertices_to_head(f_vert, 0)
        head_to_root = (root - head).numpy()
        
        head_rots = get_head_global_rots(pose)

        lidar_to_head =  head[0] - lidar_traj[0] + trans[0] 
        lidar_to_head = head_rots @ lidar_to_head.numpy()

        trans = lidar_traj + lidar_to_head + head_to_root
        # trans = lidar_traj[:, 1:4]
    
    f_vert += np.expand_dims(trans.astype(np.float32), 1)

    second_person = humans['second_person']    
    pose = second_person['pose'].copy()
    trans = second_person['mocap_trans'].copy()
    s_vert = poses_to_vertices(pose, trans)

    if 'point_clouds' in second_person:
        point_clouds = second_person['point_clouds']
        ll = second_person['point_frame']
    else:
        point_clouds = np.array([[0,0,0]])
        ll = []

    point_valid_idx = [np.where(humans['frame_num'] == l)[0][0] for l in ll ]

    if pred_file_path is not None :
        pred_s_vert = load_pred_smpl(pred_file_path, trans=trans[point_valid_idx], remote=remote)
    else:
        pred_s_vert = None

    return f_vert, s_vert, pred_s_vert, point_clouds, point_valid_idx

def vis_pt_and_smpl(smpl_list, pc, pc_idx, vis, pred_smpl_verts=None, view_list=None):
    """
    > This function takes in two SMPL meshes, a point cloud, and a list of indices that correspond to
    the point cloud. It then displays the point cloud and the two SMPL meshes in a 3D viewer
    
    :param smpl_a: the SMPL mesh that you want to visualize
    :param smpl_b: the ground truth SMPL mesh
    :param pc: the point cloud data
    :param pc_idx: the index of the point cloud that you want to visualize
    """
    # assert smpl_list[0].shape[0] == smpl_list[1].shape[0], "Groundtruth Data Shape are not compatible"
    pointcloud = o3d.geometry.PointCloud()
    smpl_geometries = []
    pred_smpl = o3d.io.read_triangle_mesh('.\\smpl\\sample.ply')
    for i in smpl_list:
        smpl_geometries.append(o3d.io.read_triangle_mesh('.\\smpl\\sample.ply')) # a ramdon SMPL mesh

    init_param = False

    for i in range(smpl_list[0].shape[0]):

        # load data
        if i in pc_idx:
            pointcloud.points = o3d.utility.Vector3dVector(pc[pc_idx.index(i)])
            if pred_smpl_verts is not None:
                pred_smpl.vertices = o3d.utility.Vector3dVector(pred_smpl_verts[pc_idx.index(i)])
                pred_smpl.paint_uniform_color(plt.get_cmap("tab20")(7)[:3])
                pred_smpl.compute_vertex_normals()
        else:
            pointcloud.points = o3d.utility.Vector3dVector(np.array([[0,0,0]]))
            pred_smpl.vertices = o3d.utility.Vector3dVector(np.asarray(pred_smpl.vertices) * 0)
            pred_smpl.compute_vertex_normals()

        # color
        pointcloud.paint_uniform_color(pt_color)

        for idx, smpl in enumerate(smpl_geometries):
            smpl.vertices = o3d.utility.Vector3dVector(smpl_list[idx][i])
            smpl.compute_vertex_normals()
            smpl.paint_uniform_color(plt.get_cmap("tab20")(idx*2 + 3)[:3])

        if view_list is not None:
            vis.set_view(view_list[i])

        # add to visualization
        if not init_param:
            vis.change_pause_status()
            vis.add_geometry(pointcloud, reset_bounding_box = False)  
            for smpl in smpl_geometries:
                vis.add_geometry(smpl, reset_bounding_box = False)  
            if pred_smpl is not None:
                vis.add_geometry(pred_smpl, reset_bounding_box = False)  
            init_param = True

        else:
            vis.vis.update_geometry(pointcloud) 
            if pred_smpl is not None:
                vis.vis.update_geometry(pred_smpl)  
            for smpl in smpl_geometries:
                vis.vis.update_geometry(smpl)    
        vis.waitKey(30, helps=False)
        
    # vis.save_imgs(os.path.join(file_path, f'imgs'))
            
    # imges_to_video(os.path.join(file_path, f'imgs'), delete=True)

def generate_views(position, direction, filter=True, rad=np.deg2rad(15)):
    assert len(position) == len(direction)

    mocap_init = np.array([[-1, 0, 0, ], [0, 0, 1], [0, 1, 0]])
    base_view = {
        "trajectory" : 
        [
            {
                "field_of_view" : 90.0,
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

    init_direction = func(direction[0]).as_matrix()

    if filter:
       position = filterTraj(position)

    view_list = []
    for t, r in zip(position, direction):
        view = deepcopy(base_view)
        # rot = func(r).as_euler('xyz', degrees=False)
        rot = func(r).as_matrix() @ init_direction.T

        view['trajectory'][0]['lookat'] = t.tolist()
        view['trajectory'][0]['up'] = rot @ np.array(view['trajectory'][0]['up'])
        view['trajectory'][0]['front'] = rot @ np.array(view['trajectory'][0]['front'])
        view_list.append(view)
    
    return view_list

def load_scene(vis, pcd_path=None, scene = None):
    from time import time
    reading_class = load_data_remote(remote=True)
    if pcd_path is not None:
        t1 = time()
        print(f'Loading scene from {pcd_path}')
        scene = reading_class.load_point_cloud(pcd_path)
        t2 = time()
        print(f'====> Scene loading comsumed {t2-t1:.1f} s.')
    vis.set_view(view)
    vis.add_geometry(scene)
    return scene

if __name__ == '__main__':    
    parser = configargparse.ArgumentParser()
    parser.add_argument("--type", '-T', type=int, default=3)
    parser.add_argument("--start", '-S', type=int, default=0)
    parser.add_argument("--end", '-e', type=int, default=-1)
    parser.add_argument("--scene", '-s', type=str,
                        default='/hdd/dyd/lidarhumanscene/data/0623/002/0623.pcd')
    parser.add_argument("--remote", '-r', type=bool, default=True)
    parser.add_argument("--file_path", '-F', type=str,
                        default='/hdd/dyd/lidarhumanscene/data/0623/002/synced_data/two_person_param.pkl')
    parser.add_argument("--pred_file_path", '-P', type=str,
                        default=None)
                        # default='/hdd/dyd/lidarhumanscene/data/0604_haiyun/synced_data/second_person/segments.pkl')
                        
    args, opts = parser.parse_known_args()

    fvis = o3dvis("First view", width=1280, height=720)
    # svis = o3dvis("Second view", width=1280, height=720)

    _ = load_scene(fvis, args.scene)
    # load_scene(svis, scene=scene)

    print(f'Load pkl in {args.file_path}')
    load_data_class = load_data_remote(args.remote)
    humans = load_data_class.load_pkl(args.file_path)

    smpl_a, smpl_b, pred_smpl_b, pc, pc_idx = load_pkl_vis(
        humans, args.start, args.end, args.pred_file_path, remote=args.remote)

    # lidar_view = generate_views(humans['first_person']['lidar_traj']
    #                      [:, 1:4], humans['first_person']['lidar_traj'][:, 4:8])

    FPV = generate_views(humans['first_person']['lidar_traj']
                         [:, 1:4], get_head_global_rots(humans['first_person']['pose']))

    SPV = generate_views(vertices_to_head(
        smpl_b) + np.array([0, 0, 0.2]), get_head_global_rots(humans['second_person']['pose']))

    vis_pt_and_smpl([smpl_a, smpl_b], pc, pc_idx, fvis, pred_smpl_verts = pred_smpl_b, view_list = FPV)
    vis_pt_and_smpl([smpl_a, smpl_b], pc, pc_idx, fvis, pred_smpl_verts = pred_smpl_b, view_list = SPV)