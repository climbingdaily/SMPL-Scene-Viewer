import numpy as np
import h5py
import configargparse
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util import o3dvis, load_data_remote, make_cloud_in_vis_center
from smpl import poses_to_vertices

view = {
	"trajectory" :
	[
		{
			"boundingbox_max" : [ 5.148459134744563, 0.18879798795550118, 0.79121494293212891 ],
			"boundingbox_min" : [ 4.534800590236415, -0.87921318347675903, -0.90700364112854004 ],
			"field_of_view" : 60.0,
			"front" : [ -0.99257767585799128, 0.091189825013992476, 0.080461004233516153 ],
			"lookat" : [ 4.9251547242143854, -0.40716893764948942, 0.15594117452305134 ],
			"up" : [ 0.086773297396398955, 0.067507214866913023, 0.99393821276770966 ],
			"zoom" : 0.71999999999999886
		}
	],
}

pt_color = plt.get_cmap("tab20")(1)[:3]
smpl_color = plt.get_cmap("tab20")(3)[:3]
gt_smpl_color = plt.get_cmap("tab20")(5)[:3]

def load_pkl_vis(file_path, start=0, end=-1, points='point_clouds', pose='pred_rotmats', remote=False):
    import pickle

    point_clouds = np.zeros((0, 512, 3))
    pred_vertices = np.zeros((0, 6890, 3))
    load_data_class = load_data_remote(remote)
    humans = load_data_class.load_pkl(file_path)
    # with open(file_path, "rb") as f:
    #     humans = pickle.load(f)
    for k,v in humans.items():
        
        pred_pose = v[pose]
        if end == -1:
            end = pred_pose.shape[0]
        pred_pose = pred_pose[start:end]
        point_clouds = np.concatenate((point_clouds, v[points][start:end]))
        pred_vertices = np.concatenate((pred_vertices, poses_to_vertices(pred_pose)))

    vis_pt_and_smpl(pred_vertices, point_clouds)


def load_hdf5_vis(file_path, start=0, end=-1, points='point_clouds', pred_rots='pred_rotmats', gt_pose='pose'):
    """
    > This function loads a hdf5 file, and visualizes the point clouds and the SMPL vertices
    
    :param file_path: the path to the hdf5 file
    :param start: the index of the first frame to visualize, defaults to 0 (optional)
    :param end: the last frame to visualize
    :param points: the name of the point cloud data in the hdf5 file, defaults to point_clouds
    (optional)
    :param pred_rots: the name of the hdf5 dataset that contains the predicted rotations, defaults to
    pred_rotmats (optional)
    :param gt_pose: the ground truth pose, defaults to pose (optional)
    """
    with h5py.File(file_path, mode='r') as f:
        # 'full_joints', 'lidar_to_mocap_RT', 'point_clouds', 'points_num', 'pose', 'rotmats', 'shape', 'trans'
        print(f.keys())
        if pred_rots in f:
            pred_pose = f[pred_rots]
            if end == -1:
                end = pred_pose.shape[0]
            pred_pose = pred_pose[start:end]
            pred_vertices = poses_to_vertices(pred_pose)
            try:
                gt_pose = f[gt_pose][start:end]
            except KeyError:
                gt_pose = f['gt_pose'][start:end]
            gt_vertices = poses_to_vertices(gt_pose)

            point_clouds = f[points][start:end]
            vis_pt_and_smpl(pred_vertices, point_clouds, gt_vertices)
        else:
            gt_pose = f[gt_pose]
            if end == -1:
                end = gt_pose.shape[0]

            gt_pose = gt_pose[start:end]
            gt_vertices = poses_to_vertices(gt_pose)
            point_clouds = f[points][start:end]
            vis_pt_and_smpl(gt_vertices, point_clouds, gt_vertices)

    
def vis_pt_and_smpl(pred_smpl, pc, gt_smpl= None):
    # assert v.shape[0] == pc.shape[0], "Groundtruth Data Shape are not compatible"
    vis = o3dvis(width=600, height=600)
    pointcloud = o3d.geometry.PointCloud()
    gt = o3d.io.read_triangle_mesh('.\\smpl\\sample.ply')  # a ramdon SMPL mesh
    pred = o3d.io.read_triangle_mesh('.\\smpl\\sample.ply')

    init_param = False
    centerz = 0
    for i in range(pred_smpl.shape[0]):

        # load data
        pointcloud.points = o3d.utility.Vector3dVector(pc[i])
        if gt_smpl is not None:
            gt.vertices = o3d.utility.Vector3dVector(gt_smpl[i])
        pred.vertices = o3d.utility.Vector3dVector(pred_smpl[i])
        
        # color
        pointcloud.paint_uniform_color(pt_color)
        if gt_smpl is not None:
            gt.paint_uniform_color(gt_smpl_color)
        pred.paint_uniform_color(smpl_color)

        # transform
        rt, center = make_cloud_in_vis_center(pointcloud) # 根据点的中心点，在XY平面将点云旋转平移
        rt[:3, 3] = np.array([5, -0.5, 0])
        if gt_smpl is not None:
            gt.transform(rt)
            gt.compute_vertex_normals()
        pred.transform(rt)

        pred.compute_vertex_normals()

        # add to visualization
        if not init_param:
            centerz = center[-1]
            vis.change_pause_status()
            vis.add_geometry(pointcloud, reset_bounding_box = True)    
            if gt_smpl is not None:
                vis.add_geometry(gt)    
            vis.add_geometry(pred)  
            vis.set_view(view)
            init_param = True

        else:
            vis.vis.update_geometry(pointcloud) 
            if gt_smpl is not None:
                vis.vis.update_geometry(gt)    
            # vis.vis.update_geometry(gt_2)    
            vis.vis.update_geometry(pred)  

        vis.waitKey(40, helps=False)
        
        # vis.save_imgs(os.path.join(file_path, f'imgs'))
            
    # imges_to_video(os.path.join(file_path, f'imgs'), delete=True)

if __name__ == '__main__':    
    parser = configargparse.ArgumentParser()
    parser.add_argument("--start", '-S', type=int, default=-2)
    parser.add_argument("--end", '-E', type=int, default=-2)
    parser.add_argument("--file_path", '-F', type=str, default=None)
                        
    args, opts = parser.parse_known_args()

    import config
    start = config.start if args.start == -2 else args.start
    end = config.end if args.end == -2 else args.end
    file_path = config.file_path if args.file_path is None else args.file_path

    if file_path.endswith('.pkl'):
        is_remote = True if '--remote' in opts else config.remote
        load_pkl_vis(file_path, start, end, remote=is_remote)
    else:
        load_hdf5_vis(file_path, start, end)