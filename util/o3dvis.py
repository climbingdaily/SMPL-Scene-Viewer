import open3d as o3d
import numpy as np
import cv2
import sys
import os
import paramiko
from util import pypcd
import matplotlib.pyplot as plt
# from util.segmentation import Segmentation
from matplotlib.animation import FuncAnimation, writers

def client_server(username = 'dyd', hostname = "10.24.80.241", port = 911):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port, username, compress=True)
    return client

def list_dir_remote(client, folder):
    stdin, stdout, stderr = client.exec_command('ls ' + folder)
    res_list = stdout.readlines()
    return [i.strip() for i in res_list]

def read_pcd_from_server(client, filepath, sftp_client = None):
    if sftp_client is None:
        sftp_client = client.open_sftp()
    remote_file = sftp_client.open(filepath, mode='rb')  # 文件路径

    try:
        pc_pcd = pypcd.PointCloud.from_fileobj(remote_file)
        pc = np.zeros((pc_pcd.pc_data.shape[0], 3))
        pc[:, 0] = pc_pcd.pc_data['x']
        pc[:, 1] = pc_pcd.pc_data['y']
        pc[:, 2] = pc_pcd.pc_data['z']
        if 'rgb' in pc_pcd.fields:
            append = pypcd.decode_rgb_from_pcl(pc_pcd.pc_data['rgb'])/255
            pc = np.concatenate((pc, append), axis=1)
        if 'normal_x' in pc_pcd.fields:        
            append = pc_pcd.pc_data['normal_x'].reshape(-1, 1)
            pc = np.concatenate((pc, append), axis=1)
        if 'normal_y' in pc_pcd.fields:        
            append = pc_pcd.pc_data['normal_y'].reshape(-1, 1)
            pc = np.concatenate((pc, append), axis=1)
        if 'normal_z' in pc_pcd.fields:        
            append = pc_pcd.pc_data['normal_z'].reshape(-1, 1)
            pc = np.concatenate((pc, append), axis=1)
        if 'intensity' in pc_pcd.fields:        
            append = pc_pcd.pc_data['intensity'].reshape(-1, 1)
            pc = np.concatenate((pc, append), axis=1)
        
        return np.concatenate((pc, append), axis=1)
    except Exception as e:
        print(f"Load point cloud {filepath} error")
    finally:
        remote_file.close()

        
colors = {
    'yellow':[251/255, 217/255, 2/255],
    'red'   :[234/255, 101/255, 144/255],
    'blue' :[27/255, 158/255, 227/255],
    'purple':[61/255, 79/255, 222/255],
    'blue2' :[75/255, 145/255, 183/255],
}

class Keyword():
    PAUSE = False       # pause the visualization
    DESTROY = False     # destory window
    REMOVE = False      # remove all geometies
    READ = False        # read the ply files
    VIS_TRAJ = False    # visualize the trajectory
    SAVE_IMG = False    # save the images in open3d window
    SET_VIEW = False    # set the view based on the info   
    VIS_STREAM = True   # only visualize the the latest mesh stream
    ROTATE = False      # rotate the view automatically
    PRESS_YES = False   #
    PRESS_NO = False   #

lidar_cap_view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : 'false',
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 74.593666076660156, 43.178085327148438, 11.551046371459961 ],
			"boundingbox_min" : [ -64.269279479980469, -34.139186859130859, -4.4510049819946289 ],
			"field_of_view" : 59.999999999999993,
			"front" : [ 0.78640128083841598, -0.28397063347164114, 0.54857424731882343 ],
			"lookat" : [ -0.84304582914768755, -1.7827584067306674, 0.088647440399997293 ],
			"up" : [ -0.50817483197168944, 0.20747905554223359, 0.83588921614161771 ],
			"zoom" : 0.080000000000000002
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}


CAMERA = {
    'phi': 0,
    'theta': -30,
    'cx': 0.,
    'cy': 0.5,
    'cz': 3.}

def set_camera(camera_pose):
    theta, phi = np.deg2rad(-(CAMERA['theta'] + 90)), np.deg2rad(CAMERA['phi'] + 180)
    theta = theta + np.pi
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    rot_x = np.array([
        [1., 0., 0.],
        [0., ct, -st],
        [0, st, ct]
    ])
    rot_z = np.array([
        [cp, -sp, 0],
        [sp, cp, 0.],
        [0., 0., 1.]
    ])
    camera_pose[:3, :3] = rot_x @ rot_z
    return camera_pose


def set_view(vis):
    Keyword.SET_VIEW = not Keyword.SET_VIEW
    print('SET_VIEW', Keyword.SET_VIEW)
    return False

def press_yes(vis):
    Keyword.PRESS_YES = not Keyword.PRESS_YES
    print(f'\r[PRESS_YES]: {Keyword.PRESS_YES} ', end='', flush=True)
    return False
    
def press_no(vis):
    Keyword.PRESS_NO = not Keyword.PRESS_NO
    print(f'\r[PRESS_NO]: {Keyword.PRESS_NO} ', end='', flush=True)
    return False

def save_imgs(vis):
    Keyword.SAVE_IMG = not Keyword.SAVE_IMG
    print('SAVE_IMG', Keyword.SAVE_IMG)
    return False

def stream_callback(vis):
    # 以视频流方式，更新式显示mesh
    Keyword.VIS_STREAM = not Keyword.VIS_STREAM
    print(f'\r[VIS_STREAM]: {Keyword.VIS_STREAM} ', end='', flush=True)
    return False

def pause_callback(vis):
    Keyword.PAUSE = not Keyword.PAUSE
    print(f'\r[Pause]: {Keyword.PAUSE} ', end='\t', flush=True)
    return False

def destroy_callback(vis):
    Keyword.DESTROY = not Keyword.DESTROY
    return False

def remove_scene_geometry(vis):
    Keyword.REMOVE = not Keyword.REMOVE
    return False

def read_dir_ply(vis):
    Keyword.READ = not Keyword.READ
    print('READ', Keyword.READ)
    return False

def read_dir_traj(vis):
    Keyword.VIS_TRAJ = not Keyword.VIS_TRAJ
    print('VIS_TRAJ', Keyword.VIS_TRAJ)
    return False

def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False

def load_render_option(vis):
    vis.get_render_option().load_from_json(
        "../../test_data/renderoption.json")
    return False

def capture_depth(vis):
    depth = vis.capture_depth_float_buffer()
    plt.imshow(np.asarray(depth))
    plt.show()
    return False

def capture_image(vis):
    image = vis.capture_screen_float_buffer()
    plt.imshow(np.asarray(image))
    plt.show()
    return False

def o3d_callback_rotate(vis):
    Keyword.ROTATE = not Keyword.ROTATE
    return False

def print_help(is_print=True):
    if is_print:
        print('============Help info============')
        print('Press R to refresh visulization')
        print('Press Q to quit window')
        print('Press D to remove the scene')
        print('Press T to load and show traj file')
        print('Press F to stop current motion')
        print('Press . to turn on auto-screenshot ')
        print('Press , to set view zoom based on json file ')
        print('Press SPACE to pause the stream')
        print('=================================')


class o3dvis():
    def __init__(self, window_name = 'DAI_VIS', width=1280, height=720):
        self.init_vis(window_name, width, height)
        self.video_writer = None
        self.width = width
        self.height = height
        self.img_save_count = 0
        print_help()

    def change_pause_status(self):
        pause_callback(self.vis)

    def add_scene_gemony(self, geometry):
        if not Keyword.REMOVE:
            self.add_geometry(geometry)

    def init_vis(self, window_name, width, height):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(ord(" "), pause_callback)
        self.vis.register_key_callback(ord("Q"), destroy_callback)
        self.vis.register_key_callback(ord("D"), remove_scene_geometry)
        self.vis.register_key_callback(ord("R"), o3d_callback_rotate)
        self.vis.register_key_callback(ord("T"), read_dir_traj)
        self.vis.register_key_callback(ord("F"), stream_callback)
        self.vis.register_key_callback(ord("."), save_imgs)
        self.vis.register_key_callback(ord(","), set_view)
        self.vis.register_key_callback(ord("N"), press_no)
        self.vis.register_key_callback(ord("Y"), press_yes)
        self.vis.create_window(window_name=window_name, width=width, height=height)

    def get_camera(self):
        ctr = self.vis.get_view_control()
        init_param = ctr.convert_to_pinhole_camera_parameters()
        return np.array(init_param.extrinsic)

    def rotate(self):
        if Keyword.ROTATE:
            CAMERA['phi'] += np.pi/10
            camera_pose = set_camera(self.get_camera())
            self.init_camera(camera_pose)

    def init_camera(self, camera_pose):
        ctr = self.vis.get_view_control()
        init_param = ctr.convert_to_pinhole_camera_parameters()
        # init_param.intrinsic.set_intrinsics(init_param.intrinsic.width, init_param.intrinsic.height, fx, fy, cx, cy)
        init_param.extrinsic = np.array(camera_pose)
        ctr.convert_from_pinhole_camera_parameters(init_param) 

    def waitKey(self, key, helps = True):
        print_help(helps)
        while True:
            self.rotate()
            self.vis.poll_events()
            self.vis.update_renderer()
            cv2.waitKey(key)
            if Keyword.DESTROY:
                self.vis.destroy_window()
            if not Keyword.PAUSE:
                break
        return Keyword.READ

    def add_geometry(self, geometry, reset_bounding_box = True, waitKey = 10):
        self.vis.add_geometry(geometry, reset_bounding_box)
        if waitKey > 0:
            self.waitKey(waitKey, helps=False)

    def remove_geometry(self, geometry, reset_bounding_box = True):
        self.vis.remove_geometry(geometry, reset_bounding_box)

    def remove_geometries(self, geometries, reset_bounding_box = True):
        for geometry in geometries:
            self.remove_geometry(geometry, reset_bounding_box)
        geometries.clear()
        

    def set_view_zoom(self, info, count, steps):
        """
        It takes a dictionary of parameters, and sets the view of the vis object to the values in the
        dictionary. 
        
        The dictionary can be either a single view, or a list of views. 
        
        If it's a list of views, then the function will interpolate between the views. 
        
        The interpolation is done by the count parameter, which is the number of frames that have been
        rendered so far. 
        
        The steps parameter is the number of frames that will be rendered in total. 
        
        The function returns False, which means that the rendering will continue. 
        
        If it returned True, then the rendering would stop. 
        
        The function is called by the render_animation function, which is defined below.
        
        :param info: 
        :param count: the current frame number
        :param steps: the number of frames in the animation
        """
        
        ctr = self.vis.get_view_control()
        elements = ['zoom', 'lookat', 'up', 'front', 'field_of_view']
        # if 'step1' in info.keys():
        #     steps = info['step1']
        if 'views' in info.keys() and 'steps' in info.keys():
            views = info['views']
            fit_steps = info['steps']
            count += info['start']
            for i, v in enumerate(views):
                if i == len(views) - 1:
                    continue
                if count >= fit_steps[i+1]:
                    continue
                for e in elements:
                    z1 = np.array(views[i]['trajectory'][0][e])
                    z2 = np.array(views[i+1]['trajectory'][0][e])
                    if e in elements:
                        value = z1 + (count - fit_steps[i])  * (z2-z1) / (fit_steps[i+1] - fit_steps[i])
                    if e == 'zoom':
                        ctr.set_zoom(value)
                    elif e == 'lookat':
                        ctr.set_lookat(value)
                    elif e == 'up':
                        ctr.set_up(value)
                    elif e == 'front':
                        ctr.set_front(value)
                break    

        elif 'trajectory' in info.keys():
            self.vis.reset_view_point(True)
            ctr.set_zoom(np.array(info['trajectory'][0]['zoom']))
            ctr.set_lookat(np.array(info['trajectory'][0]['lookat']))
            ctr.set_up(np.array(info['trajectory'][0]['up']))
            ctr.set_front(np.array(info['trajectory'][0]['front']))

        return False
    
    def add_mesh_together(self, plydir, mesh_list, colors = None, geometies = None, transformation=None):
        """_summary_

        Args:
            plydir (_type_): _description_
            mesh_list (_type_): _description_
            color (_type_): _description_
        """
        if geometies is None:
            geometies = []
        if transformation is None:
            transformation = []
        
        for idx, mesh_file in enumerate(mesh_list):
            plyfile = os.path.join(plydir, mesh_file)
            # print(plyfile)
            mesh = o3d.io.read_triangle_mesh(plyfile)
            mesh.compute_vertex_normals()
            if colors is not None:
                mesh.paint_uniform_color(colors[idx])
            else:
                mesh.paint_uniform_color(plt.get_cmap("tab20")(int(mesh_file.split('.')[0]))[:3])
            
            # if not mesh.has_triangle_uvs():
            #     uv = np.array([[0.0, 0.0]] * (3 * len(mesh.triangles)))
            #     mesh.triangle_uvs = o3d.utility.Vector2dVector(uv)

            # mesh.vertices = Vector3dVector(np.array(mesh.vertices) - trajs[num[i],1:4] + mocap_trajs[num[i],1:4])
            if len(transformation) > idx:
                mesh.transform(transformation[idx])
            if len(geometies) > idx:
                geometies[idx].vertices = mesh.vertices
                geometies[idx].vertex_colors = mesh.vertex_colors
                geometies[idx].vertex_normals = mesh.vertex_normals
                self.vis.update_geometry(geometies[idx])
            else:
                geometies.append(mesh)
                self.add_geometry(mesh, reset_bounding_box = False, waitKey=0)
        return geometies

    def add_mesh_by_order(self, plydir, mesh_list, color, strs='render', order = True, start=None, end=None, info=None):
        """[summary]

        Args:
            plydir ([str]): directory name of the files
            mesh_list ([list]): file name list
            color ([str]): [red, yellow, green, blue]
            strs (str, optional): [description]. Defaults to 'render'.
            order (bool, optional): [description]. Defaults to True.
            start ([int], optional): [description]. Defaults to None.
            end ([int], optional): [description]. Defaults to None.
            info ([type], optional): [description]. Defaults to None.
        Returns:
            [list]: [A list of geometries]
        """        
        save_dir = os.path.join(plydir, strs)
        
        if order:
            num = np.array([int(m.split('_')[0]) for m in mesh_list], dtype=np.int32)
            idxs = np.argsort(num)
        else:
            idxs = np.arange(len(mesh_list))
        pre_mesh = None
        
        geometies = []
        helps = True
        count = 0

        # trajs = np.loadtxt('G:\\Human_motion\\visualization\\trajs\\campus_lidar_filt_synced_offset.txt')
        # mocap_trajs = np.loadtxt('G:\\Human_motion\\visualization\\trajs\\mocap_trans_synced.txt')

        sphere_list = []
        for i in idxs:
            # set view zoom
            if info is not None and Keyword.SET_VIEW:
                self.set_view_zoom(info, count, end-start)
            if order and end > start:
                if num[i] < start or num[i] > end:
                    continue

            plyfile = os.path.join(plydir, mesh_list[i])
            # print(plyfile)
            mesh = o3d.io.read_triangle_mesh(plyfile)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(colors[color])
            # mesh.vertices = Vector3dVector(np.array(mesh.vertices) - trajs[num[i],1:4] + mocap_trajs[num[i],1:4])
            if Keyword.VIS_STREAM and pre_mesh is not None:
                self.remove_geometry(pre_mesh, reset_bounding_box = False)
                geometies.pop()
            Keyword.VIS_STREAM = True # 
            geometies.append(mesh)
            self.add_geometry(mesh, reset_bounding_box = False)
                
            # if count % 5 == 0:
            #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            #     sphere.vertices = Vector3dVector(np.asarray(sphere.vertices) + trajs[num[i],1:4])
            #     sphere.compute_vertex_normals()
            #     sphere.paint_uniform_color(color)
            #     sphere_list.append(sphere)
            #     self.add_geometry(sphere, reset_bounding_box = False)

            
            pre_mesh = mesh
            if not self.waitKey(10, helps=helps):
                break
            helps = False
            self.save_imgs(save_dir, strs + '_{:04d}.jpg'.format(count)) 
            count += 1

        for s in sphere_list:
            self.remove_geometry(s, reset_bounding_box = False)

        return geometies
    
    def visualize_traj(self, plydir, sphere_list):
        """[读取轨迹文件]

        Args:
            plydir ([str]): [description]
            sphere_list ([list]): [description]
        """        
        if not Keyword.VIS_TRAJ:
            return sphere_list

        for sphere in sphere_list:
            self.remove_geometry(sphere, reset_bounding_box = False)
        sphere_list.clear()
        traj_files = os.listdir(plydir)

        # 读取文件夹内所有的轨迹文件
        for trajfile in traj_files:
            if trajfile.split('.')[-1] != 'txt':
                continue
            print('name', trajfile)
            if trajfile.split('_')[-1] == 'offset.txt':
                color = 'red'
            elif trajfile.split('_')[-1] == 'synced.txt':
                color = 'yellow'
            else:
                color = 'blue'
            trajfile = os.path.join(plydir, trajfile)
            trajs = np.loadtxt(trajfile)[:,1:4]
            traj_cloud = o3d.geometry.PointCloud()
            # show as points
            traj_cloud.points = o3d.utility.Vector3dVector(trajs)
            traj_cloud.paint_uniform_color(color)
            sphere_list.append(traj_cloud)
            # for t in range(1400, 2100, 1):
            #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            #     sphere.vertices = Vector3dVector(np.asarray(sphere.vertices) + trajs[t])
            #     sphere.compute_vertex_normals()
            #     sphere.paint_uniform_color(color)
            #     sphere_list.append(sphere)

        # 轨迹可视化
        for sphere in sphere_list:
            self.add_geometry(sphere, reset_bounding_box = False)

        Keyword.VIS_TRAJ = False
        return sphere_list

    def set_view(self, view=lidar_cap_view):
        ctr = self.vis.get_view_control()
        if view is not None:
            # self.vis.reset_view_point(True)
            ctr.set_zoom(np.array(view['trajectory'][0]['zoom']))
            ctr.set_lookat(np.array(view['trajectory'][0]['lookat']))
            ctr.set_up(np.array(view['trajectory'][0]['up']))
            ctr.set_front(np.array(view['trajectory'][0]['front']))
            return True
        return False

    def return_press_state(self):
        while True:
            if Keyword.PRESS_YES:
                Keyword.PRESS_YES = False
                return True
            if Keyword.PRESS_NO:
                Keyword.PRESS_NO = False
                return False
            self.waitKey(10, helps=False)

    def save_imgs(self, out_dir, filename=None):
        """[summary]

        Args:
            out_dir ([str]): [description]
            filename ([str]): [description]
        """        
            
        if Keyword.SAVE_IMG:
            # outname = os.path.join(out_dir, filename)
            outname = os.path.join(out_dir, f'{self.img_save_count:04d}.jpg')
            self.img_save_count += 1
            # outname = os.path.join(out_dir, filename).replace('.jpg', '.mp4')
            # outname = os.path.join(out_dir, filename).replace('.mp4', '.avi')
            # img = np.asarray(self.vis.capture_screen_float_buffer())
            # if self.video_writer is None:
            #     fourcc = cv2.VideoWriter_fourcc(*"DIVX")
            #     self.video_writer = cv2.VideoWriter(outname, fourcc, 15.0, (img.shape[1], img.shape[0]), True)
            os.makedirs(out_dir, exist_ok=True)
            self.vis.capture_screen_image(outname)
            # r = img[..., 0:1]
            # g = img[..., 1:2]
            # b = img[..., 2:3]
            # new_img = np.concatenate((b, g, r), axis = 2)
            # # cv2.imshow('test', new_img)
            # self.video_writer.write(new_img)

    def visulize_point_clouds(self, file_path, skip = 150, view = None, remote = False):
        """visulize the point clouds stream

        Args:
            file_path (str): [description]
            skip (int, optional): Defaults to 150.
            view (dict): A open3d format viewpoint, 
                         you can get one view by using 'ctrl+c' in the visulization window. 
                         Default None.
        """ 
        if remote:
            client = client_server()
            client_client = client.open_sftp()
            files = sorted(list_dir_remote(client, file_path))
        else:
            files = sorted(os.listdir(file_path))

        pointcloud = o3d.geometry.PointCloud()
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])

        self.add_geometry(axis_pcd)
        self.add_geometry(pointcloud)

        Reset = True

        mesh_list = []

        for i, file_name in enumerate(files):
            print(f'Processing {file_name}')
            if i < skip:
                continue

            for mesh in mesh_list:
                self.remove_geometry(mesh, reset_bounding_box = False)
            mesh_list.clear()

            if file_name.endswith('.txt'):
                pts = np.loadtxt(os.path.join(file_path, file_name))
                pointcloud.points = o3d.utility.Vector3dVector(pts[:, :3])  
            elif file_name.endswith('.pcd') or file_name.endswith('.ply'):
                if remote:
                    pcd = read_pcd_from_server(client, file_path + '/' + file_name, client_client)
                    pointcloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
                    if pcd.shape[1] == 6:
                        pointcloud.colors = o3d.utility.Vector3dVector(pcd[:, 3:])  
                else:
                    pcd = o3d.io.read_point_cloud(os.path.join(file_path, file_name))
                    pointcloud.points = pcd.points
                    # print(len(pcd.poits))
                    pointcloud.colors = pcd.colors
                    
                    # ! Temp code, for visualization test
                    # mesh_dir = os.path.join(os.path.join(os.path.dirname(
                    #     file_path), 'instance_human'), file_name.split('.')[0])
                    # if os.path.exists(mesh_dir):
                    #     mesh_list += self.add_mesh_together(
                    #         mesh_dir, os.listdir(mesh_dir), 'blue')
            else:
                continue
            
            self.vis.update_geometry(pointcloud)

            # ! Segment plane
            # pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.20, max_nn=20))
            # pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            # for i in range(6):
            #     _, inliers = pointcloud.segment_plane(distance_threshold=0.12, ransac_n=3, num_iterations=1200)
                
            #     temp_cloud = pointcloud.select_by_index(inliers, invert=True)
            #     pointcloud.points = temp_cloud.points
            #     pointcloud.colors = temp_cloud.colors
            #     pointcloud.normals= temp_cloud.normals

            # ! if cluster the point cloud and visualize it
            # labels = np.array(pointcloud.cluster_dbscan(eps=0.4, min_points=20))
            # max_label = labels.max()
            # # for i in range(max_label):
            # #     list[np.where(labels == i)[0]]
            # print(f"point cloud has {max_label + 1} clusters")

            # colors = plt.get_cmap("tab20")(
            #     labels / (max_label if max_label > 0 else 1))
            # colors[labels < 0] = 0
            # pointcloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

            self.vis.update_geometry(pointcloud)

            if Reset:
                Reset = self.set_view_zoom(view, 0, 0)

            self.waitKey(10, helps=False)
            self.save_imgs(os.path.join(file_path, 'imgs'),
                           '{:04d}.jpg'.format(i-skip))
        if remote:
            client.close()
        while True:
            self.waitKey(10, helps=False)
        
