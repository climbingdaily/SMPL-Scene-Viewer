################################################################################
# File: \vis_test.py                                                           #
# Created Date: Friday August 12th 2022                                        #
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
import open3d as o3d
import open3d.visualization.gui as gui
import sys
import threading
import os
import matplotlib.pyplot as plt

sys.path.append('.')

from gui_vis import HUMAN_DATA, Setting_panal as setting, Menu, creat_chessboard, add_box, mat_set, add_btn, vertices_to_joints
from util import load_scene as load_pts, read_json_file, cam_to_extrinsic, extrinsic_to_cam

sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'smpl', 'sample.ply')
data_format = read_json_file(os.path.join(os.path.dirname(__file__), 'smpl_key.json'))

POSE_COLOR = {}
for i, key in enumerate(data_format['color_order']):
    if len(data_format['color_order'][key]) > 0:
        POSE_COLOR[key] = np.array(data_format['color_order'][key])/255
    else:
        POSE_COLOR[key] = plt.get_cmap(data_format['color_map'])(i*2 + 1)[:3]

def points_to_sphere(geometry):
    """
    > It takes a point cloud and returns a mesh of spheres, each sphere centered at a point in the point
    cloud
    
    Args:
        geometry: the geometry of the point cloud
    
    Returns:
        A triangle mesh
    """
    points = np.asarray(geometry.points)

    skip=20
    traj = o3d.geometry.TriangleMesh()

    for ii in range(0, points.shape[0], skip):
        s = o3d.geometry.TriangleMesh.create_sphere(0.2 * o3dvis.SCALE, resolution=5)
        s.translate(points[ii] - s.get_center())
        s.compute_vertex_normals()
        s.paint_uniform_color(POSE_COLOR['points'])
        traj += s
    return traj

class o3dvis(setting, Menu):
    # PAUSE = False
    IMG_COUNT = 0

    def __init__(self, width=1280, height=768, is_remote=False, name='MainGui'):
        super(o3dvis, self).__init__(width, height, name)
        self.scene_name = 'ramdon'
        self.Human_data = HUMAN_DATA(is_remote, data_format)
        self.fetched_data = {}
        for i, plane in enumerate(creat_chessboard()):
            self.add_geometry(plane, name=f'ground_{i}', archive=True, reset_bounding_box=True)
        self.load_scene(sample_path, [0,0,0.16])
        self.window.set_needs_layout()

    def load_scene(self, path, translate=[0,0,0], load_data_class=None, reset_bounding_box = False):
        self.window.close_dialog()
        if not os.path.isfile(path):
            self.warning_info(f'{path} is not a valid file')
            return
        name = os.path.basename(path).split('.')[0]
        self.scene_name = name
        # self._on_load_dialog_done(scene_path)
        geometry = load_pts(None, pcd_path=path, load_data_class=load_data_class)
        geometry.translate(translate)
        self.add_geometry(geometry, name=name, reset_bounding_box=reset_bounding_box)

    def load_traj(self, path, translate=[0,0,0], load_data_class=None):
        """
        `traj = load_pts(None, pcd_path=path, load_data_class=load_data_class)`
        
        The `load_pts` function is defined in `open3d/open3d/io/point_cloud.py` and it's a function that
        takes in a `pcd_path` and returns a `PointCloud` object
        
        Args:
          path: the path to the file you want to load
          translate: [0,0,0]
          load_data_class: the class of the data to be loaded, which is used to determine the data format.
        """
        self.window.close_dialog()
        if not os.path.isfile(path):
            self.warning_info(f'{path} is not a valid file')
            return
        name = os.path.basename(path).split('.')[0]
        # self._on_load_dialog_done(scene_path)
        try:
            traj = load_pts(None, pcd_path=path, load_data_class=load_data_class)
            traj.translate(translate)
            # traj = points_to_sphere(traj)
            self.add_geometry(traj, name=name)
        except Exception as e:
            self.warning_info("xyz = [1,2,3] if traj.shape[0] == 9 else [0,1,2]")

    def _on_load_smpl_done(self, filename):
        """
        The function loads the SMPL model and the corresponding tracking data, and then creates a point
        cloud and a triangle mesh for each joint in the SMPL model.
        
        Args:
          filename: the path to the SMPL model
        """
        self.window.close_dialog()
        self.Human_data.load(filename) 

        # camera settings
        cams = self.Human_data.set_cameras(offset_center = -0.2)
        for cam in cams:
            self.camera_setting.add_item(cam)

        # UI setting
        self.camera_setting.enabled = True
        self._on_select_camera(cams[0], 0)
        self.window.set_needs_layout()
        self.frame_slider_bar.enabled = True
        self.frame_edit.enabled = True
        self.play_btn.enabled = True
        
        # data setting
        if self.Human_data:
            self.tracking_setting.visible = True
            setting._START_FRAME_NUM = self.Human_data.humans['frame_num'][0]
            humans = self.Human_data.vis_data_list['humans']
            keys = list(humans.keys())
            self.total_frames = humans[keys[0]]['verts'].shape[0]
            
            data = {}
            self.human_points = []
            if 'point cloud' in self.Human_data.vis_data_list:
                data['human points'] = o3d.geometry.TriangleMesh()
                human_points = self.Human_data.vis_data_list['point cloud'][0]
                max_points = max([hp.shape[0] for hp in human_points])
                lenght = 0.015 * o3dvis.SCALE
                for ii in range(max_points):
                    p = o3d.geometry.TriangleMesh.create_sphere(lenght, resolution=5)
                    # p = o3d.geometry.TriangleMesh.create_box(lenght,lenght,lenght)
                    p.compute_vertex_normals()
                    p.paint_uniform_color(POSE_COLOR['points'])
                    data['human points'] += p
                    self.human_points.append(p)

            for key in keys:
                humans[key]['trans'] = vertices_to_joints(humans[key]['verts'], 0)
                traj = o3d.geometry.PointCloud()
                traj.points = o3d.utility.Vector3dVector(humans[key]['trans'])
                traj.paint_uniform_color(POSE_COLOR[key])
                # self.update_geometry(traj, f'traj_{key}')
                data[f'seg_traj_{key}'] = traj

                smpl = o3d.io.read_triangle_mesh(sample_path)
                smpl.vertex_colors = o3d.utility.Vector3dVector()
                data[key] = smpl

            self.fetched_data = dict(
                sorted(data.items(), key=lambda x: x[0]))

            if len(self.tracking_list) > 0:
                try:
                    start, end = self.Human_data.humans['frame_num'][0], self.Human_data.humans['frame_num'][-1]
                    self.tracking_list = self.tracking_list[start:end+1]
                except Exception as e:
                    print(e)


            self.update_data = self.update_smpl
            self.fetch_data = self.fetch_smpl
            self.add_thread(threading.Thread(target=self.thread))

    def _loading_pcds(self, path):
        super(o3dvis, self)._loading_pcds(path)
        if len(self.tracking_list) > 0:
            try:
                start, end = self.Human_data.humans['frame_num'][0], self.Human_data.humans['frame_num'][-1]
                self.tracking_list = self.tracking_list[start:end+1]

            except Exception as e:
                # self.update_data = setting.update_data
                self.frame_slider_bar.enabled = True
                self.frame_edit.enabled = True
                self.play_btn.enabled = True
                self.total_frames = len(self.tracking_list)
                self.add_thread(threading.Thread(target=self.thread))
                print(e)

    def update_smpl(self, data, initialized=True):
        """
        The "update_data" function is called by the "thread" function. 
        
        It updates the geometry of the scene
        
        Args:
          data: a dictionary of numpy arrays, where the keys are the names of the objects
          initialized: If True, the data is initialized. If False, the data is updated. Defaults to True
        """
        
        def func():
            for name in data:
                self.update_geometry(data[name], name, reset_bounding_box=False, freeze=o3dvis.FREEZE)
            self.window.set_needs_layout()
            self._unfreeze()
        gui.Application.instance.post_to_main_thread(self.window, func)

        if not initialized:
            self.change_pause_status()

    def fetch_smpl(self, ind):
        """
        It takes in a frame number, and returns a dictionary of all the data for that frame. 
        
        The data is stored in a dictionary called `fetched_data`. 
        
        Args:
          ind: the index of the frame
        
        Returns:
          the fetched data.
        """
        def set_smpl(smpl, key, iid):
            if iid >= 0:
                smpl.vertices = o3d.utility.Vector3dVector(
                    self.Human_data.vis_data_list['humans'][key]['verts'][iid])
                smpl.vertex_normals = o3d.utility.Vector3dVector()
                smpl.triangle_normals = o3d.utility.Vector3dVector()
                smpl.compute_vertex_normals()
                if len(smpl.vertex_colors) == 0:
                    smpl.paint_uniform_color(POSE_COLOR[key])
            else:
                smpl.vertices = o3d.utility.Vector3dVector(np.zeros((6890, 3)))

        try:
            vis_data = self.Human_data.vis_data_list

            if 'point cloud' in vis_data:
                pts = self.fetched_data['human points']
                points = vis_data['point cloud'][0]
                indexes = vis_data['point cloud'][1]
                pts.clear()
                if ind in indexes:
                    index = indexes.index(ind)
                    dd = self.human_points[:points[index].shape[0]]
                    for ii, p in enumerate(dd):
                        p.translate(points[index][ii] - p.get_center())
                        pts += p
                    # vertices = np.vstack([np.asarray(p.vertices) for p in dd])
                    pts.vertex_normals = o3d.utility.Vector3dVector()
                    pts.triangle_normals = o3d.utility.Vector3dVector()
                    pts.compute_vertex_normals()
                else:
                    index = -1
        except Exception as e:
            print("Error: %s" % e)

        try:
            for key, geometry in self.fetched_data.items():
                iid = index if 'pred' in key.lower() else ind
                if ('(s)' in key.lower() or '(f)' in key.lower()) and 'seg_traj_' not in key.lower():
                    set_smpl(geometry, key, iid)
                    traj = self.fetched_data['seg_traj_' + key]

                    xyz = vis_data['humans'][key]['trans'][:ind+1]
                    traj.points = o3d.utility.Vector3dVector(xyz)
                    traj.normals = o3d.utility.Vector3dVector()
                    traj.paint_uniform_color(POSE_COLOR[key])
                    
        except Exception as e:
            print(e)

        if len(self.tracking_list)>0:
            try:
                if 'LiDAR frame' not in self._geo_list or self._geo_list['LiDAR frame']['box'].checked:
                    point_cloud = self.get_tracking_data(ind)
                    if len(point_cloud.points) > 0:
                        self.fetched_data['LiDAR frame'] = point_cloud
            except Exception as e:
                print(e)

        return self.fetched_data

    def set_camera(self, new_id, ind, pov):
        """
        It sets the camera to the given index and point of view.
        
        Args:
          ind: the index of the camera
          pov: the camera's point of view
        """
        posistions, extrinsics = self.Human_data.get_extrinsic(pov)
        new_ex = extrinsics[new_id] @ self.COOR_INIT

        if self.btn_freeview.checked:
            # rot和trans都可以自由拖动
            # 自由视角
            ex = extrinsics[ind] @ self.COOR_INIT
            new_cam = extrinsic_to_cam(new_ex)
            cam = extrinsic_to_cam(ex)
            cur_cam = self._scene.scene.camera.get_model_matrix()

            pos_to_cam = cur_cam[:3, 3] - self.COOR_INIT[:3, :3] @ posistions[ind]
            rela_rot = new_cam[:3, :3] @ cam[:3, :3].T
            rela_trans = self.COOR_INIT[:3, :3] @ (posistions[new_id] - posistions[ind])
            rel_pos = rela_rot @ pos_to_cam - pos_to_cam + rela_trans
            new_rot = rela_rot @ cur_cam[:3, :3]

            if new_id > 0 and self.btn_rela_trans.checked:
                # trans是相对trans
                cur_cam[:3, -1] = cur_cam[:3, -1] + rela_trans # new camera position
                new_ex = cam_to_extrinsic(cur_cam)

            elif new_id > 0:
                # trans是相对trans
                cur_cam[:3, -1] = cur_cam[:3, -1] + rel_pos # new camera position
                cur_cam[:3, :3] = new_rot # new camera position
                new_ex = cam_to_extrinsic(cur_cam)

        self.init_camera(new_ex)

    def add_geometry(self, 
                    geometry, 
                    name=None, 
                    mat=None, 
                    reset_bounding_box=False, 
                    archive=False, 
                    freeze=False):
        """
        If the geometry is a point cloud / mesh, it will be added to the scene. 
        
        Args:
          geometry: The geometry to be added.
          name: The name of the geometry.
          mat: material
          reset_bounding_box: If True, the camera will be reset to fit the geometry. Defaults to True
          archive: If True, the geometry will be saved in the archive. Defaults to False
          freeze: If True, the geometry will be added to the scene as a frozen object. Defaults to False
        """
        if mat is None:
            mat = self.settings.material 
        if name is None:
            name = self.scene_name

        try: 
            if geometry.has_points():
                if not geometry.has_normals():
                    geometry.estimate_normals()
                geometry.normalize_normals()
                gtype = 'point'
        except:
            try:
                if not geometry.has_triangle_normals():
                    geometry.compute_vertex_normals()
                if len(geometry.vertex_colors) == 0:
                    geometry.paint_uniform_color([1, 1, 1])
                # Make sure the mesh has texture coordinates
                if not geometry.has_triangle_uvs():
                    uv = np.array([[0.0, 0.0]] * (3 * len(geometry.triangles)))
                    geometry.triangle_uvs = o3d.utility.Vector2dVector(uv)
                gtype = 'mesh'
                # self_intersecting = geometry.is_self_intersecting()
                # watertight = geometry.is_watertight()
            except Exception as e:
                self.remove_geometry(name)
                # print(e)
                # print("[Info]", "not pointcloud or mesh.")
                return 

        if name not in self._geo_list:
            self.make_material(geometry, name, gtype, archive, point_size=2)

        if self._geo_list[name]['box'].checked and geometry:
            
            if 'seg_traj' in name:
                geometry = sample_traj(geometry)

            geometry.scale(o3dvis.SCALE, (0.0, 0.0, 0.0))
            self.remove_geometry(name)
            if freeze:
                self.add_freeze_data(name, geometry, self._geo_list[name]['mat'].material)
            else:
                self._scene.scene.add_geometry(name, geometry, self._geo_list[name]['mat'].material)
                self._scene.scene.set_geometry_transform(name, self.COOR_INIT)

                if self._scene_traj.visible:
                    self._scene_traj.scene.add_geometry(name, geometry, self._geo_list[name]['mat'].material)
                    self._scene_traj.scene.set_geometry_transform(name, self.COOR_INIT)

        else:
            self.remove_geometry(name)
                    
        if reset_bounding_box:
            try:
                bounds = geometry.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, bounds.get_center())
                self._scene_traj.setup_camera(60, bounds, bounds.get_center())
            except:
                print("[WARNING] It is t.geometry type")

        self._on_show_geometry(True)

    def set_view(self, view):
        pass

    def get_camera(self):
        """
        > The function `get_camera` returns the camera view matrix of the current scene
        The Y-axis and Z-axis of the scene are opposite of the camera in world coordinates.
        
        Returns:
          The camera view matrix.
        """
        view = self._scene.scene.camera.get_view_matrix()
        view[1, :] = - view[1, :]
        view[2, :] = - view[2, :]
        view[:3, 3] /= o3dvis.SCALE 
        return view

    def get_camera_pos(self):
        ex = self.get_camera()
        return -ex[:3, :3].T @ ex[:3, 3]

    def camera_fix(self, axsis = 'roll', extrinsic=None, setup=True):
        if extrinsic is None:
            extrinsic = self.get_camera()
        cam = extrinsic_to_cam(extrinsic)
        cam_pos = cam[:3, 3]
        if axsis == 'roll':
            world = cam_pos - 4 * cam[:3, 2] # cam's Z direction in world coordinates
            up = self.COOR_INIT[:3, :3] @ np.array([0,0,1])
        elif axsis == 'down':
            world = cam_pos - self.COOR_INIT[:3, :3] @ np.array([0,0,1]) * 4
            up = cam[:3, 1]
            up[1] = 0    # Y-axis is the Up direction in world coordinate # 
            if sum(abs(up)) < 1e-6:
              up = -cam[:3, 2]
            up = up / np.linalg.norm(up)
        elif axsis == 'forward':
            up = self.COOR_INIT[:3, :3] @ np.array([0,0,1])
            forward = cam[:3, 2]
            forward[1] = 0
            if sum(abs(forward)) < 1e-6:
              forward = -cam[:3, 1]
            world = cam_pos - forward * 4

        self._scene.look_at(world, cam_pos, up)

    def init_camera(self, extrinsic_matrix=None, intrinsic_factor=None): 
        """
        > The function `init_camera` sets up the camera for the scene. 
        
        The function takes in an optional argument `extrinsic_matrix` which is a 4x4 matrix that represents
        the camera's position and orientation in the world. If this argument is not provided, the function
        will use the camera's current position and orientation. 
        
        The function then sets up the camera for the scene. The camera is set up by calling the function
        `setup_camera` from the `Scene` class. The function `setup_camera` takes in the following arguments:
        
        - `intrinsic`: a 3x3 matrix that represents the camera's intrinsic parameters. 
        - `extrinsic`: a 4x4 matrix that represents the camera's extrinsic parameters. 
        - `width`: the width of the image. 
        - `height`: the height of the image.
        
        Args:
          extrinsic_matrix: the camera pose
        """
        bounds = self._scene.scene.bounding_box
        x = self._scene.frame.width
        y = self._scene.frame.height

        cx, cy = x/2, y/2
        if intrinsic_factor is None:
            intrinsic_factor = o3dvis.INTRINSIC_FACTOR
        fx = fy = cx * intrinsic_factor
        self.intrinsic = np.array([[fx, 0., cx],
                                    [0. , fy, cy],
                                    [0. , 0., 1.]])
        if o3dvis.FIX_CAMERA or extrinsic_matrix is None:
            extrinsic_matrix = self.get_camera()

        extrinsic_matrix[:3, 3] *= o3dvis.SCALE 

        if self._fix_roll.is_on:
            self.camera_fix('roll', extrinsic_matrix)
            extrinsic_matrix = self.get_camera()

        self._scene.setup_camera(self.intrinsic, extrinsic_matrix, x, y, bounds)


    def make_material(self, geometry, name, gtype, is_archive=False, point_size=2, color=[0.9, 0.9, 0.9, 1.0]):
        if is_archive:
            box = self.archive_box
        else:
            hh = gui.Horiz(0.5 * self.window.theme.font_size)
            # add_btn(hh, 'Del', lambda: self._remove_geo_list(name))
            add_btn(hh, 'Set', lambda: self._on_material_setting(name))
            box = add_box(hh, name, lambda c : self._show_geo_by_name(name, c), True)
            root = self.geo_check_boxes.get_root_item()
            self.geo_check_boxes.add_item(root, hh)

        self.window.set_needs_layout()
        settings = mat_set()
        if 'traj' in name.lower() or 'tracking' in name.lower():
            shader = settings.UNLIT
            point_size = 4
            color[-1] = 0.8
        else:
            shader = settings.LIT

        if ('(s)' in name.lower() or '(f)' in name.lower()) and 'ours' not in name.lower():
            box.checked = False

        settings.set_material(shader)
        settings.material.point_size = int(point_size)
        settings.material.base_color = color
        # settings.apply_material_prefab(settings.DEFAULT_MATERIAL_NAME)

        self._geo_list[name] = {
            'geometry': geometry, 
            'type': gtype, 
            'box': box,
            'mat': settings, 
            'archive': is_archive,
            'freeze': False}

def sample_traj(point_cloud, dist=0.1):
    xyz = np.asarray(point_cloud.points)
    diffs = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    indices = [0]
    sum_diff = 0
    for i in range(diffs.shape[0]):
        sum_diff += diffs[i]
        if sum_diff > dist:
            indices.append(i+1)
            sum_diff = 0
    return point_cloud.select_by_index(indices)
    # return point_cloud.uniform_down_sample(5)

def main():
    gui.Application.instance.initialize()

    w = o3dvis(1280, 720)
    
    gui.Application.instance.run()

    w.close_thread()

if __name__ == "__main__":
    main()
