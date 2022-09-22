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

from statistics import geometric_mean
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import sys
import cv2
import threading
import os
import time
import matplotlib.pyplot as plt

sys.path.append('.')

from gui_vis import HUMAN_DATA, Setting_panal as setting, Menu, creat_chessboard, add_box, mat_set, add_btn
from util import load_scene as load_pts
sample_path = os.path.join(os.path.dirname(__file__), 'smpl', 'sample.ply')

# POSE_KEY = ['First opt_pose', 'Second opt_pose', 'First pose', 'Second pose', 'Second pred']
POSE_KEY = ['Ours(F)', 'Ours(S)', 'Baseline2(F)', 'Baseline2(S)',
            'Baseline1(F)', 'Baseline1(S)', 'Second pred', 'Ours_opt(F)']
# POSE_COLOR = {'points': plt.get_cmap("tab20b")(1)[:3]}
POSE_COLOR = {'points': [1,1,1]}
for i, color in enumerate(POSE_KEY):
    POSE_COLOR[color] = plt.get_cmap("tab20")(i*2 + 1)[:3]

POSE_COLOR['Ours(F)'] = [58/255, 147/255, 189/255]
POSE_COLOR['Ours(S)'] = [208/255, 80/255, 80/255]

mat_box = o3d.visualization.rendering.MaterialRecord()
mat_box.shader = 'defaultLitTransparency'
# mat_box.shader = 'defaultLitSSR'
# mat_box.base_color = [0.467, 0.467, 0.467, 0.2]
# mat_box.base_roughness = 0.0
# mat_box.base_reflectance = 0.0
# mat_box.base_clearcoat = 1.0
# mat_box.thickness = 1.0
# mat_box.transmission = 1.0
# mat_box.absorption_distance = 10
# mat_box.absorption_color = [0.5, 0.5, 0.5]

class o3dvis(setting, Menu):
    # PAUSE = False
    IMG_COUNT = 0

    def __init__(self, width=1280, height=768, is_remote=False):
        super(o3dvis, self).__init__(width, height)
        self.COOR_INIT = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        self.scene_name = 'ramdon'
        self.Human_data = HUMAN_DATA(is_remote)
        self.is_done = False
        for i, plane in enumerate(creat_chessboard()):
            self.add_geometry(plane, name=f'ground_{i}', archive=True)
        self.load_scene(sample_path, [0,0,0.16])
        self.window.set_needs_layout()

    def load_scene(self, path, translate=[0,0,0], load_data_class=None):
        self.window.close_dialog()
        if not os.path.isfile(path):
            self.warning_info(f'{path} is not a valid file')
            return
        name = os.path.basename(path).split('.')[0]
        self.scene_name = name
        # self._on_load_dialog_done(scene_path)
        geometry = load_pts(None, pcd_path=path, load_data_class=load_data_class)
        geometry.translate(translate)
        self.add_geometry(geometry, name=name)

    def load_traj(self, path, translate=[0,0,0], load_data_class=None):
        self.window.close_dialog()
        if not os.path.isfile(path):
            self.warning_info(f'{path} is not a valid file')
            return
        name = os.path.basename(path).split('.')[0]
        # self._on_load_dialog_done(scene_path)
        traj = load_pts(None, pcd_path=path, load_data_class=load_data_class)
        traj.translate(translate)
        # traj = self.points_to_sphere(traj)
        self.add_geometry(traj, name=name)

    def points_to_sphere(self, geometry):
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

    def _on_load_smpl_done(self, filename):
        self.window.close_dialog()
        self.Human_data.load(filename) 

        cams = self.Human_data.set_cameras(offset_center = -0.2)
        for cam in cams:
            self.camera_setting.add_item(cam)
        self.camera_setting.enabled = True
        self.window.set_needs_layout()

        self._on_select_camera(cams[0], 0)

        self.frame_slider_bar.enabled = True
        self.frame_edit.enabled = True
        self.play_btn.enabled = True
        
        if self.Human_data:
            self.tracking_setting.visible = True

            humans = self.Human_data.vis_data_list['humans']
            keys = list(humans.keys())
            self.total_frames = humans[keys[0]]['verts'].shape[0]
            
            data = {}
            # data['human points'] = o3d.geometry.PointCloud()
            data['human points'] = o3d.geometry.TriangleMesh()
            
            self.human_points = []

            for ii in range(512):
                p = o3d.geometry.TriangleMesh.create_sphere(0.015 * o3dvis.SCALE, resolution=5)
                p.compute_vertex_normals()
                p.paint_uniform_color(POSE_COLOR['points'])
                data['human points'] += p
                self.human_points.append(p)

            for key in keys:
                traj = o3d.geometry.PointCloud()
                traj.points = o3d.utility.Vector3dVector(humans[key]['trans'].squeeze())
                self.update_geometry(traj, f'traj_{key}')

                smpl = o3d.io.read_triangle_mesh(sample_path)
                smpl.vertex_colors = o3d.utility.Vector3dVector()
                data[key] = smpl

            self.fetched_data = dict(
                sorted(data.items(), key=lambda x: x[0]))

            self.add_thread(threading.Thread(target=self.thread))

    def _start_tracking(self, path):
        super(o3dvis, self)._start_tracking(path)
        if len(self.tracking_list) > 0:
            try:
                start, end = self.Human_data.humans['frame_num'][0], self.Human_data.humans['frame_num'][-1]
                self.tracking_list = self.tracking_list[start:end+1]
            except Exception as e:
                print(e)

    def update_data(self, data, initialized=True):
        def func():
            for name in data:
                self.update_geometry(data[name], name, reset_bounding_box=False, freeze=o3dvis.FREEZE)
            self._unfreeze()
        gui.Application.instance.post_to_main_thread(self.window, func)
        # time.sleep(0.01)

        if not initialized:
            self.change_pause_status()

    def fetch_data(self, ind):
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

        vis_data = self.Human_data.vis_data_list

        for ii, p in enumerate(self.human_points):
            p.translate(-p.get_center())

        if 'point cloud' in vis_data:
            points = vis_data['point cloud'][0]
            indexes = vis_data['point cloud'][1]

            if ind in indexes:
                index = indexes.index(ind)
                # pointcloud.points = o3d.utility.Vector3dVector(points[index])
                for ii, p in enumerate(self.human_points):
                    p.translate(points[index][ii] - p.get_center())
            else:
                index = -1
                # pointcloud.points = o3d.utility.Vector3dVector(np.array([[0,0,0]]))
        hps = self.fetched_data['human points']
        vertices = np.vstack([np.asarray(p.vertices) for p in self.human_points])
        hps.vertices = o3d.utility.Vector3dVector(vertices)
        hps.vertex_normals = o3d.utility.Vector3dVector()
        hps.triangle_normals = o3d.utility.Vector3dVector()
        hps.compute_vertex_normals()

        for key, geometry in self.fetched_data.items():
            iid = index if 'pred' in key.lower() else ind
            if '(s)' in key.lower() or '(f)' in key.lower():
                set_smpl(geometry, key, iid)
        try:
            self.fetched_data['LiDAR frame'] = self.get_tracking_data(ind)
        except Exception as e:
            pass

        return self.fetched_data

    def set_camera(self, ind, pov):
        _, extrinsics = self.Human_data.get_extrinsic(pov)
        if ind > 0 and o3dvis.FREE_VIEW:
            camera_pose = self.get_camera()
            relative_trans = -extrinsics[ind][:3, :3].T @ extrinsics[ind][:3, 3] + extrinsics[ind-1][:3, :3].T @ extrinsics[ind-1][:3, 3]
            relative_trans = self.COOR_INIT[:3, :3] @ relative_trans
            camera_positon = -(camera_pose[:3, :3].T @ camera_pose[:3, 3])
            camera_pose[:3, 3] = -(camera_pose[:3, :3] @ (camera_positon + relative_trans))
            self.init_camera(camera_pose)
        else:
            self.init_camera(extrinsics[ind] @ self.COOR_INIT)  

    def add_geometry(self, 
                    geometry, 
                    name=None, 
                    mat=None, 
                    reset_bounding_box=True, 
                    archive=False, 
                    freeze=False):

        if mat is None:
            mat =self.settings.material
        if name is None:
            name = self.scene_name

        try: 
            if geometry.has_points():
                if not geometry.has_normals():
                    geometry.estimate_normals()
                geometry.normalize_normals()
                type = 'point'
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
                type = 'mesh'
                # self_intersecting = geometry.is_self_intersecting()
                # watertight = geometry.is_watertight()
            except Exception as e:
                print(e)
                print("[Info]", "not pointcloud or mehs.")
                return 

        self.remove_geometry(name)

        if name not in self.geo_list:
            if archive:
                box = self.archive_box
            else: 
                hh = gui.Horiz(0.5 * self.window.theme.font_size)
                add_btn(hh, 'Property', self._on_material_setting)
                box = add_box(hh, name, self._on_show_geometry, True)
                self.check_boxes.add_item(self.check_boxes.get_root_item(), hh)

            self.window.set_needs_layout()
            self.geo_list[name] = {
                'geometry': geometry, 
                'type': type, 
                'box': box,
                'mat': mat_set(), 
                'archive': archive,
                'freeze': False}

        if self.geo_list[name]['box'].checked:
            geometry.rotate(self.COOR_INIT[:3, :3], self.COOR_INIT[:3, 3])
            geometry.scale(o3dvis.SCALE, (0.0, 0.0, 0.0))
            if freeze:
                self.add_freeze_data(name, geometry, self.geo_list[name]['mat'].material)
            else:
                self._scene.scene.add_geometry(name, geometry, self.geo_list[name]['mat'].material)
                if self._scene_traj.visible:
                    self._scene_traj.scene.add_geometry(name, geometry, self.geo_list[name]['mat'].material)
                    
        if reset_bounding_box:
            try:
                bounds = geometry.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, bounds.get_center())
                self._scene_traj.setup_camera(60, bounds, bounds.get_center())
            except:
                print("[WARNING] It is t.geometry type")

    def update_geometry(self, geometry, name, mat=None, reset_bounding_box=False, archive=False, freeze=False):
        self.add_geometry(geometry, name, mat, reset_bounding_box, archive, freeze) 

    def remove_geometry(self, name):
        if self._scene.scene.has_geometry(name):
            self._scene.scene.remove_geometry(name)
        if self._scene_traj.scene.has_geometry(name):
            self._scene_traj.scene.remove_geometry(name)

    def set_view(self, view):
        pass

    def get_camera(self):
        view = self._scene.scene.camera.get_view_matrix()
        view[1, :] = - view[1, :]
        view[2, :] = - view[2, :]
        view[:3, 3] /= o3dvis.SCALE 
        return view

    def get_camera_pos(self):
        ex = self.get_camera()
        return -ex[:3, :3].T @ ex[:3, 3]

    def init_camera(self, extrinsic_matrix=None): 
        bounds = self._scene.scene.bounding_box
        x = self._scene.frame.width
        y = self._scene.frame.height

        cx, cy = x/2, y/2
        fx = fy = cx * o3dvis.INTRINSIC_FACTOR
        self.intrinsic = np.array([[fx, 0., cx],
                                    [0. , fy, cy],
                                    [0. , 0., 1.]])

        if o3dvis.FIX_CAMERA or extrinsic_matrix is None:
            extrinsic_matrix = self.get_camera()

        extrinsic_matrix[:3, 3] *= o3dvis.SCALE 
        self._scene.setup_camera(self.intrinsic, extrinsic_matrix, x, y, bounds)
        # self._scene_traj.setup_camera(self.intrinsic, extrinsic_matrix, x, y, bounds)

def main():
    gui.Application.instance.initialize()

    w = o3dvis(1280, 720)
    
    gui.Application.instance.run()

    w.close_thread()

if __name__ == "__main__":
    main()
