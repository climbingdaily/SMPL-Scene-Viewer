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
import cv2
import threading
import os
import time
import matplotlib.pyplot as plt

sys.path.append('.')

from gui_vis import HUMAN_DATA, Setting_panal as setting, Menu, creat_chessboard
from util import load_scene as load_pts, images_to_video, icp_mesh_and_point
sample_path = os.path.join(os.path.dirname(__file__), 'smpl', 'sample.ply')


POSE_KEY = ['First opt_pose', 'First pose', 'Second opt_pose', 'Second pose', 'Second pred']
POSE_COLOR = {'points': plt.get_cmap("tab20b")(1)[:3]}
for i, color in enumerate(POSE_KEY):
    POSE_COLOR[color] = plt.get_cmap("Paired")(i*2)[:3]

class o3dvis(setting, Menu):
    # PAUSE = False
    IMG_COUNT = 0

    def __init__(self, width=1280, height=768, is_remote=False):
        super(o3dvis, self).__init__(width, height)
        self.COOR_INIT = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        # self._scene.scene.view.set_shadowing(1)
        self.scene_name = 'ramdon'
        self.Human_data = HUMAN_DATA(is_remote)
        self.is_done = False
        self.data_names = {}
        for i, plane in enumerate(creat_chessboard()):
            self.add_geometry(plane, name=f'ground_{i}', archive=True)
        self.load_data(sample_path, [0,0,0.16])
        self.window.set_needs_layout()

    def load_data(self, path, translate=[0,0,0], load_data_class=None):
        self.window.close_dialog()
        if not os.path.isfile(path):
            print(f'{path} is not a valid file')
            return
        name = os.path.basename(path).split('.')[0]
        # self._on_load_dialog_done(scene_path)
        geometry = load_pts(None, pcd_path=path, load_data_class=load_data_class)
        geometry.translate(translate)
        self.add_geometry(geometry, name=name)

    def load_scene(self, scene_path):
        self.window.close_dialog()
        if not os.path.isfile(scene_path):
            return
        self.scene_name = os.path.basename(scene_path).split('.')[0]
        # self._on_load_dialog_done(scene_path)
        load_pts(self, scene_path)

    def _on_load_smpl_done(self, filename):
        self.window.close_dialog()
        self.Human_data.load(filename) 

        cams = self.Human_data.set_cameras(offset_center = -0.2)
        for cam in cams:
            self.camera_setting.add_item(cam)
        self.camera_setting.enabled = True
        self.window.set_needs_layout()

        self._on_select_camera(cams[0], 0)

        if self.Human_data:
            threading.Thread(target=self.update_thread).start()

    def reset_settings(self):
        o3dvis.IMG_COUNT = 0
        # o3dvis.FREE_VIEW = False
        # o3dvis.PAUSE = False
        # o3dvis.POV = 'first'
        # o3dvis.RENDER = False
        # self._set_slider_value(0)

    def update_thread(self):
        """
        The function is to render the 3D point cloud and the SMPL mesh in the same window
        """
        vis_data = self.Human_data.vis_data_list

        smpl_geometries = []
        human_data = vis_data['humans']

        pointcloud = o3d.geometry.PointCloud()
        if 'point cloud' in vis_data:
            points = vis_data['point cloud'][0]
            indexes = vis_data['point cloud'][1]
        else:
            indexes = []

        for i in human_data:
            smpl = o3d.io.read_triangle_mesh(sample_path)
            smpl.vertex_colors = o3d.utility.Vector3dVector()
            smpl_geometries.append(smpl) # a ramdon SMPL mesh
    
        keys = list(human_data.keys())
        init_param = False

        total_frames = human_data[keys[0]].shape[0]

        self._set_slider_limit(0, total_frames - 1)

        while True:
            video_name = self.scene_name + time.strftime("-%Y-%m-%d_%H-%M", time.localtime())
            image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'temp_{video_name}')
            self.reset_settings()
            self._set_slider_value(0)
            
            while self._get_slider_value() < total_frames - 1:
                def fetch_meshs(ind):

                    if ind in indexes:
                        index = indexes.index(ind)
                        pointcloud.points = o3d.utility.Vector3dVector(points[index])
                    else:
                        index = -1
                        pointcloud.points = o3d.utility.Vector3dVector(np.array([[0,0,0]]))

                    pointcloud.normals = o3d.utility.Vector3dVector()
                    pointcloud.paint_uniform_color(POSE_COLOR['points'])

                    for idx, smpl in enumerate(smpl_geometries):
                        # smpl = o3d.geometry.TriangleMesh()
                        key = keys[idx]
                        if 'pred' in key.lower():
                            if index >= 0:
                                smpl.vertices = o3d.utility.Vector3dVector(human_data[key][index])
                                smpl.vertex_normals = o3d.utility.Vector3dVector()
                                smpl.triangle_normals = o3d.utility.Vector3dVector()
                                # rt = self.Human_data.humans['first_person']['lidar_traj'][index, 1:8]
                                # _, delta = icp_mesh_and_point(smpl, pointcloud, rt, 0.05)
                                # smpl.translate(delta)
                                smpl.compute_vertex_normals()
                                if len(smpl.vertex_colors) == 0:
                                    smpl.paint_uniform_color(POSE_COLOR[key])
                            else:
                                smpl.vertices = o3d.utility.Vector3dVector(np.zeros((6890, 3)))
                        elif 'first' in key.lower() or 'second' in key.lower():
                            smpl.vertices = o3d.utility.Vector3dVector(human_data[key][ind])
                            smpl.vertex_normals = o3d.utility.Vector3dVector()
                            smpl.triangle_normals = o3d.utility.Vector3dVector()
                            # if 'second' in key.lower() and index > 0:
                            #     rt = self.Human_data.humans['first_person']['lidar_traj'][index, 1:8]
                            #     _, delta = icp_mesh_and_point(smpl, pointcloud, rt, 0.05)
                            #     smpl.translate(delta)
                            smpl.compute_vertex_normals()
                            if len(smpl.vertex_colors) == 0:
                                smpl.paint_uniform_color(POSE_COLOR[key])
                        else :
                            print('Edit your key in human_data here!')
                
                def add_first_cloud():
                    self.add_geometry(pointcloud, reset_bounding_box = False, name='human points')  
                    for si, smpl in enumerate(smpl_geometries):
                        self.add_geometry(smpl, reset_bounding_box = False, name=keys[si])  

                def updata_cloud():
                    freeze = o3dvis.FREEZE
                    self.update_geometry(pointcloud,  name='human points', freeze=freeze) 
                    for si, smpl in enumerate(smpl_geometries):
                        self.update_geometry(smpl, name=keys[si], freeze=freeze)  
                    self._unfreeze()

                def save_img():
                    self.save_imgs(image_dir)

                frame_index = self._get_slider_value()
                fetch_meshs(frame_index)
                self.set_camera(frame_index, o3dvis.POV)
                
                if not init_param:
                    init_param = True
                    gui.Application.instance.post_to_main_thread(self.window, add_first_cloud)
                    self.change_pause_status()
                else:
                    gui.Application.instance.post_to_main_thread(self.window, updata_cloud)

                if o3dvis.RENDER:
                    gui.Application.instance.post_to_main_thread(self.window, save_img)
                time.sleep(0.05)

                while True:
                    cv2.waitKey(10)
                    if o3dvis.CLICKED:
                        if frame_index != self._get_slider_value() or o3dvis.FREEZE:
                            frame_index = self._get_slider_value()
                            fetch_meshs(frame_index)
                            gui.Application.instance.post_to_main_thread(self.window, updata_cloud)
                        self.set_camera(frame_index, o3dvis.POV)

                        self._clicked()

                    if not o3dvis.PAUSE:
                        break
                
                self._set_slider_value(frame_index+1)
                    
            images_to_video(image_dir, video_name, delete=True)
            if not o3dvis.PAUSE:
                self.change_pause_status()

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

    def add_geometry(self, geometry, name=None, mat=None, 
                        reset_bounding_box=True, 
                        archive=False, 
                        freeze=False):
        if mat is None:
            mat =self.settings.material
        if name is None:
            name = self.scene_name
        geometry_type = ''
        try: 
            if geometry.has_points():
                if not geometry.has_normals():
                    geometry.estimate_normals()
                geometry.normalize_normals()
                if name not in self.point_list:
                    self.point_list[name] = geometry
                geometry_type = 'point'

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
                if name not in self.mesh_list:
                    self.mesh_list[name] = geometry
                geometry_type = 'mesh'
                # self_intersecting = geometry.is_self_intersecting()
                # watertight = geometry.is_watertight()
            except Exception as e:
                print(e)
                print("[Info]", "not pointcloud or mehs.")

        geometry.rotate(self.COOR_INIT[:3, :3], self.COOR_INIT[:3, 3])
        geometry.scale(o3dvis.SCALE, (0.0, 0.0, 0.0))

        if archive:
            self.archive_data.append(name)
            self._scene.scene.add_geometry(name, geometry, mat)

        elif name not in self.data_names.keys():
            box = gui.Checkbox(name)
            box.set_on_checked(self._on_show_geometry)
            box.checked = True
            self.check_boxes.add_child(box)
            self.data_names[name] = box
            self.window.set_needs_layout()

        elif self._scene.scene.has_geometry(name):
            self._scene.scene.remove_geometry(name)
        
        if name in self.data_names and self.data_names[name].checked:
            if freeze:
                ss = time.time()
                fname = f'{name}_freeze_{ss}'
                self.freeze_data.append(fname)
                self._scene.scene.add_geometry(fname, geometry, mat)
                if geometry_type == 'point':
                    self.point_list[name] = geometry
                elif geometry_type == 'mesh':
                    self.mesh_list[name] = geometry
            else:
                self._scene.scene.add_geometry(name, geometry, mat)
        
        if reset_bounding_box:
            try:
                bounds = geometry.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except:
                print("[WARNING] It is t.geometry type")

    def _on_show_geometry(self, show):
        for name, box in self.data_names.items():
            self._scene.scene.show_geometry(name, box.checked)
        # self._apply_settings()

    def update_geometry(self, geometry, name, freeze=False, reset_bounding_box=True):
        self.add_geometry(geometry, name, reset_bounding_box=False, freeze=freeze)

    def remove_geometry(self, name):
        self._scene.scene.remove_geometry(name)

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
        x = self.window.content_rect.width
        y = self.window.content_rect.height
        cx, cy = x/2, y/2
        fx = fy = cx * o3dvis.INTRINSIC_FACTOR
        self.intrinsic = np.array([[fx, 0., cx],
                                    [0. , fy, cy],
                                    [0. , 0., 1.]])
        if o3dvis.FIX_CAMERA or extrinsic_matrix is None:
            extrinsic_matrix = self.get_camera()
        extrinsic_matrix[:3, 3] *= o3dvis.SCALE 
        self._scene.setup_camera(self.intrinsic, extrinsic_matrix, x, y, bounds)

    def save_imgs(self, img_dir):
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_path = os.path.join(img_dir, f'{o3dvis.IMG_COUNT:04d}.jpg')
        o3dvis.IMG_COUNT += 1
        self.export_image(img_path, 1280, 720)

    def waitKey(self, key=0, helps=False):
        pass
    
def main():
    gui.Application.instance.initialize()

    w = o3dvis(1280, 720)

    gui.Application.instance.run()

if __name__ == "__main__":
    main()
