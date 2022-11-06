################################################################################
# File: \video_gui.py                                                          #
# Created Date: Tuesday September 20th 2022                                    #
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
import open3d.visualization.gui as gui
import sys
import threading
import os
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.append('.')

from gui_vis.main_gui import o3dvis as base_gui

class ImagWindow(base_gui):

    def __init__(self, width=1280, height=768, is_remote=False, name='MyWindow'):
        super(ImagWindow, self).__init__(width, height, is_remote, name)
        self.tracked_frame = {}
        self.remote = is_remote
        self._scene.set_on_mouse(self._on_mouse_widget3d)
        self.tracking_setting.visible = True
        self.window.set_needs_layout()

    def _loading_imgs(self, path=None):
        self._on_show_skybox(False)
        self._on_bg_color(gui.Color(1, 1, 1))
        self._on_show_ground_plane(False)
        self.archive_box.checked = False
        self._on_show_geometry(True)

        self.tracking_list = []

        if path is None:
            path = self.remote_info['folder'].strip()
            
        img_paths = self._list_dir(path)

        for img_path in img_paths:
            if img_path.endswith('.jpg') or img_path.endswith('.png'):
                self.tracking_list.append(img_path)

        if len(self.tracking_list) > 0:
            self.tracking_list = sorted(self.tracking_list, key=lambda x: float(x[:-4]))
            if not ImagWindow.PAUSE:
                self.change_pause_status()
            self.warning_info(f"Images loaded from '{path}'", type='info')
            
            self.frame_slider_bar.enabled = True
            self.frame_edit.enabled = True
            self.play_btn.enabled = True
            self.total_frames = len(self.tracking_list)

            # A thread that will be called when the frameid changes
            self.update_data = self.update_img
            self.fetch_data = self.fetch_img
            self.add_thread(threading.Thread(target=self.thread))

    def update_img(self, data, initialized=True):
        def func():
            for name in data:
                self._scene.scene.set_background([1, 1, 1, 1], data[name])
                pass
            self._unfreeze()
        gui.Application.instance.post_to_main_thread(self.window, func)
        # time.sleep(0.01)
        if not initialized:
            self.change_pause_status()

    def fetch_img(self, index):
        image = self.data_loader.load_imgs(self.tracking_foler + '/' + self.tracking_list[index])
        return {'imgs': image}

    def _on_mouse_widget3d(self, event):
        """
        It takes the mouse click event, and then uses the depth image to get the 3D coordinates of the point
        clicked. 
        
        The depth image is a 2D image that contains the depth of each pixel in the scene. 
        The depth image is obtained by rendering the scene to a depth image. 
        The depth image is then used to get the 3D coordinates of the point clicked. 

        The 3D coordinates are then used to create a 3D label and a sphere at the clicked point. 
        The 3D label and the sphere are added to the scene and the dictionary `tracked_frame`. 

        The dictionary `tracked_frame` is used to store the 3D labels and the spheres. 
        
        Args:
          event: The event that triggered the callback.
        
        Returns:
          The return value is a gui.Widget.EventCallbackResult.HANDLED or
        gui.Widget.EventCallbackResult.IGNORED.
        """
        # We could override BUTTON_DOWN without a modifier, but that would
        # interfere with manipulating the scene.
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.CTRL):

            def depth_callback(depth_image):
                # Note that even if the
                # scene widget is the only thing in the window, if a menubar
                # exists it also takes up space in the window (except on macOS).
                x = event.x - self._scene.frame.x
                y = event.y - self._scene.frame.y
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                    text = ""
                else:
                    world = self._scene.scene.camera.unproject(
                        x, y, depth, self._scene.frame.width,
                        self._scene.frame.height)
                    frame = self._get_slider_value()
                    gui.Application.instance.post_to_main_thread(
                        self.window, lambda: self.update_label(world, frame))

            self._scene.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def update_label(self, world, frame):
        # position 
        position = self.COOR_INIT[:3, :3].T @ world
        text = "{:.3f}, {:.3f}, {:.3f}".format(
            position[0], position[1], position[2])
        try:
            time = self.tracking_list[frame].split('.')[0].replace('_', '.')
            text = f'{text} T: {time}'
        except:
            pass
        frame += ImagWindow._START_FRAME_NUM
        def create_sphere(world):
            ratio = min(frame / self._get_max_slider_value(), 1)
            # name = f'{frame}_trkpts'
            # point = o3d.geometry.PointCloud()
            # point.points = o3d.utility.Vector3dVector(position.reshape(-1, 3))
            # point.paint_uniform_color(plt.get_cmap("hsv")(ratio)[:3])

            # if name not in self.geo_list:
            #     self.make_material(point, name, 'point', is_archive=False, point_size=9)
            # self.update_geometry(
            #     point, name, reset_bounding_box=False, freeze=True)

            sphere = o3d.geometry.TriangleMesh.create_sphere(
                0.05 * base_gui.SCALE, 20, create_uv_map=True)
            sphere.translate(position)
            sphere.paint_uniform_color(plt.get_cmap("hsv")(ratio)[:3])

            self.update_geometry(
                sphere, f'{frame}_trkpts', reset_bounding_box=False, freeze=True)
        
        if frame in self.tracked_frame:
            self.tracked_frame[frame][0] = f'{frame}: {text}'
            self.tracked_frame[frame][1].position = world
            create_sphere(world)
        else:
            point_info = f'{frame}: {text}'
            label_3d = self._scene.add_3d_label(world, f'{frame}')
            label_3d.color = gui.Color(r=0, b=1, g=0.9)
            create_sphere(world)

            self.tracked_frame[frame] = []
            self.tracked_frame[frame].append(point_info)
            self.tracked_frame[frame].append(label_3d)

            # set the camera view
            cam_to_select = world - self.get_camera_pos()
            eye = world - 2.5 * base_gui.SCALE * cam_to_select / np.linalg.norm(cam_to_select) 
            up = self.COOR_INIT[:3, :3] @ np.array([0, 0, 1])
            self._scene.look_at(world, eye, up)

        self.update_tracked_points()

        base_gui.CLICKED = True
        self._on_slider(self._get_slider_value() + base_gui.TRACKING_STEP)

    def _load_tracked_traj(self, path, translate=[0,0,0], load_data_class=None):
        trajs = super(base_gui, self)._load_tracked_traj(path, translate, load_data_class)
        if trajs is not None:
            for p in trajs:
                self._set_slider_value(int(p[3]))
                self.update_label(self.COOR_INIT[:3, :3] @ p[:3], int(p[3]))

    def _save_traj(self):
        """
        > The function `_save_traj` saves the trajectory of the tracked object in the current video
        """
        try:
            keys = sorted(list(self.tracked_frame.keys()))
            positions = [self.tracked_frame[frame][1].position for frame in keys]
            try:
                times = [float(self.tracking_list[frame].split('.')[0].replace('_', '.')) for frame in keys]
            except Exception as e:
                times = [0] * len(keys)
            traj = np.hstack((np.array(positions) @ self.COOR_INIT[:3, :3], 
                            np.array(keys)[:, None], 
                            np.array(times)[:, None]))
            savepath = os.path.dirname(self.tracking_foler) + '/tracking_traj.txt'
            self.data_loader.write_txt(savepath, traj)
            self.warning_info(f'File saved in {savepath}', 'INFO')
        except Exception as e:
            self.warning_info(e.args[0])

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Run Vis Tools')
    parser.add_argument('--width', type=int, default=1280)  # predicted by IPNet
    parser.add_argument('--height', type=int, default=720)  # predicted by IPNet
    parser.add_argument('--name', type=str, default='MyWindow')  # predicted by IPNet
    args = parser.parse_args()
    
    gui.Application.instance.initialize()

    w = ImagWindow(args.width, args.height, name=args.name)

    try:
        gui.Application.instance.run()
    except Exception as e:
        print(e)

    w.close_thread()