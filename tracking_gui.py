################################################################################
# File: \tracking_gui.py                                                       #
# Created Date: Sunday August 21st 2022                                        #
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
import open3d as o3d
import matplotlib.pyplot as plt
import time

sys.path.append('.')

from main_gui import o3dvis as base_gui
from util import load_scene as load_pts

class trackingVis(base_gui):
    FRAME = 1

    def __init__(self, width=1280, height=768, is_remote=False):
        super(trackingVis, self).__init__(width, height)
        self.tracked_frame = {}
        self.remote = is_remote
        self._scene.set_on_mouse(self._on_mouse_widget3d)
        self.tracking_setting.visible = True
        self.window.set_needs_layout()

    def _start_tracking(self, path):
        super(trackingVis, self)._start_tracking(path)
        self.frame_slider_bar.enabled = True
        self.frame_edit.enabled = True
        self.play_btn.enabled = True
        self.total_frames = len(self.tracking_list)
        self.add_thread(threading.Thread(target=self.thread))

    def fetch_data(self, index):
        name = 'tracking frame'
        geometry = self.get_tracking_data(index)
        if name not in self.geo_list:
            self.make_material(geometry, name, 'point', is_archive=False)
            self.geo_list[name]['mat'].material.point_size = 8
        return {name: geometry}

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
        The dictionary `tracked_frame` is used to update the 3D labels and the spheres when the slider ismoved.
        
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
            text = f'{text} time: {time}'
        except:
            pass

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
                0.05 * trackingVis.SCALE, 20, create_uv_map=True)
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
            eye = world - 2.5 * trackingVis.SCALE * cam_to_select / np.linalg.norm(cam_to_select) 
            up = self.COOR_INIT[:3, :3] @ np.array([0, 0, 1])
            self._scene.look_at(world, eye, up)

        self.update_tracked_points()

        trackingVis.CLICKED = True
        self._on_slider(self._get_slider_value() + trackingVis.TRACKING_STEP)

    def load_traj(self, path, translate=[0,0,0], load_data_class=None):
        self.window.close_dialog()
        if not os.path.isfile(path):
            self.warning_info(f'{path} is not a valid file')
            return
        try:
            trajs = np.loadtxt(path)
        except Exception as e:
            self.warning_info('Load traj failed.')
            return 

        if trajs.shape[1] > 5 or trajs.shape[1] < 4:
            self.warning_info("Tracking trajs must contains: 'x y z frameid time' in every line!!!")
            return 

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
            times = [float(self.tracking_list[frame].split('.')[0].replace('_', '.')) for frame in keys]
            traj = np.hstack((np.array(positions) @ self.COOR_INIT[:3, :3], 
                            np.array(keys)[:, None], 
                            np.array(times)[:, None]))
            savepath = os.path.dirname(self.tracking_foler) + '/tracking_traj.txt'
            self.data_loader.write_txt(savepath, traj)
            self.warning_info(f'File saved in {savepath}', 'INFO')
        except Exception as e:
            self.warning_info(e.args[0])

    def set_camera(self, ind, pov):
        pass
    
def main():
    gui.Application.instance.initialize()

    w = trackingVis(1280, 720, is_remote=True)

    gui.Application.instance.run()

    w.close_thread()

if __name__ == "__main__":
    main()
