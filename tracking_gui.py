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

sys.path.append('.')

from main_gui import o3dvis as base_gui

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
        geomety = self.remote_load.load_point_cloud(self.tracking_foler + '/' + self.tracking_list[index])
        return {'tracking frame': geomety}

    def update_data(self, data, initialized=True):
        def func():
            for name in data:
                self.add_geometry(data[name], name, reset_bounding_box=False)
        gui.Application.instance.post_to_main_thread(self.window, func)
        if not initialized:
            self.change_pause_status()

    def _on_mouse_widget3d(self, event):
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
                        event.x, event.y, depth, self._scene.frame.width,
                        self._scene.frame.height)
                    text = "{:.3f}, {:.3f}, {:.3f}".format(
                        world[0], world[1], world[2])

                    def update_label():
                        frame = self._get_slider_value()
                        if frame in self.tracked_frame:
                            self.tracked_frame[frame][0] = f'{frame}: {text}'
                            self.tracked_frame[frame][1].position = world
                            square_box = o3d.geometry.TriangleMesh.create_sphere(0.5, 20, create_uv_map=True)
                            square_box.translate(self.COOR_INIT[:3, :3].T @ world)
                            self.update_geometry(square_box, f'{frame}_trkpts', reset_bounding_box=False, freeze=True)
                        else:
                            point_info = f'{frame}: {text}'
                            label_3d = self._scene.add_3d_label(world, f'{frame}')
                            label_3d.color = gui.Color(r=0, b=1, g=0.9)

                            square_box = o3d.geometry.TriangleMesh.create_sphere(0.5, 20, create_uv_map=True)
                            square_box.translate(self.COOR_INIT[:3, :3].T @ world)
                            self.add_geometry(square_box, f'{frame}_trkpts', reset_bounding_box=False, freeze=True)

                            self.tracked_frame[frame] = []
                            self.tracked_frame[frame].append(point_info)
                            self.tracked_frame[frame].append(label_3d)

                            cam_to_select = world - self.get_camera_pos()
                            eye = world - 3.5 * trackingVis.SCALE * cam_to_select / np.linalg.norm(cam_to_select) 
                            up = self.COOR_INIT[:3, :3] @ np.array([0, 0, 1])
                            self._scene.look_at(world, eye, up)

                        self.update_tracked_points()

                        trackingVis.CLICKED = True
                        self._on_slider(self._get_slider_value() +
                                        trackingVis.TRACKING_STEP)

                    gui.Application.instance.post_to_main_thread(
                        self.window, update_label)

            self._scene.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def update_label(self, world):
        frame = self._get_slider_value()
        
        text = "{:.3f}, {:.3f}, {:.3f}".format(
            world[0], world[1], world[2])
        square_box = o3d.geometry.TriangleMesh.create_sphere(0.2, 20, create_uv_map=True)
        square_box.translate(self.COOR_INIT[:3, :3].T @ world)

        if frame in self.tracked_frame:
            self.tracked_frame[frame][0] = f'{frame}: {text}'
            self.tracked_frame[frame][1].position = world
            self.update_geometry(square_box, f'{frame}_trkpts', reset_bounding_box=False, freeze=True)
        else:
            point_info = f'{frame}: {text}'
            label_3d = self._scene.add_3d_label(world, f'{frame}')
            label_3d.color = gui.Color(r=0, b=1, g=0.9)

            self.add_geometry(square_box, f'{frame}_trkpts', reset_bounding_box=False, freeze=True)

            self.tracked_frame[frame] = []
            self.tracked_frame[frame].append(point_info)
            self.tracked_frame[frame].append(label_3d)

            cam_to_select = world - self.get_camera_pos()
            eye = world - 3.5 * trackingVis.SCALE * cam_to_select / np.linalg.norm(cam_to_select) 
            up = self.COOR_INIT[:3, :3] @ np.array([0, 0, 1])
            self._scene.look_at(world, eye, up)

    def _save_traj(self):
        try:
            keys = sorted(list(self.tracked_frame.keys()))
            positions = [self.tracked_frame[frame][1].position for frame in keys]
            times = [float(self.tracking_list[frame].split('.')[0].replace('_', '.')) for frame in keys]
            traj = np.hstack((np.array(positions) @ self.COOR_INIT[:3, :3], 
                            np.array(keys)[:, None], 
                            np.array(times)[:, None]))
            savepath = os.path.dirname(self.tracking_foler) + '/tracking_traj.txt'
            self.remote_load.write_txt(savepath, traj)
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
