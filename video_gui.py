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
import open3d as o3d

sys.path.append('.')

from main_gui import o3dvis as base_gui
from util import load_data_remote, images_to_video

class ImagWindow(base_gui):
    FRAME = 1

    def __init__(self, width=1280, height=768, is_remote=False):
        super(ImagWindow, self).__init__(width, height)
        self.tracked_frame = {}
        self.remote = is_remote
        # self._scene.set_on_mouse(self._on_mouse_widget3d)
        self.tracking_setting.visible = True
        self.window.set_needs_layout()

    def _loading_imgs(self, path):
        self._on_show_skybox(False)
        self._on_bg_color(gui.Color(1, 1, 1))
        self._on_show_ground_plane(False)
        self.archive_box.checked = False
        self._on_show_geometry(True)

        if path is None:
            path = self.remote_info['folder'].strip()
        self.tracking_foler = path

        self.tracking_list = []
        
        try:
            self.data_loader = load_data_remote(False)
            img_paths = self.data_loader.list_dir(path)
        except:
            try:
                password = self.remote_info['pwd'].strip()
                username = self.remote_info['username'].strip()
                hostname = self.remote_info['hostname'].strip()
                port = self.remote_info['port'].strip()
                self.data_loader = load_data_remote(True, username, hostname, int(port), password)
                img_paths = self.data_loader.list_dir(path)
            except Exception as e:
                print(e)
                self.warning_info(f"'{path}' \n Not valid! Please input the right remote info!!!")
                return
        if len(img_paths) <=0:
            self.warning_info(f"'{path}' \n Not valid! Please input the correct folder!!!")
            return

        for img_path in img_paths:
            if img_path.endswith('.jpg') or img_path.endswith('.png'):
                self.tracking_list.append(img_path)
        self.tracking_list = sorted(self.tracking_list, key=lambda x: float(x.split('.')[0].replace('_', '.')))
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
                # 1. 这里定义图片显示的方式
                self._scene.scene.set_background([1, 1, 1, 1], data[name])
                # self.update_geometry(data[name], name, reset_bounding_box=False, freeze=o3dvis.FREEZE)
                pass

            self._unfreeze()
        gui.Application.instance.post_to_main_thread(self.window, func)
        # time.sleep(0.01)

        if not initialized:
            self.change_pause_status()

    def fetch_img(self, index):
        # 2.这里定义你读取数据的方式，改成远程读取图片即可
        image = self.data_loader.load_imgs(self.tracking_foler + '/' + self.tracking_list[index])
        return {'imgs': image}

    def update_label(self, world):
        frame = self._get_slider_value()
        
        text = "{:.3f}, {:.3f}, {:.3f}".format(
            world[0], world[1], world[2])
        square_box = o3d.geometry.TriangleMesh.create_sphere(0.2 * ImagWindow.SCALE, 20, create_uv_map=True)
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
            eye = world - 3.5 * ImagWindow.SCALE * cam_to_select / np.linalg.norm(cam_to_select) 
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
            self.data_loader.write_txt(savepath, traj)
            self.warning_info(f'File saved in {savepath}', 'INFO')
        except Exception as e:
            self.warning_info(e.args[0])

    def set_camera(self, ind, pov):
        pass
    

def main():
    gui.Application.instance.initialize()

    w = ImagWindow(1280, 720, is_remote=True)

    gui.Application.instance.run()

    w.close_thread()

if __name__ == "__main__":
    main()
