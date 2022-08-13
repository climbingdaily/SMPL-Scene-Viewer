import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import sys
import cv2
import threading
import os
import time
from scipy.spatial.transform import Rotation as R

def create_button(name, callback, h_padding=0.5, v_padding=0):
    button = gui.Button(name)
    button.horizontal_padding_em = h_padding
    button.vertical_padding_em = v_padding
    button.set_on_clicked(callback)
    return button


def add_checkbox(vis, name, callback):
    vis._show_skybox = gui.Checkbox(name)
    self._show_skybox.set_on_checked(self._on_show_skybox)
    view_ctrls.add_child(self._show_skybox)


def _init_left_panel(self):
    em = self.window.theme.font_size
    def go_to_pcd_frame():
        frame = int(self._pcd_frame_index_edit.text_value)
        # self.pcd_frame_step(frame - self.cur_pcd_index)

    button_layout8 = gui.Horiz()
    self._pcd_frame_index_edit = gui.TextEdit()
    button_layout8.add_child(self._pcd_frame_index_edit)
    button_layout8.add_fixed(0.25 * em)
    button_layout8.add_child(create_button('go', go_to_pcd_frame))

    button_layout6 = gui.Horiz()
    button_layout6.add_child(gui.Label('Start,End:'))
    self._pcds_range_edit = gui.TextEdit()
    self._pcds_range_edit.text_value = '0,-1'
    button_layout6.add_child(self._pcds_range_edit)
    button_layout6.add_fixed(0.25 * em)
    button_layout6.add_child(create_button('go', go_to_pcd_frame))
    button_layout6.add_fixed(0.25 * em)
    button_layout6.add_child(create_button('l', go_to_pcd_frame))
    button_layout6.add_fixed(0.25 * em)
    button_layout6.add_child(create_button('r', go_to_pcd_frame))

    self.left_pannel = gui.Vert()
    self.left_pannel.add_child(button_layout8)
    self.left_pannel.add_fixed(0.5 * em)
    self.left_pannel.add_child(button_layout6)
    self.left_pannel.add_fixed(0.5 * em)
    self.window.add_child(self.left_pannel)