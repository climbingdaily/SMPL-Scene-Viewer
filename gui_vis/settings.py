################################################################################
# File: \settings.py                                                           #
# Created Date: Saturday August 13th 2022                                      #
# Author: climbingdaily                                                        #
# -----                                                                        #
# Modified By: the developer climbingdaily at yudidai@stu.xmu.edu.cn           #
# https://github.com/climbingdaily                                             #
# -----                                                                        #
# Copyright (c) 2022 yudidai                                                   #
# -----                                                                        #
# HISTORY:                                                                     #
################################################################################

from copy import deepcopy
import open3d.visualization.gui as gui
import sys
import os
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append('.')
sys.path.append('..')
from .base_gui import AppWindow as GUI_BASE, creat_btn
from util import load_data_remote, images_to_video

def create_combobox(func, names=None):
    combobox = gui.Combobox()
    if names is not None:
        for name in names:
            combobox.add_item(name)
    combobox.set_on_selection_changed(func)
    return combobox

def add_btn(layout, name, func, color=None):
    btn = gui.Button(name)
    btn.horizontal_padding_em = 0.2
    btn.vertical_padding_em = 0
    if color is not None:
        btn.background_color = gui.Color(r=color[0], b=color[1], g=color[2])
    btn.set_on_clicked(func)
    layout.add_child(btn)
    return btn


def add_Switch(layout, name, func, checked=False):
    switch = gui.ToggleSwitch(name)
    switch.set_on_clicked(func)
    switch.is_on = checked
    layout.add_child(switch)
    return switch

def add_box(layout, name, func, checked=False):
    box = gui.Checkbox(name)
    box.set_on_checked(func)
    box.checked = checked
    layout.add_child(box)
    return box

class Setting_panal(GUI_BASE):
    TRACKING_STEP = 1
    FREEZE = False
    FIX_CAMERA = False
    PAUSE = False
    POV = 'first'
    RENDER = False
    VIDEO_SAVE = False
    CLICKED = False
    INTRINSIC_FACTOR = 1
    SCALE = 1
    MYTHREAD = None
    IMG_COUNT = 0
    _START_FRAME_NUM = 0

    def __init__(self, width=1280, height=720, name='Settings'):
        super(Setting_panal, self).__init__(width, height, name)
        self.total_frames = 1
        self.tracking_list = []
        self.tracked_frame = {}
        self._selected_geo = 'sample'
        em = self.window.theme.font_size

        self.stream_setting = self.create_stream_settings()
        human_setting, camera_setting = self.create_humandata_settings()
        self.tracking_setting = self.tracking_tool_setting()

        # tabs = gui.TabControl()
        tabs = gui.Vert()
        tabs.add_child(human_setting)
        tabs.add_child(self.tracking_setting)
        # tabs.add_child(camera_setting)
        # tabs.add_tab('SMPL data', human_setting)
        # tabs.add_tab('Tracking tool', self.tracking_setting)
        # tabs.add_tab('Cameras', camera_setting)

        # collapse = gui.CollapsableVert("My settings", 0.33 * em,
        #                                 gui.Margins(em, 0, 0, 0))
        collapse = gui.Vert()

        # collapse.add_child(stream_setting)
        collapse.add_child(tabs)
        self.window.add_child(self.stream_setting)
        self.window.set_on_layout(self._on_setting_layout)

        self.tracking_setting.visible = False
        # self._settings_panel.add_child(collapse)
        self._settings_panel.get_children()[0].add_tab('Tools', collapse)
        # self._settings_panel.get_children()[0].add_tab('PCDs', self.remote_setting())
        self._settings_panel.get_children()[0].add_tab('Camera', camera_setting)

    def _on_setting_layout(self, layout_context):
        r = self.window.content_rect

        pref = self.stream_setting.calc_preferred_size(layout_context,
                                             gui.Widget.Constraints())
        play_btn_height = min(r.height, pref.height)

        setting_width = 17 * layout_context.theme.font_size
        setting_height = min(
            r.height - play_btn_height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        # self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

        bar_width = pref.width if r.width / 2 < pref.width else r.width/2
        bar_width = min(bar_width, r.width)
        # leftup x, leftup y, width, height

        def resize_1280(w, h):
            if h >= int(w/1280*720):
                h = int(w/1280*720)
            else:
                w = int(h/720*1280)
            return w, h

        if not self._settings_panel.visible:
            w, h = resize_1280(r.width, r.height - play_btn_height)
            self._scene.frame = gui.Rect(int((r.width - w)/2), r.y, w, h)
            self.stream_setting.frame = gui.Rect(
                r.get_left(), r.get_bottom() - pref.height, r.width, pref.height)
        else:
            w, h = resize_1280(r.width - setting_width, r.height - play_btn_height)
            self._scene.frame = gui.Rect(r.get_left() + setting_width, r.y, w, h)
            height = max(pref.height, r.get_bottom() - h)
            self.stream_setting.frame = gui.Rect(
                r.get_left() + setting_width, r.get_bottom() - height, r.width - setting_width, height)

        self._settings_panel.frame = gui.Rect(r.get_left(), r.y, setting_width, r.height)
        self._scene_traj.frame = gui.Rect(r.width * 3/4, r.y, r.width/4, r.height/4)
        # self.stream_setting.frame = gui.Rect(
        #     (r.width-bar_width)/2, r.get_bottom() - pref.height, bar_width, pref.height)

    def create_stream_settings(self):
        em = self.window.theme.font_size
        separation_height = int(round(0.5 * em))

        vert_layout = gui.Vert(0.15 * em)
        # collapse = gui.CollapsableVert("Data stream", 0.33 * em,
        #                                 gui.Margins(em, 0, 0, 0))

        play_btn     = creat_btn('  >>|| (Play / Stop)  ', self.change_pause_status, color = [0, 0, 0.5])

        frame_edit = gui.NumberEdit(gui.NumberEdit.INT)
        frame_edit.int_value = 0
        frame_edit.set_limits(0, 100)  # value coerced to 1
        frame_edit.set_on_value_changed(self._on_slider)
        
        text_frames = gui.TextEdit()
        text_frames.set_on_value_changed(self._on_slider)

        frame_slider = gui.Slider(gui.Slider.INT)
        frame_slider.set_limits(0, 1000)
        frame_slider.set_on_value_changed(self._on_slider)

        # horiz_layout = gui.Horiz(0.15 * em)
        # horiz_layout.add_child(intedit)
        # horiz_layout.add_stretch()
        # horiz_layout.add_child(gui.Label('frames'))
        # horiz_layout.add_child(text_frames)

        horiz_layout = gui.Horiz(0.15 * em)
        horiz_layout.add_child(frame_edit)
        # horiz_layout.add_child(minus_btn)
        horiz_layout.add_child(play_btn)
        horiz_layout.add_child(frame_slider)
        
        h2 = gui.Horiz(1 * em)
        # add_btn(h2, 'Follow camera', self._on_camera_view, True)
        # self.btn_rela_trans =  add_Switch(h2, 'Rela trans', self._on_free_view, False)
        self.archive_box = add_box(h2, 'Chess Board', self._on_show_geometry, True)
        h2.add_child(self._show_skybox)
        h2.add_child(self._show_axes)
        h2.add_child(self._show_ground_plane)
        add_Switch(h2, 'Render Img', self._change_render_states)
        self.btn_video_save = add_btn(h2, 'Save Video', self._click_video_saving)
        add_btn(h2, 'Freeze current frame', self._freeze_frame)

        vert_layout.add_child(horiz_layout)
        vert_layout.add_child(h2)
        # vert_layout.add_child(play_btn)


        frame_slider.enabled = False
        frame_edit.enabled = False
        play_btn.enabled = False

        self.frame_slider_bar, self.play_btn, self.frame_edit = frame_slider, play_btn, frame_edit

        return vert_layout

    def tracking_tool_setting(self):
        em = self.window.theme.font_size
        separation_height = int(round(0.5 * em))
        tracking_tool = gui.CollapsableVert("Tracking tool", 0.33 * em,
                                        gui.Margins(em, 0, 0, 0))
        # tracking_tool = gui.Vert(0.15 * em)
        tracking_tool.add_child(gui.Label("Ctrl-click to pick / Double click to delete"))
        tracking_tool.add_child(gui.Label("The camera will look at the point"))
        save_traj_btn = creat_btn('Save tracked traj', self._save_traj)
        text_step = gui.TextEdit()
        text_step.set_on_value_changed(self._set_tracking_step)
        text_step.text_value = str(Setting_panal.TRACKING_STEP)
        horiz = gui.Horiz(0.15 * em)
        horiz.add_child(gui.Label('Auto Jump Step'))
        horiz.add_child(text_step)
        # horiz.add_child(gui.Label('frames'))
        # horiz.add_child(text_frames)
        # horiz.add_child(save_traj_btn)

        # tracked_points = gui.Verts(0.15 * em)
        # tracked_points.background_color = gui.Color(r=0, b=0, g=0)
        # self.trackpoints_list = gui.CollapsableVert("Tracked points", 0.33 * em,
        #                                 gui.Margins(em, 0, 0, 0))
        self.trackpoints_list = gui.ListView()
        self.trackpoints_list.set_max_visible_items(5)
        self.trackpoints_list.set_on_selection_changed(self._on_track_list)

        # collapse.add_child(remote_layout)
        tracked_info = gui.Horiz(0.25 * em)
        tracked_info.add_child(gui.Label('Tracked label'))
        tracked_info.add_child(creat_btn('Clear', self._clear_3dlabel))
        tracking_tool.add_child(tracked_info)
        tracking_tool.add_child(self.trackpoints_list)
        tracking_tool.add_child(horiz)
        tracking_tool.add_child(save_traj_btn)

        return tracking_tool

    def _on_track_list(self, new_val, is_dbl_click):
        frame = int(new_val.split(':')[0])
        if is_dbl_click:
            self._scene.remove_3d_label(self.tracked_frame[frame][1])
            try:
                self._on_freeze_list(f'{frame-Setting_panal._START_FRAME_NUM}_freeze_{frame}_trkpts', True)
            except Exception as e:
                print(e)

            del self.tracked_frame[frame]
            self.update_tracked_points()
        else:
            self._on_slider(frame -  Setting_panal._START_FRAME_NUM)
            world = new_val.split(':')[1].split(',')
            try:
                world = self.COOR_INIT[:3, :3] @ np.array([float(v.strip().split(' ')[0]) for v in world])
                cam_to_select = world - self.get_camera_pos()
                eye = world - 2.5 * Setting_panal.SCALE * cam_to_select / np.linalg.norm(cam_to_select) 
                up = self.COOR_INIT[:3, :3] @ np.array([0, 0, 1])
                self._scene.look_at(world, eye, up)
            except Exception as e:
                print(e)

    def update_tracked_points(self):
        keys = sorted(list(self.tracked_frame.keys()))
        items = [self.tracked_frame[k][0] for k in keys]
        self.trackpoints_list.set_items(items)
        self.window.set_needs_layout()
        
    def update_tracked_cameras(self):
        # keys = sorted(list(self._camera_list))
        self.camera_list_view.set_items(list(self._camera_list))
        self.window.set_needs_layout()

    def add_freeze_data(self, name, geometry, mat):
        frameidx = self._get_slider_value()

        if not self._geo_list[name]['freeze']:
            fname = f'{frameidx}_freeze_{name}' 
        else:
            fname = name

        self._geo_list[fname] = {
            'geometry': geometry, 
            'type': self._geo_list[name]['type'], 
            'box': self.freeze_box,
            'mat': self._geo_list[name]['mat'], 
            'archive': self._geo_list[name]['archive'],
            'freeze': True}

        self.remove_geometry(fname)

        self._scene.scene.add_geometry(fname, geometry, mat)
        self._scene.scene.set_geometry_transform(fname, self.COOR_INIT)
        self.update_frozen_points()

    def _on_freeze_list(self, new_val, is_dbl_click):
        if is_dbl_click:
            self.remove_geometry(new_val)
            try:
                self._geo_list.pop(new_val)
            except Exception as e:
                print(e)
            self.update_frozen_points()
        else:
            print(new_val)
            
    def _on_camera_list(self, new_val, is_dbl_click):
        if is_dbl_click:
            self.remove_geometry(new_val)
            try:
                self._camera_list.pop(new_val)
            except Exception as e:
                print(e)
            self.update_tracked_cameras()
        else:
            try:
                ex = np.array(self._camera_list[new_val]['extrinsic']).reshape(4,4)
                self.init_camera(ex @ self.COOR_INIT, self._camera_list[new_val]['intrin_scale'])
            except Exception as e:
                self.warning_info(e.args[0])
            # print(new_val)

    def _load_camera_list(self):
        import json
        def load_json(path):
            self.window.close_dialog()
            try:
                with open(path, 'r') as fb:
                    self._camera_list.update(json.load(fb))
                self.update_tracked_cameras()
            except Exception as e:
                print(e.args[0])

        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose camera file to load",
                             self.window.theme)
        dlg.add_filter(
            ".json",
            "JSON files (.json)")

        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(load_json)
        self.window.show_dialog(dlg)

    
    def _save_came(self, path, is_print=True):
        import json
        try:
            for cam in self._camera_list:
                self._camera_list[cam]['extrinsic'] = np.array(self._camera_list[cam]['extrinsic']).tolist()
            with open(path, 'w') as fp:
                json.dump(self._camera_list, fp, indent=4)
            if is_print:
                self.warning_info(f'Camera list saved in {path}', 'INFO')
        except Exception as e:
            if is_print:
                self.warning_info(e.args[0])  

    def _save_camera_list(self):

        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".json", "JSON files (.json)")
        dlg.set_on_cancel(self.window.close_dialog)

        dlg.set_on_done(self._save_came)
        self.window.show_dialog(dlg)   

    
    def update_frozen_points(self):
        frozen_list = []
        for name in self._geo_list:
            if self._geo_list[name]['freeze']:
                frozen_list.append(name)
        self.frozen_list.set_items(frozen_list)
        self.window.set_needs_layout()

    def _list_dir(self, path):
        self.tracking_foler = path
        
        files = []
        try:
            self.data_loader = load_data_remote(False)
            files = self.data_loader.list_dir(path)
        except:
            try:
                password = self.remote_info['pwd'].strip()
                username = self.remote_info['username'].strip()
                hostname = self.remote_info['hostname'].strip()
                port = self.remote_info['port'].strip()
                self.data_loader = load_data_remote(True, username, hostname, int(port), password)
                files = self.data_loader.list_dir(path)
            except Exception as e:
                print(e)
                self.warning_info(f"'{path}' \n Not valid! Please input the right remote info!!!")

        if len(files) <=0:
            self.warning_info(f"'{path}' \n Not valid! Please input the right folder info!!!")
        
        return files

    def _loading_pcds(self, path=None):
        self._on_show_skybox(False)
        self._on_bg_color(gui.Color(0, 0, 0))

        if path is None:
            path = self.remote_info['folder'].strip()
        pcd_paths = self._list_dir(path)

        self.tracking_list = []

        for pcd_path in pcd_paths:
            if pcd_path.endswith('.pcd'):
                self.tracking_list.append(pcd_path)

        if len(self.tracking_list) > 0:
            self.tracking_list = sorted(self.tracking_list, key=lambda x: float(
                x.split('.')[0].replace('_', '.')))

            if not Setting_panal.PAUSE:
                self.change_pause_status()

            self.warning_info(f"Pcd loaded from '{path}'", type='info')
        
    
    def _set_tracking_step(self, value):
        Setting_panal.TRACKING_STEP = round(float(value))
    
    def _save_traj(self):
        pass
        
    def create_humandata_settings(self, collapse=None, checkboxes=None):
        em = self.window.theme.font_size
        separation_height = int(round(0.5 * em))

        if collapse is None:
            # collapse = gui.CollapsableVert("Geometries", 0.33 * em,
            #                                 gui.Margins(em, 0, 0, 0))
            collapse = gui.Vert(0.15 * em)
            geo_info = gui.Horiz(0.15 * em)
            geo_info.add_child(gui.Label("Geometries"))
            geo_info.add_child(creat_btn('Delete', lambda: self.dlg_yes_or_no(None, self._remove_geo_list)))
            collapse.add_fixed(separation_height)
            collapse.add_child(geo_info)
                                            
        factor_slider = gui.Slider(gui.Slider.INT)
        factor_slider.set_limits(2, 60)
        factor_slider.int_value = 10
        factor_slider.set_on_value_changed(self._on_factor_slider)

        scale_slider = gui.Slider(gui.Slider.INT)
        scale_slider.set_limits(1, 20)
        scale_slider.int_value = 1
        scale_slider.set_on_value_changed(self._on_scale_slider)

        clear_freeze_btn   = creat_btn('Clear', self._clear_freeze)

        # tabs = gui.TabControl()
        tabs = gui.Vert(0.15 * em)
        # check_boxes = gui.VGrid(2, 0.15 * em)
        data_list = gui.Horiz(0.15 * em)
        data_list.preferred_height = 25 * em
        self.geo_check_boxes = gui.TreeView()
        data_list.add_child(self.geo_check_boxes)
        self.geo_check_boxes.set_on_selection_changed(self._on_tree)
        try:
            # self.freeze_box = add_box(check_boxes, 'frozen data', self._on_show_freeze_geometry, True)
            for box in checkboxes:
                hh = gui.Horiz()
                hh.add_child(box)
                add_btn(hh, 'Set', self._on_material_setting, True)
                self.geo_check_boxes.add_item(self.geo_check_boxes.get_root_item(), hh)
        except:
            pass

        cameras = create_combobox(self._on_select_camera)
        self._fix_roll = gui.ToggleSwitch('Camera Stabilization')
        cam_grid = gui.VGrid(2, 0.25 * em)
        cam_grid.add_child(gui.Label('POV'))
        cam_grid.add_child(cameras)
        cam_grid.add_child(gui.Label('Focal factor'))
        cam_grid.add_child(factor_slider)
        cam_grid.add_child(gui.Label('Geometry scale'))
        cam_grid.add_child(scale_slider)
        # cam_grid.add_child(gui.Label('Stabilization'))
        # cam_grid.add_child(self._fix_roll)
        cam_grid_tool = gui.VGrid(3, 0.15 * em)
        cam_grid_tool.add_child(creat_btn('Left +10°', self.camera_fix))
        cam_grid_tool.add_child(creat_btn('Up +10°', self.camera_fix))
        cam_grid_tool.add_child(creat_btn('Roll +10°', self.camera_fix))
        cam_grid_tool.add_child(creat_btn('Left -10°', self.camera_fix))
        cam_grid_tool.add_child(creat_btn('Up -10°', self.camera_fix))
        cam_grid_tool.add_child(creat_btn('Roll -10°', self.camera_fix))
        cam_grid_tool.add_child(creat_btn('Level', self.camera_fix))
        cam_grid_tool.add_child(creat_btn('Look forward', lambda: self.camera_fix('forward')))
        cam_grid_tool.add_child(creat_btn('Look Down', lambda: self.camera_fix('down')))
        # cam_grid.add_child(scale_slider)

        # horz = gui.Horiz(0.25 * em)
        # add_Switch(cam_grid, 'Follow camera', self._on_camera_view, True)
        self.camera_list_view = gui.ListView()
        self.camera_list_view.set_max_visible_items(15)
        self.camera_list_view.set_on_selection_changed(self._on_camera_list)

        tab2 = gui.Vert(0.25 * em)
        # tab2 = gui.CollapsableVert("Cameras", 0.33 * em,
        #                                 gui.Margins(em, 0, 0, 0))
        tab2.add_child(cam_grid)
        tab2.add_child(cam_grid_tool)
        tab2.add_child(gui.Label('Camera lists') )
        tab2.add_child(self.camera_list_view)
        _click_camera = lambda: self.pop_up_text('Input a name', lambda name: self._click_camera_saving(name, True))
        tab2.add_child(creat_btn('Add camera', _click_camera))

        tab2.add_fixed(separation_height)
        tab2.add_child(creat_btn('Export cameras', self._save_camera_list))
        tab2.add_child(creat_btn('Load cameras', self._load_camera_list))

        tab2.add_fixed(separation_height)
        tab2.add_child(self._fix_roll)
        add_Switch(tab2, 'Camera Following the POV', self._on_camera_view, True)
        self.btn_freeview = add_box(tab2, 'Enable Free Viewpoint', self._on_free_view, False)
        self.btn_rela_trans =  add_box(tab2, 'Lock rotation', lambda check: check, False)

        # tab2.add_child(horz)

        freeze_info = gui.Horiz(0.25 * em)
        # temp_layout.add_child(freeze_btn)
        self.freeze_box = add_box(freeze_info, 'Show Frozen Data', self._on_show_geometry, True)
        freeze_info.add_child(clear_freeze_btn)

        self.frozen_list = gui.ListView()
        self.frozen_list.set_max_visible_items(5)
        self.frozen_list.set_on_selection_changed(self._on_freeze_list)

        freeze_data_layout = gui.Vert(0.15 * em)
        freeze_data_layout.add_child(freeze_info)
        freeze_data_layout.add_child(self.frozen_list)

        tabs.add_child(data_list)
        tabs.add_fixed(separation_height)

        # tabs.add_child(gui.Label('Frozen data'))
        tabs.add_child(freeze_data_layout)
        tabs.add_fixed(separation_height)

        collapse.add_child(tabs)

        self.camera_setting = cameras
        self.camera_setting.enabled = False

        return collapse, tab2

    def _on_scale_slider(self, value):
        pre_scale = Setting_panal.SCALE
        Setting_panal.SCALE = int(value)
        for name, g in self._geo_list.items():
            g['geometry'].scale(1/pre_scale, (0.0, 0.0, 0.0))
            # g['geometry'].rotate(self.COOR_INIT[:3, :3].T, self.COOR_INIT[:3, 3])
            self.update_geometry(g['geometry'], name)

    def _on_camera_view(self, show):
        self.btn_freeview.visible = show
        self.btn_rela_trans.visible = show and self.btn_freeview.checked
        Setting_panal.FIX_CAMERA = not show
        self.window.set_needs_layout()

    def _on_free_view(self, show):
        self.btn_rela_trans.visible = show and not Setting_panal.FIX_CAMERA
        self.window.set_needs_layout()

    def _on_tree(self, new_item_id):
        pass
        # print(new_item_id)

    def _on_select_camera(self, name, index):
        Setting_panal.POV = name
        Setting_panal.CLICKED = True

    def _clear_freeze(self):
        nlist = [k for k in self._geo_list]
        for name in nlist:
            if self._geo_list[name]['freeze']:
                self.remove_geometry(name)

                self._geo_list.pop(name)

        self.update_frozen_points()

        
    def _clear_3dlabel(self):
        for frame in self.tracked_frame:
            self._scene.remove_3d_label(self.tracked_frame[frame][1])
        self.tracked_frame.clear()
        self.update_tracked_points()
        
    def _freeze_frame(self):
        Setting_panal.FREEZE = True
        Setting_panal.CLICKED = True
    
    def _unfreeze(self):
        Setting_panal.FREEZE = False

    def _remove_geo_list(self, name=''):
        self.geo_check_boxes.remove_item(self.geo_check_boxes.selected_item)
        if self._selected_geo in self._geo_list:
            self._geo_list.pop(self._selected_geo)
        self.remove_geometry(self._selected_geo)
        self.window.close_dialog()


    def _show_geo_by_name(self, name, show):
        self._selected_geo = name
        for geo_name, geo_data in self._geo_list.items():
            if name in geo_name:
                if geo_data['freeze'] == True:
                    self._scene.scene.show_geometry(geo_name, geo_data['box'].checked and show)
                    self._scene_traj.scene.show_geometry(geo_name, geo_data['box'].checked and show)
                else:
                    self._scene.scene.show_geometry(geo_name, show)
                    self._scene_traj.scene.show_geometry(geo_name, show)

    def _on_show_geometry(self, show):
        for name, data in self._geo_list.items():
            self._scene.scene.show_geometry(name, data['box'].checked)
            self._scene_traj.scene.show_geometry(name, data['box'].checked)
            if data['freeze'] == True and data['box'].checked:
                origin_name = name.split('_freeze_')[-1]
                box = self._geo_list[origin_name]['box']
                self._scene.scene.show_geometry(name, box.checked)
                self._scene_traj.scene.show_geometry(name, box.checked)
        # self._apply_settings()

    def _add_frame(self):
        self._on_slider(self.frame_slider_bar.int_value+1)
        
    def _minus_frame(self):
        if self.frame_slider_bar.int_value > 0:
            self._on_slider(self.frame_slider_bar.int_value-1)

    def _change_render_states(self, is_render):
        Setting_panal.RENDER = is_render
        
    def _click_video_saving(self):
        Setting_panal.VIDEO_SAVE = True

    def _click_camera_saving(self, cam_name=None, is_print=True):
        extrinsic = self._scene.scene.camera.get_view_matrix()
        extrinsic[1, :] = - extrinsic[1, :]
        extrinsic[2, :] = - extrinsic[2, :]
        extrinsic[:3, 3] /= self.SCALE 

        extrinsic = extrinsic @ np.linalg.inv(self.COOR_INIT)
        cam_pose = self._scene.scene.camera.get_model_matrix()
        cam_pose[:3, 3] /= self.SCALE 
        cam_pose = self.COOR_INIT @ cam_pose
        if is_print:
            quat = R.from_matrix(cam_pose[:3, :3]).as_quat()
            print(f'quat: {quat}\n Trans: {cam_pose[:3, 3]}')
            self.warning_info(f'quat: {quat}\n Trans: {cam_pose[:3, 3]}', type='Extrinsic')
        frame = self._get_slider_value()
        if cam_name is None:
            cam_name = f'cam_{frame}'
        while cam_name in self._camera_list:
            cam_name = f'cam_{frame+1}_manual'
            frame +=1
        self._camera_list[cam_name] = {'extrinsic': extrinsic, 
                                       'intrin_scale': Setting_panal.INTRINSIC_FACTOR}
        self.update_tracked_cameras()
        return extrinsic, cam_pose

    def _clicked(self):
        Setting_panal.CLICKED = False

    def _set_slider_value(self, value):
        self.frame_slider_bar.int_value = int(value)
        self.frame_edit.int_value = int(value)
        
    def _get_slider_value(self):
        return self.frame_slider_bar.int_value
        
    def _get_max_slider_value(self):
        return self.frame_slider_bar.get_maximum_value
        
    def _set_slider_limit(self, min, max):
        self.frame_slider_bar.set_limits(min, max)
        self.frame_edit.set_limits(min, max)

    def _on_FPV(self, show):
        Setting_panal.POV = 'first' if show else 'second'
        Setting_panal.CLICKED = True

    def change_pause_status(self):
        Setting_panal.PAUSE = not Setting_panal.PAUSE
        text = '||' if not Setting_panal.PAUSE else '|>>'
        color = gui.Color(r=0.5, b=0, g=0) if Setting_panal.PAUSE else gui.Color(r=0, b=0, g=0.5)
        self.play_btn.text = f'        {text}        '
        self.play_btn.background_color = color

    def _on_slider(self, value):
        self.frame_slider_bar.int_value = int(value)
        self.frame_edit.int_value = int(value)
        if not Setting_panal.PAUSE:
            self.change_pause_status()
        Setting_panal.CLICKED = True

    def _on_factor_slider(self, value):
        Setting_panal.INTRINSIC_FACTOR = value/10
        # Setting_panal.CLICKED = True
        self.init_camera()
    
    def init_camera(self):
        pass

    def _add_text(self, visible=False):

        info = gui.Label("Frames")
        info.visible = visible

        def _on_tex_layout(layout_context):
            r = self.window.content_rect
            self._scene.frame = r
            pref = info.calc_preferred_size(layout_context, gui.Widget.Constraints())
            info.frame = gui.Rect(r.x, r.get_bottom() - pref.height, pref.width, pref.height)

        self.window.set_on_layout(_on_tex_layout)
        self.window.add_child(info)
        return info

    def add_thread(self, thread):
        self.close_thread()
        Setting_panal.MYTHREAD = thread
        Setting_panal.MYTHREAD.start()

    def close_thread(self):
        if Setting_panal.MYTHREAD is not None:
            stop_thread(Setting_panal.MYTHREAD)

    def reset_settings(self):
        Setting_panal.IMG_COUNT = 0
        Setting_panal.VIDEO_SAVE = False
        # Setting_panal.PAUSE = False
        # Setting_panal.POV = 'first'
        # Setting_panal.RENDER = False
        # self._set_slider_value(0)

    def thread(self):
        """
        The function is a thread that runs in the background and updates the data in the GUI

        > Youn can define your `fetch_data` and `update_data` functions here
        """
        initialized = False
        self._set_slider_limit(0, self.total_frames - 1)
        def set_video_name():
            try:
                video_name = self.scene_name + time.strftime("-%Y-%m-%d_%H-%M", time.localtime())
            except Exception as e:
                video_name = 'test' + time.strftime("-%Y-%m-%d_%H-%M", time.localtime())
            return video_name

        def save_video(image_dir, video_name, delete=True):
            try:
                is_success, info = images_to_video(image_dir, video_name, delete=delete, inpu_fps=20)
                came_path = os.path.join((os.path.dirname(image_dir)), 'vis_data', video_name+'.json')
                if is_success:
                    self._save_came(came_path, is_print=False)
                video_name = set_video_name()
                self.warning_info(info, type='info')
            except Exception as e:
                self.warning_info(e.args[0])

            Setting_panal.VIDEO_SAVE = False
            Setting_panal.IMG_COUNT = 0
            
            if not Setting_panal.PAUSE:
                self.change_pause_status()
            return video_name

        while True:
            video_name = set_video_name()
            image_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f'temp_{video_name}')

            self.reset_settings()
            self._set_slider_value(0)

            while self._get_slider_value() < self.total_frames - 1:
                index = self._get_slider_value()
                data = self.fetch_data(index)
                self.update_data(data, initialized)
                time.sleep(0.02)
                try:
                    self.set_camera(index, index-1, Setting_panal.POV)
                except Exception as e:
                    print(e)

                initialized = True

                if Setting_panal.RENDER and not Setting_panal.PAUSE:
                    gui.Application.instance.post_to_main_thread(self.window, lambda: self.save_imgs(image_dir))
                    time.sleep(0.02)

                if Setting_panal.VIDEO_SAVE:
                    video_name = save_video(image_dir, video_name)

                while True:
                    time.sleep(0.01)
                    if Setting_panal.CLICKED:
                        rule = (index != self._get_slider_value() or Setting_panal.FREEZE)
                        if rule:
                            new_index = self._get_slider_value()
                            data = self.fetch_data(new_index)
                            self.update_data(data)
                            time.sleep(0.01)
                        else: 
                            new_index = index
                        try:
                            self.set_camera(new_index, index-1, Setting_panal.POV)
                        except Exception as e:
                            print(e)
                        index = new_index
                        self._clicked()

                    if Setting_panal.VIDEO_SAVE:
                        video_name = save_video(image_dir, video_name)

                    if not Setting_panal.PAUSE:
                        break
                    
                self._set_slider_value(index+1)

            if Setting_panal.RENDER:
                video_name = save_video(image_dir, video_name, delete=True)

            self._on_slider(0)

    def save_imgs(self, img_dir):
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_path = os.path.join(img_dir, f'{Setting_panal.IMG_COUNT:05d}.png')
        self._click_camera_saving(f'VIDEO_{Setting_panal.IMG_COUNT:05d}', is_print=False)
        Setting_panal.IMG_COUNT += 1
        self.export_image(img_path, 1920, 1080)

    def set_camera(self, new_ind, ind, pov):
        pass

    def get_tracking_data(self, index):
        geomety = self.data_loader.load_point_cloud(self.tracking_foler + '/' + self.tracking_list[index])
        return geomety
        
    def fetch_data(self, index):
        # your function here
        name = 'Tracking frame'
        geometry = self.get_tracking_data(index)
        if len(geometry.points) > 0 and name not in self._geo_list:
            self.make_material(geometry, name, 'point', is_archive=False)
            self._geo_list[name]['mat'].material.point_size = 8
        return {name: geometry}

    def update_data(self, data, initialized=True):
        def func():
            # your function here
            for name in data:
                self.update_geometry(data[name], name, reset_bounding_box=False, freeze=Setting_panal.FREEZE)
            self.window.set_needs_layout()
            self._unfreeze()
        gui.Application.instance.post_to_main_thread(self.window, func)

        if not initialized:
            # your function here

            self.change_pause_status()

def _async_raise(tid, exctype):
    import ctypes
    import inspect

    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


def main():
    gui.Application.instance.initialize()

    w = Setting_panal(1080, 720)

    gui.Application.instance.run()

if __name__ == "__main__":
    main()
