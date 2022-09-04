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

import open3d.visualization.gui as gui
import sys
import time
import cv2

sys.path.append('.')
sys.path.append('..')
from .base_gui import AppWindow as GUI_BASE

def create_combobox(func, names=None):
    combobox = gui.Combobox()
    if names is not None:
        for name in names:
            combobox.add_item(name)
    combobox.set_on_selection_changed(func)
    return combobox

def creat_btn(name, func, color=None):
    btn = gui.Button(name)
    btn.horizontal_padding_em = 0.2
    btn.vertical_padding_em = 0
    if color is not None:
        btn.background_color = gui.Color(r=color[0], b=color[1], g=color[2])
    btn.set_on_clicked(func)
    return btn

def add_Switch(layout, name, func, checked=False):
    switch = gui.ToggleSwitch(name)
    switch.set_on_clicked(func)
    switch.is_on = checked
    layout.add_child(switch)

def add_box(layout, name, func, checked=False):
    box = gui.Checkbox(name)
    box.set_on_checked(func)
    box.checked = checked
    layout.add_child(box)

class Setting_panal(GUI_BASE):
    TRACKING_STEP = 50
    FREEZE = False
    FREE_VIEW = False
    FIX_CAMERA = False
    PAUSE = False
    POV = 'first'
    RENDER = False
    CLICKED = False
    INTRINSIC_FACTOR = 1
    SCALE = 1
    MYTHREAD = None

    def __init__(self, width=1280, height=720):
        super(Setting_panal, self).__init__(width, height)
        self.archive_data = []
        self.freeze_data = []
        em = self.window.theme.font_size
        self.tracked_frame = {}

        stream_setting = self.create_stream_settings()
        human_setting, camera_setting = self.create_humandata_settings()
        self.tracking_setting = self.tracking_tool_setting()
        
        tabs = gui.TabControl()
        tabs.add_tab('SMPL data', human_setting)
        tabs.add_tab('Tracking tool', self.tracking_setting)
        tabs.add_tab('Cameras', camera_setting)

        collapse = gui.CollapsableVert("My settings", 0.33 * em,
                                        gui.Margins(em, 0, 0, 0))
        collapse.add_child(stream_setting)
        collapse.add_child(tabs)

        self.tracking_setting.visible = False
        self._settings_panel.add_child(collapse)

    def create_stream_settings(self):
        em = self.window.theme.font_size
        separation_height = int(round(0.5 * em))

        collapse = gui.Vert(0.15 * em)
        # collapse = gui.CollapsableVert("Data stream", 0.33 * em,
        #                                 gui.Margins(em, 0, 0, 0))

        minus_btn    = creat_btn('-', self._minus_frame)
        add_btn      = creat_btn('+', self._add_frame)
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

        prog_layout = gui.Horiz(0.15 * em)
        prog_layout.add_child(frame_edit)
        # prog_layout.add_child(minus_btn)
        # prog_layout.add_child(add_btn)
        prog_layout.add_child(frame_slider)
        
        # collapse.add_child(horiz_layout)
        collapse.add_child(prog_layout)
        collapse.add_child(play_btn)
        add_Switch(collapse, 'Auto Render', self.change_render_states)

        frame_slider.enabled = False
        frame_edit.enabled = False
        play_btn.enabled = False
        self.frame_slider_bar, self.play_btn, self.frame_edit = frame_slider, play_btn, frame_edit

        return collapse

    def tracking_tool_setting(self):
        self.remote_info = {}
        em = self.window.theme.font_size
        separation_height = int(round(0.5 * em))
        # collapse = gui.CollapsableVert("Traking tool", 0.33 * em,
        #                                 gui.Margins(em, 0, 0, 0))
        collapse = gui.Vert(0.15 * em)
                                        
        self._fileedit = gui.TextEdit()
        self._fileedit.set_on_value_changed(self._start_tracking)
        username = gui.TextEdit()
        hostname = gui.TextEdit()
        port = gui.TextEdit()
        self._fileedit.text_value = '/hdd/dyd/lidarhumanscene/data/0623/003/tracking_data/segment_by_tracking'
        username.text_value = 'dyd'
        hostname.text_value = '10.24.80.241'
        port.text_value = '911'

        filedlgbutton = creat_btn('Folder', self._on_traj_folder)

        self.remote_info['username'] = username
        self.remote_info['hostname'] = hostname
        self.remote_info['port'] = port
        self.remote_info['folder'] = self._fileedit

        remote_layout = gui.VGrid(2, 0.15 * em)
        remote_layout.add_child(gui.Label('user'))
        remote_layout.add_child(username)
        remote_layout.add_child(gui.Label('host'))
        remote_layout.add_child(hostname)
        remote_layout.add_child(gui.Label('port'))
        remote_layout.add_child(port)
        remote_layout.add_child(filedlgbutton)
        remote_layout.add_child(self._fileedit)

        btn = creat_btn('Save traj', self._save_traj)
        text_step = gui.TextEdit()
        text_step.set_on_value_changed(self._set_tracking_step)
        text_step.text_value = str(Setting_panal.TRACKING_STEP)
        horiz = gui.Horiz(0.15 * em)
        horiz.add_child(gui.Label('step'))
        horiz.add_child(text_step)
        # horiz.add_child(gui.Label('frames'))
        # horiz.add_child(text_frames)
        horiz.add_child(btn)

        # tracked_points = gui.Verts(0.15 * em)
        # tracked_points.background_color = gui.Color(r=0, b=0, g=0)
        # self.trackpoints_list = gui.CollapsableVert("Tracked points", 0.33 * em,
        #                                 gui.Margins(em, 0, 0, 0))
        self.trackpoints_list = gui.ListView()
        self.trackpoints_list.set_max_visible_items(5)
        self.trackpoints_list.set_on_selection_changed(self._on_track_list)

        collapse.add_child(remote_layout)
        collapse.add_child(horiz)
        collapse.add_child(gui.Label('Tracked points'))
        collapse.add_child(self.trackpoints_list)

        return collapse

    def _on_track_list(self, new_val, is_dbl_click):
        frame = int(new_val.split(':')[0])
        if is_dbl_click:
            self._scene.remove_3d_label(self.tracked_frame[frame][1])
            del self.tracked_frame[frame]
            self.update_tracked_points()
        else:
            self._on_slider(frame)

    def update_tracked_points(self):
        keys = sorted(list(self.tracked_frame.keys()))
        items = [self.tracked_frame[k][0] for k in keys]
        self.trackpoints_list.set_items(items)
        self.window.set_needs_layout()

    def add_freeze_data(self, name, geometry, mat, geometry_type):
        ss = time.time()
        frameidx = self._get_slider_value()
        fname = f'{frameidx}_freeze_{name}'
        if not self._scene.scene.has_geometry(fname):
            self.freeze_data.append(fname)
            self._scene.scene.add_geometry(fname, geometry, mat)
            if geometry_type == 'point':
                self.point_list[fname] = geometry
            elif geometry_type == 'mesh':
                self.mesh_list[fname] = geometry
            self.update_freezed_points()

    def _on_freeze_list(self, new_val, is_dbl_click):
        if is_dbl_click:
            if self._scene.scene.has_geometry(new_val):
                self._scene.scene.remove_geometry(new_val)
            try:
                self.freeze_data.remove(new_val)
            except Exception as e:
                print(e)
            self.update_freezed_points()
        else:
            print(new_val)

    def update_freezed_points(self):
        self.freezed_list.set_items(self.freeze_data)
        self.window.set_needs_layout()

    def _start_tracking(self, path):
        pass
    
    def _set_tracking_step(self, value):
        Setting_panal.TRACKING_STEP = round(float(value))
    
    def _save_traj(self):
        pass

    def _on_traj_folder(self):
        filedlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Select file",
                                 self.window.theme)
        filedlg.add_filter(".obj .ply .stl", "Triangle mesh (.obj, .ply, .stl)")
        filedlg.add_filter("", "All files")
        filedlg.set_on_cancel(self._on_file_dialog_cancel)
        filedlg.set_on_done(self._on_filedlg_done)
        self.window.show_dialog(filedlg)
        
    def _on_filedlg_done(self, path):
        self._fileedit.text_value = path
        self.window.close_dialog()
        self._start_tracking(path)
        
    def create_humandata_settings(self, collapse=None, checkboxes=None):
        em = self.window.theme.font_size
        separation_height = int(round(0.5 * em))

        if collapse is None:
            # collapse = gui.CollapsableVert("Human data", 0.33 * em,
            #                                 gui.Margins(em, 0, 0, 0))
            collapse = gui.Vert(0.15 * em)
                                            
        factor_slider = gui.Slider(gui.Slider.INT)
        factor_slider.set_limits(2, 60)
        factor_slider.int_value = 10
        factor_slider.set_on_value_changed(self._on_factor_slider)

        scale_slider = gui.Slider(gui.Slider.INT)
        scale_slider.set_limits(1, 20)
        scale_slider.int_value = 1
        scale_slider.set_on_value_changed(self._on_scale_slider)

        freeze_btn   = creat_btn('Freeze', self._freeze_frame)
        clear_freeze_btn   = creat_btn('Clear', self._clear_freeze)

        # tabs = gui.TabControl()
        tabs = gui.Vert(0.15 * em)
        tab1 = gui.VGrid(2, 0.15 * em)

        try:
            add_box(tab1, 'Chess Board', self._on_show_archive_geometry, True)
            for box in checkboxes:
                tab1.add_child(box)
        except:
            pass

        cameras = create_combobox(self._on_select_camera)
        cam_grid = gui.VGrid(2, 0.25 * em)
        cam_grid.add_child(gui.Label('POV'))
        cam_grid.add_child(cameras)
        cam_grid.add_child(gui.Label('Focal factor'))
        cam_grid.add_child(factor_slider)
        cam_grid.add_child(gui.Label('Geometry scale'))
        cam_grid.add_child(scale_slider)

        # horz = gui.Horiz(0.25 * em)
        add_Switch(cam_grid, 'Follow camera', self._on_camera_view, True)
        add_Switch(cam_grid, 'Only Trans', self._on_free_view, False)

        tab2 = gui.Vert(0.25 * em)
        tab2.add_child(cam_grid)
        # tab2.add_child(horz)

        temp_layout = gui.Horiz(0.15 * em)
        temp_layout.add_child(freeze_btn)
        add_box(temp_layout, 'Show', self._on_show_freeze_geometry, True)
        temp_layout.add_child(clear_freeze_btn)

        self.freezed_list = gui.ListView()
        self.freezed_list.set_max_visible_items(5)
        self.freezed_list.set_on_selection_changed(self._on_freeze_list)

        tab3 = gui.Vert(0.15 * em)
        tab3.add_child(temp_layout)
        tab3.add_child(self.freezed_list)

        tabs.add_child(gui.Label('Show Data'))
        tabs.add_child(tab1)
        tabs.add_fixed(separation_height)
        tabs.add_child(gui.Label('Freezed data'))
        tabs.add_child(tab3)
        tabs.add_fixed(separation_height)

        # tabs.add_tab("Render Option", tab3)
        # tabs.add_tab("Show Data", tab1)
        # tabs.add_tab("Cameras", tab2)

        collapse.add_child(tabs)

        self.check_boxes, self.camera_setting = tab1, cameras
        self.camera_setting.enabled = False

        return collapse, tab2

    def _on_scale_slider(self, value):
        pre_scale = Setting_panal.SCALE
        Setting_panal.SCALE = int(value)
        for name, g in self.point_list.items():
            g.scale(1/pre_scale, (0.0, 0.0, 0.0))
            g.rotate(self.COOR_INIT[:3, :3].T, self.COOR_INIT[:3, 3])
            self.update_geometry(g, name)
        for name, g in self.mesh_list.items():
            g.scale(1/pre_scale, (0.0, 0.0, 0.0))
            g.rotate(self.COOR_INIT[:3, :3].T, self.COOR_INIT[:3, 3])
            self.update_geometry(g, name)

    def _on_camera_view(self, show):
        Setting_panal.FIX_CAMERA = not show

    def _on_select_camera(self, name, index):
        Setting_panal.POV = name
        Setting_panal.CLICKED = True

    def _clear_freeze(self):
        for name in self.freeze_data:
            self._scene.scene.remove_geometry(name)
            self.freeze_data.remove(name)
        self.update_freezed_points()

    def _on_free_view(self, show):
        Setting_panal.FREE_VIEW = show
        # print(show)
        
    def _freeze_frame(self):
        Setting_panal.FREEZE = True
        Setting_panal.CLICKED = True
    
    def _unfreeze(self):
        Setting_panal.FREEZE = False

    def _on_show_archive_geometry(self, show):
        for name in self.archive_data:
            self._scene.scene.show_geometry(name, show)

    def _on_show_freeze_geometry(self, show):
        for name in self.freeze_data:
            self._scene.scene.show_geometry(name, show)

    def _add_frame(self):
        self._on_slider(self.frame_slider_bar.int_value+1)
        
    def _minus_frame(self):
        if self.frame_slider_bar.int_value > 0:
            self._on_slider(self.frame_slider_bar.int_value-1)

    def change_render_states(self, render):
        Setting_panal.RENDER = render

    def _clicked(self):
        Setting_panal.CLICKED = False

    def _set_slider_value(self, value):
        self.frame_slider_bar.int_value = int(value)
        self.frame_edit.int_value = int(value)
        
    def _get_slider_value(self):
        return self.frame_slider_bar.int_value
        
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
        Setting_panal.CLICKED = True
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

    def thread(self):
        """
        The function is a thread that runs in the background and updates the data in the GUI

        > Youn can define your `fecth_data` and `update_data` functions here
        """
        init_param = False
        self._set_slider_limit(0, self.total_frames - 1)
        self._set_slider_value(0)
        while True:
            while self._get_slider_value() < self.total_frames - 1:
                index = self._get_slider_value()
                data = self.fecth_data(index)
                self.update_data(data, init_param)
                time.sleep(0.05)
                init_param = True

                while True:
                    cv2.waitKey(5)
                    if Setting_panal.CLICKED:
                        rule = (index != self._get_slider_value())
                        if rule:
                            index = self._get_slider_value()
                            data = self.fecth_data(index)
                            self.update_data(data)
                            time.sleep(0.05)
                        self._clicked()

                    if not Setting_panal.PAUSE:
                        break

                self._set_slider_value(index+1)

            self._on_slider(0)

    def fetch_data(self, index):
        # your function here
        data = None
        return data
    
    def update_data(self, data, init_param=True):
        def func():
            # your function here
            pass
        gui.Application.instance.post_to_main_thread(self.window, func)
        if not init_param:
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
