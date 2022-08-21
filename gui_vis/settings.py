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
sys.path.append('.')
sys.path.append('..')
from .menu import Menu

def create_combobox(func, names=None):
    combobox = gui.Combobox()
    if names is not None:
        for name in names:
            combobox.add_item(name)
    combobox.set_on_selection_changed(func)
    return combobox

def creat_btn(name, func, color=None):
    btn = gui.Button(name)
    btn.horizontal_padding_em = 0.5
    btn.vertical_padding_em = 0
    if color is not None:
        btn.background_color = gui.Color(r=color[0], b=color[1], g=color[2])
    btn.set_on_clicked(func)
    return btn

def add_box(layout, name, func, checked=False):
    box = gui.Checkbox(name)
    box.set_on_checked(func)
    box.checked = checked
    layout.add_child(box)

class Setting_panal(Menu):
    FREEZE = False
    FREE_VIEW = False
    FIX_CAMERA = False
    # FRAME = 0
    PAUSE = False
    POV = 'first'
    RENDER = False
    CLICKED = False
    INTRINSIC_FACTOR = 1
    # PLAY_ONCE = False

    def __init__(self, width=1280, height=720):
        super(Setting_panal, self).__init__(width, height)
        self.archive_data = []
        self.freeze_data = []
        self.create_checkboxes()

    def create_checkboxes(self, collapse=None, checkboxes=None):
        em = self.window.theme.font_size
        separation_height = int(round(0.5 * em))

        if collapse is None:
            collapse = gui.CollapsableVert("Human data", 0.33 * em,
                                            gui.Margins(em, 0, 0, 0))
        frame_slider = gui.Slider(gui.Slider.INT)
        frame_slider.set_limits(0, 1000)
        frame_slider.set_on_value_changed(self._on_slider)
        
        factor_slider = gui.Slider(gui.Slider.INT)
        factor_slider.set_limits(2, 60)
        factor_slider.int_value = 10
        factor_slider.set_on_value_changed(self._on_factor_slider)

        minus_btn    = creat_btn('-', self._minus_frame)
        add_btn      = creat_btn('+', self._add_frame)
        play_btn     = creat_btn('    >>||    ', self.change_pause_status, color = [0, 0, 0.5])
        freeze_btn   = creat_btn('Freeze frame', self._freeze_frame)
        clear_freeze_btn   = creat_btn('Clear freezed data', self._clear_freeze)

        prog_layout = gui.Horiz(0.15 * em)
        prog_layout.add_child(minus_btn)
        prog_layout.add_child(add_btn)
        prog_layout.add_child(frame_slider)

        horiz_layout = gui.Horiz(em)
        horiz_layout.add_child(play_btn)
        add_box(horiz_layout, 'Auto Render', self.change_render_states)
        # horiz_layout.add_stretch()

        tabs = gui.TabControl()
        tab1 = gui.Vert(0.15 * em)
        tab1.add_fixed(separation_height)

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

        tab2 = gui.Vert(0.25 * em)
        tab2.add_fixed(separation_height)
        tab2.add_child(cam_grid)
        add_box(tab2, 'Relative view', self._on_free_view)
        add_box(tab2, 'Fixed camera', self._on_fix_view)

        tab3 = gui.Vert(0.15 * em)
        tab3.add_fixed(separation_height)
        add_box(tab3, 'Show freezed data', self._on_show_freeze_geometry, True)
        temp_layout = gui.Horiz(0.15 * em)
        temp_layout.add_child(freeze_btn)
        temp_layout.add_child(clear_freeze_btn)
        tab3.add_child(temp_layout)

        tabs.add_tab("Show Data", tab1)
        tabs.add_tab("Cameras", tab2)
        tabs.add_tab("Freeze Option", tab3)

        collapse.add_child(prog_layout)
        collapse.add_child(horiz_layout)
        collapse.add_child(tabs)

        self._settings_panel.add_child(collapse)

        self.check_boxes, self.camera_setting, self.frame_slider_bar, self.play_btn = tab1, cameras, frame_slider, play_btn

    def _on_fix_view(self, show):
        Setting_panal.FIX_CAMERA = show

    def _on_select_camera(self, name, index):
        Setting_panal.POV = name
        Setting_panal.CLICKED = True

    def _clear_freeze(self):
        for name in self.freeze_data:
            self._scene.scene.remove_geometry(name)

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
        self.frame_slider_bar.int_value += 1
        self._on_slider(self.frame_slider_bar.int_value)
        
    def _minus_frame(self):
        if self.frame_slider_bar.int_value > 0:
            self.frame_slider_bar.int_value -= 1
            self._on_slider(self.frame_slider_bar.int_value)

    def change_render_states(self, render):
        Setting_panal.RENDER = render

    def _clicked(self):
        Setting_panal.CLICKED = False


    def _set_slider_value(self, value):
        self.frame_slider_bar.int_value = int(value)
        
    def _get_slider_value(self):
        return self.frame_slider_bar.int_value
        
    def _set_slider_limit(self, min, max):
        self.frame_slider_bar.set_limits(min, max)

    def _on_FPV(self, show):
        Setting_panal.POV = 'first' if show else 'second'
        Setting_panal.CLICKED = True

    def change_pause_status(self):
        Setting_panal.PAUSE = not Setting_panal.PAUSE
        text = '||' if not Setting_panal.PAUSE else '|>>'
        color = gui.Color(r=0.5, b=0, g=0) if Setting_panal.PAUSE else gui.Color(r=0, b=0, g=0.5)
        self.play_btn.text = f'    {text}    '
        self.play_btn.background_color = color

    def _on_slider(self, value):
        if not Setting_panal.PAUSE:
            self.change_pause_status()
        Setting_panal.CLICKED = True

    def _on_factor_slider(self, value):
        Setting_panal.INTRINSIC_FACTOR = value/10
        print(Setting_panal.INTRINSIC_FACTOR)
        Setting_panal.CLICKED = True

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

def main():
    gui.Application.instance.initialize()

    w = Setting_panal(1080, 720)

    gui.Application.instance.run()

if __name__ == "__main__":
    main()