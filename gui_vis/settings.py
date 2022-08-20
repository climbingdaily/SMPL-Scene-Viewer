import numpy as np
import open3d as o3d
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
    # FRAME = 0
    PAUSE = False
    POV = 'first'
    RENDER = False
    CLICKED = False
    # PLAY_ONCE = False

    def __init__(self, width=1280, height=720):
        super(Setting_panal, self).__init__(width, height)
        self.archive_data = []
        self.freeze_data = []
        self.create_checkboxes()

    def create_checkboxes(self, collapse=None, checkboxes=None):
        em = self.window.theme.font_size

        if collapse is None:
            collapse = gui.CollapsableVert("Human data", 0.33 * em,
                                            gui.Margins(em, 0, 0, 0))
        slider = gui.Slider(gui.Slider.INT)
        slider.set_limits(0, 1000)
        slider.set_on_value_changed(self._on_slider)

        minus_btn    = creat_btn('-', self._minus_frame)
        add_btn      = creat_btn('+', self._add_frame)
        play_btn     = creat_btn('    >>||    ', self.change_pause_status, color = [0, 0, 0.5])
        freeze_btn   = creat_btn('Freeze frame', self._freeze_frame)
        clear_freeze_btn   = creat_btn('Clear freezed data', self._clear_freeze)

        prog_layout = gui.Horiz(0.15 * em)
        prog_layout.add_child(minus_btn)
        prog_layout.add_child(add_btn)
        prog_layout.add_child(slider)

        horiz_layout = gui.Horiz(em)
        horiz_layout.add_child(play_btn)
        add_box(horiz_layout, 'Auto Render', self.change_render_states)
        # horiz_layout.add_stretch()

        tabs = gui.TabControl()
        tab1 = gui.Vert(0.15 * em)

        try:
            add_box(tab1, 'Chess Board', self._on_show_archive_geometry, True)
            for box in checkboxes:
                tab1.add_child(box)
        except:
            pass

        cameras = create_combobox(self._on_select_camera)
        cam_grid = gui.Horiz(0.25 * em)
        cam_grid.add_child(cameras)
        add_box(cam_grid, 'Free view', self._on_free_view)

        tab2 = gui.Vert()
        # add_box(tab2, 'First person view', self._on_FPV, True)
        tab2.add_child(gui.Label('Cameras'))
        tab2.add_child(cam_grid)

        tab3 = gui.Vert(0.15 * em)
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

        self.check_boxes, self.camera_setting, self.slider_bar, self.play_btn = tab1, cameras, slider, play_btn


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
        self.slider_bar.int_value += 1
        self._on_slider(self.slider_bar.int_value)
        
    def _minus_frame(self):
        if self.slider_bar.int_value > 0:
            self.slider_bar.int_value -= 1
            self._on_slider(self.slider_bar.int_value)

    def change_render_states(self, render):
        Setting_panal.RENDER = render

    def _clicked(self):
        Setting_panal.CLICKED = False


    def _set_slider_value(self, value):
        self.slider_bar.int_value = int(value)
        
    def _get_slider_value(self):
        return self.slider_bar.int_value
        
    def _set_slider_limit(self, min, max):
        self.slider_bar.set_limits(min, max)

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
        # print(int(value))

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