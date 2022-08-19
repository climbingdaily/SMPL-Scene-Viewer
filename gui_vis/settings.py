import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import sys
sys.path.append('.')
sys.path.append('..')
from .menu import Menu

def creat_btn(name, func, color=None):
    btn = gui.Button(name)
    btn.vertical_padding_em = 0
    btn.horizontal_padding_em = 0.5
    if color is not None:
        btn.background_color = gui.Color(r=color[0], b=color[1], g=color[2])
    btn.set_on_clicked(func)
    return btn

class Setting_panal(Menu):

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
        self.create_checkboxes()

    def create_checkboxes(self, collapse=None, checkboxes=None):
        em = self.window.theme.font_size

        if collapse is None:
            collapse = gui.CollapsableVert("Human data", 0.33 * em,
                                            gui.Margins(em, 0, 0, 0))

        play_btn = creat_btn('Play / Stop', self.change_pause_status, color = [0, 0, 0.5])

        slider = gui.Slider(gui.Slider.INT)
        slider.set_limits(0, 1000)
        slider.set_on_value_changed(self._on_slider)

        prog_layout = gui.Horiz()
        minus_btn = creat_btn('-', self._minus_frame)
        add_btn = creat_btn('+', self._add_frame)

        # prog_layout.add_child(gui.Label("Frames"))
        prog_layout.add_child(minus_btn)
        prog_layout.add_child(add_btn)
        prog_layout.add_child(slider)

        tabs = gui.TabControl()
        tab1 = gui.Vert()

        try:
            box = gui.Checkbox('Archive')
            box.set_on_checked(self._on_show_archive_geometry)
            box.checked = True
            tab1.add_child(box)

            for box in checkboxes:
                tab1.add_child(box)
        except:
            pass


        tab2 = gui.Vert()
        box = gui.Checkbox('First person view')
        box.set_on_checked(self._on_FPV)
        box.checked = True
        tab2.add_child(box)
        
        box = gui.Checkbox('Free view')
        box.set_on_checked(self._on_free_view)
        # tab2.add_stretch()
        tab2.add_child(box)


        tab3 = gui.Vert()
        box = gui.Checkbox('Auto Render')
        box.set_on_checked(self.change_render_states)
        # box.checked = True
        tab3.add_child(box)

        tabs.add_tab("Data", tab1)
        tabs.add_tab("Cameras", tab2)
        tabs.add_tab("Render Option", tab3)

        collapse.add_child(prog_layout)
        collapse.add_child(play_btn)
        collapse.add_child(tabs)
        self._settings_panel.add_child(collapse)
        # self.window.set_on_layout(self._on_layout)

        self.check_boxes, self.camera_setting, self.slider_bar, self.play_btn = tab1, tab2, slider, play_btn

    def _on_free_view(self, show):
        Setting_panal.FREE_VIEW = show
        # print(show)
        
    def _on_show_archive_geometry(self, show):
        for name in self.archive_data:
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
        text = 'Stoped' if Setting_panal.PAUSE else 'Playing'
        color = gui.Color(r=0.5, b=0, g=0) if Setting_panal.PAUSE else gui.Color(r=0, b=0, g=0.5)
        self.play_btn.text = f'>|| ({text})'
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