import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import sys
sys.path.append('.')
sys.path.append('..')
from menu import Menu

class Setting_panal(Menu):

    FREE_VIEW = False
    FRAME = 0
    PAUSE = False
    POV = 'first'

    def __init__(self, width=1280, height=720):
        super(Setting_panal, self).__init__(width, height)
        self.create_checkboxes()

    def create_checkboxes(self, collapse=None, checkboxes=None):
        em = self.window.theme.font_size

        if collapse is None:
            collapse = gui.CollapsableVert("Human data", 0.33 * em,
                                            gui.Margins(em, 0, 0, 0))

        play_btn = gui.Button('Play / Stop')
        play_btn.vertical_padding_em = 0
        play_btn.background_color = gui.Color(r=0, b=0, g=0.5)
        play_btn.set_on_clicked(self.change_pause_status)
        # self._settings_panel.add_child(play_btn)

        slider = gui.Slider(gui.Slider.INT)
        slider.set_limits(0, 1000)
        slider.set_on_value_changed(self._on_slider)

        prog_layout = gui.Horiz()
        prog_layout.add_child(gui.Label("Frames"))
        prog_layout.add_child(slider)

        tabs = gui.TabControl()
        tab1 = gui.Vert()

        try:
            for box in checkboxes:
                tab1.add_child(box)
        except:
            pass


        tab2 = gui.Vert()
        # tab2.add_child(gui.Label("No plugins detected"))
        
        box = gui.Checkbox('First person view')
        box.set_on_checked(self._on_FPV)
        box.checked = True
        tab2.add_child(box)
        
        box = gui.Checkbox('Free view')
        box.set_on_checked(self._on_free_view)
        # tab2.add_stretch()
        tab2.add_child(box)

        tabs.add_tab("Data", tab1)
        tabs.add_tab("Cameras", tab2)

        collapse.add_child(prog_layout)
        collapse.add_child(play_btn)
        collapse.add_child(tabs)
        self._settings_panel.add_child(collapse)
        # self.window.set_on_layout(self._on_layout)

        self.check_boxes, self.camera_setting, self.slider_bar, self.play_btn = tab1, tab2, slider, play_btn

    def _on_free_view(self, show):
        Setting_panal.FREE_VIEW = show
        print(show)
        
    def _on_FPV(self, show):
        Setting_panal.POV = 'first' if show else 'second'
        print(show)

    def change_pause_status(self):
        Setting_panal.PAUSE = not Setting_panal.PAUSE
        text = 'Stoped' if Setting_panal.PAUSE else 'Playing'
        color = gui.Color(r=0.5, b=0, g=0) if Setting_panal.PAUSE else gui.Color(r=0, b=0, g=0.5)
        self.play_btn.text = f'>|| ({text})'
        self.play_btn.background_color = color

    def _on_slider(self, value):
        Setting_panal.FRAME = int(value)
        if not Setting_panal.PAUSE:
            self.change_pause_status()
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

def creat_plane():
    import open3d.visualization as vis

    ground_plane = o3d.geometry.TriangleMesh.create_box(
        50.0, 50, 0.01, create_uv_map=True, map_texture_to_each_face=True)
    ground_plane.compute_triangle_normals()
    # rotate_180 = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi, 0, 0))
    # ground_plane.rotate(rotate_180)
    ground_plane.translate((-25.0, -25, -0.01))
    ground_plane.paint_uniform_color((1, 1, 1))
    # ground_plane = o3d.t.geometry.TriangleMesh.from_legacy(ground_plane)

    # Material to make ground plane more interesting - a rough piece of glass
    # mat_ground = vis.Material("defaultLitSSR")
    # mat_ground.scalar_properties['roughness'] = 0.15
    # mat_ground.scalar_properties['reflectance'] = 0.72
    # mat_ground.scalar_properties['transmission'] = 0.6
    # mat_ground.scalar_properties['thickness'] = 0.3
    # mat_ground.scalar_properties['absorption_distance'] = 0.1
    # mat_ground.vector_properties['absorption_color'] = np.array(
    #     [0.82, 0.98, 0.972, 1.0])
    
    return ground_plane

def main():
    gui.Application.instance.initialize()

    w = Setting_panal(1080, 720)

    gui.Application.instance.run()

if __name__ == "__main__":
    main()