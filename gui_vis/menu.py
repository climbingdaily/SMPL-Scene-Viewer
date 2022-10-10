################################################################################
# File: \menu.py                                                               #
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

from .base_gui import AppWindow as GUI_BASE
from .gui_material import Settings

class Menu(GUI_BASE):
    MENU_SCENE = 31
    MENU_SMPL = 32
    MENU_VIS = 33

    def __init__(self, width=1280, height=768):
        super(Menu, self).__init__(width, height)
        self.geo_settings = {}
        self.add_menu()

    def add_menu(self):
        file_menu = gui.Menu()
        if gui.Application.instance.menubar is not None:
            menu = gui.Application.instance.menubar
            smpl_menu = gui.Menu()
            smpl_menu.add_item("Open scene PCD", Menu.MENU_SCENE)
            smpl_menu.add_item("Open SMPL pkl", Menu.MENU_SMPL)
            smpl_menu.add_item("Load a trajectory", Menu.MENU_VIS)
            menu.add_menu("SMPL", smpl_menu)

        self.window.set_on_menu_item_activated(Menu.MENU_SCENE, self._on_menu_scene)
        self.window.set_on_menu_item_activated(Menu.MENU_SMPL, self._on_menu_smpl)
        self.window.set_on_menu_item_activated(Menu.MENU_VIS, self._on_menu_traj)

    def _on_menu_scene(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose pcd/ply/obj to load",
                self.window.theme)
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, .pcd, .pts)")

        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self.load_scene)
        self.window.show_dialog(dlg)

    def _on_menu_traj(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose traj file to load",
                             self.window.theme)
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .txt .pcd .pts",
            "trajectory files (.xyz, .xyzn, .xyzrgb, .txt, .pcd, .pts)")

        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self.load_traj)
        self.window.show_dialog(dlg)

    def _on_menu_smpl(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose pkl file to load",
                             self.window.theme)
        dlg.add_filter(".pkl", "SMPL files (.pkl)")
        dlg.add_filter(".hdf5 .h5py, ", "H5PY format (.hdf5, .h5py)")

        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_smpl_done)
        self.window.show_dialog(dlg)

    def warning_info(self, info, type='Warning'):
        em = self.window.theme.font_size
        dlg = gui.Dialog(f'[{type}]')
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(f'[{type}]: {info}'))
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _updata_material(self, settings, name):
        if self._scene.scene.has_geometry(name):
            self._scene.scene.modify_geometry_material(name, settings.material)
        if self._scene_traj.scene.has_geometry(name):
            self._scene_traj.scene.modify_geometry_material(name, settings.material)

        # self._material_prefab.enabled = (
        #     settings.material.shader in [Settings.LIT, Settings.Transparency, Settings.LitSSR])

        # c = gui.Color(settings.material.base_color[0],
        #               settings.material.base_color[1],
        #               settings.material.base_color[2],
        #               settings.material.base_color[3])

        # self._material_color.color_value = c
        # self._point_size.double_value = settings.material.point_size

    def _on_material_setting(self):
        name_list = []
        for name, item in self.geo_list.items():
            if not item['freeze'] and not item['archive']:
                name_list.append(name)

        name = name_list[self.data_id-1]
        if name not in self.geo_list:
            self.warning_info(f'No such geometry: {name} ')
            return 

        if 'mat' not in self.geo_list[name]:
            self.geo_list[name]['mat'] = Settings()

        settings = self.geo_list[name]['mat']

        em = self.window.theme.font_size
        dlg = gui.Dialog(name)
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(f'\t\t\tUpdate material for {name}\t\t\t'))

        def _on_shader(_, index):
            settings.set_material(GUI_BASE.MATERIAL_SHADERS[index])
            self._updata_material(settings, name)
            
        def _on_material_prefab(nn, index):
            settings.apply_material_prefab(nn)
            settings.apply_material = True
            self._updata_material(settings, name)
            
        def _on_material_color(color):
            settings.material.base_color = [
                color.red, color.green, color.blue, color.alpha
            ]
            if settings.material.shader in [Settings.Transparency, Settings.LitSSR]:
                settings.material.absorption_color = [
                    color.red, color.green, color.blue]
            settings.apply_material = True
            self._updata_material(settings, name)
            
        def _on_point_size(size):
            settings.material.point_size = int(size)
            settings.apply_material = True
            self._updata_material(settings, name)

        material_settings = gui.Vert()

        # shader settings
        _shader = gui.Combobox()
        for shader in GUI_BASE.MATERIAL_NAMES:
            _shader.add_item(shader)
        _shader.selected_index = GUI_BASE.MATERIAL_SHADERS.index(settings.material.shader)
        _shader.set_on_selection_changed(_on_shader)

        # _material_prefab settings
        _material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            _material_prefab.add_item(prefab_name)
        _material_prefab.selected_text = settings.prefab
        # _material_prefab.enabled = (
        #     settings.material.shader in [Settings.LIT, Settings.Transparency, Settings.LitSSR])
        _material_prefab.set_on_selection_changed(_on_material_prefab)

        # color settings
        _material_color = gui.ColorEdit()
        c = gui.Color(settings.material.base_color[0],
                settings.material.base_color[1],
                settings.material.base_color[2],
                settings.material.base_color[3])
        _material_color.color_value = c
        _material_color.set_on_value_changed(_on_material_color)

        # point size setting
        _point_size = gui.Slider(gui.Slider.INT)
        _point_size.set_limits(1, 10)
        _point_size.double_value = settings.material.point_size
        _point_size.set_on_value_changed(_on_point_size)

        # layout
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Type"))
        grid.add_child(_shader)
        grid.add_child(gui.Label("Material"))
        grid.add_child(_material_prefab)
        grid.add_child(gui.Label("Color"))
        grid.add_child(_material_color)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(_point_size)
        material_settings.add_child(grid)

        dlg_layout.add_child(material_settings)

        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_menu_show(self):
        self.warning_info('Please load the human data')

    def _on_load_smpl_done(self, filename):
        pass

    # def load_scene(self, scene_path):
        # pass

    def load_traj(self, traj_path):
        pass

def main():
    gui.Application.instance.initialize()

    w = Menu(1280, 720)

    gui.Application.instance.run()

if __name__ == "__main__":
    main()