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
import os
import functools
import platform

# from regex import F

sys.path.append('.')
sys.path.append('..')

from .base_gui import AppWindow as GUI_BASE, creat_btn
from .gui_material import Settings
from .utils import generate_mesh

isMacOS = (platform.system() == "Darwin")

class Menu(GUI_BASE):
    MENU_OPEN = 11
    MENU_OPEN_REMOTE = 12
    MENU_EXPORT = 13
    MENU_QUIT = 14
    
    MENU_SHOW_SETTINGS = 21
    MENU_SHOW_WINDOWS = 22

    MENU_SCENE = 31
    MENU_SMPL = 32
    MENU_TRAJ = 33

    MENU_TOOLS_1 = 41
    MENU_TOOLS_2 = 42
    MENU_TOOLS_3 = 43
    MENU_TOOLS_4 = 44

    MENU_ABOUT = 51

    def __init__(self, width=1280, height=768, name='Menu'):
        super(Menu, self).__init__(width, height, name)
        self.geo_settings = {}
        # self.remote_setting = self._remote_setting()
        self.add_menu()

    def add_menu(self):
        w = self.window
        
        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created

        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", Menu.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", Menu.MENU_QUIT)
                
            # file menu
            file_menu = gui.Menu()
            file_menu.add_item("Open 3D file...", Menu.MENU_SCENE)
            file_menu.add_item("Open pcds folder...", Menu.MENU_OPEN)
            file_menu.add_item("Open remote pcds folder...", Menu.MENU_OPEN_REMOTE)
            # file_menu.add_item("Open smpl pkl", Menu.MENU_SMPL)
            file_menu.add_item("Export Current Image...", Menu.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", Menu.MENU_QUIT)

            # setting menu
            settings_menu = gui.Menu()
            # settings_menu.add_item("Settings",Menu.MENU_SHOW_SETTINGS)
            settings_menu.add_item("Window 0", Menu.MENU_SHOW_WINDOWS)

            # settings_menu.set_checked(Menu.MENU_SHOW_SETTINGS, True)
            settings_menu.set_checked(Menu.MENU_SHOW_WINDOWS, True)

            # smpl file tool menu
            smpl_menu = gui.Menu()
            smpl_menu.add_item("Open SMPL pkl", Menu.MENU_SMPL)
            smpl_menu.add_item("Load a trajectory", Menu.MENU_TRAJ)

            # tracking tool menu
            tools_menu = gui.Menu()
            tools_menu.add_item("Load remote pcds for tracking", Menu.MENU_TOOLS_1)
            tools_menu.add_item("Load remote images(.png .jpg)", Menu.MENU_TOOLS_2)
            tools_menu.add_item("Load tracking_traj", Menu.MENU_TOOLS_3)
            tools_menu.add_item("Mesh generating", Menu.MENU_TOOLS_4)

            # help menu
            help_menu = gui.Menu()
            help_menu.add_item("Copyright", Menu.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("View", settings_menu)
                menu.add_menu("Tools", tools_menu)
                menu.add_menu("SMPL", smpl_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("View", settings_menu)
                menu.add_menu("SMPL", smpl_menu)
                menu.add_menu("Tools", tools_menu)
                menu.add_menu("About", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.

        # menu for file loading
        w.set_on_menu_item_activated(Menu.MENU_OPEN, self._on_pcd_folder)
        w.set_on_menu_item_activated(Menu.MENU_OPEN_REMOTE, self._on_remote_pcd_folder)
        w.set_on_menu_item_activated(Menu.MENU_EXPORT, self._on_menu_export)
        w.set_on_menu_item_activated(Menu.MENU_QUIT, self._on_menu_quit)

        # menu for view settings
        # w.set_on_menu_item_activated(Menu.MENU_SHOW_SETTINGS,self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(Menu.MENU_SHOW_WINDOWS,self._on_WINDOW_toggle_settings_panel)

        # menu for smpl file loading
        w.set_on_menu_item_activated(Menu.MENU_SCENE, self._on_menu_scene)
        w.set_on_menu_item_activated(Menu.MENU_SMPL, self._on_menu_smpl)
        w.set_on_menu_item_activated(Menu.MENU_TRAJ, self._on_menu_traj)

        # menu for tools functions
        w.set_on_menu_item_activated(Menu.MENU_TOOLS_1, self._on_remote_pcd_folder)
        w.set_on_menu_item_activated(Menu.MENU_TOOLS_2, self._on_remote_imgs)
        w.set_on_menu_item_activated(Menu.MENU_TOOLS_3, self._on_menu_trackingtraj)
        w.set_on_menu_item_activated(Menu.MENU_TOOLS_4, self._on_mesh_generating)

        w.set_on_menu_item_activated(Menu.MENU_ABOUT, self._on_menu_about)

    def _on_remote_pcd_folder(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("Connect remote machine with SSH")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Connect remote machine with SSH"))
        remote_info = gui.Vert(em, gui.Margins(em, em, em, em))
        remote_info.preferred_width = 30 * em
        remote_info.add_child(self._remote_setting())
        dlg_layout.add_child(remote_info)

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        cancel = gui.Button("Cancel")
        cancel.set_on_clicked(self._on_about_ok)

        connect = creat_btn('Connect', lambda: self._loading_pcds(None))

        h = gui.Horiz(em)
        h.add_stretch()
        h.add_child(connect)
        h.add_child(cancel)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_remote_imgs(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("Connect remote machine with SSH")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Connect remote machine with SSH"))
        remote_info = gui.Vert(em, gui.Margins(em, em, em, em))
        remote_info.preferred_width = 30 * em
        remote_info.add_child(self._remote_setting())
        dlg_layout.add_child(remote_info)

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        cancel = gui.Button("Cancel")
        cancel.set_on_clicked(self._on_about_ok)

        connect = creat_btn('Connect', lambda: self._loading_imgs(None))

        h = gui.Horiz(em)
        h.add_stretch()
        h.add_child(connect)
        h.add_child(cancel)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_pcd_folder(self):
        filedlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Select file",
                                 self.window.theme)
        filedlg.add_filter(".pcd .ply", "Triangle mesh (.pcd, .ply)")
        filedlg.add_filter("", "All files")
        filedlg.set_on_cancel(self._on_file_dialog_cancel)
        filedlg.set_on_done(self._on_filedlg_done)
        self.window.show_dialog(filedlg)
        
    def _on_filedlg_done(self, path):
        # self.remote_info['folder'].text_value = path
        self.window.close_dialog()
        self._loading_pcds(path)

    def _loading_pcds(path):
        pass

    def _loading_imgs(path):
        pass

    def _on_menu_trackingtraj(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose traj file to load",
                             self.window.theme)
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .txt .pcd .pts",
            "trajectory files (.xyz, .xyzn, .xyzrgb, .txt, .pcd, .pts)")

        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._load_tracked_traj)
        self.window.show_dialog(dlg)

    def _on_mesh_generating(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose traj file to load",
                             self.window.theme)
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .txt .pcd .pts",
            "trajectory files (.xyz, .xyzn, .xyzrgb, .txt, .pcd, .pts)")

        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._generate_mesh)
        self.window.show_dialog(dlg)

    def _generate_mesh(self, path):
        generate_mesh(path, depth=13, radius=0.1)
        self.window.close_dialog()

    def _load_tracked_traj(self, path, translate=[0,0,0], load_data_class=None):
        import numpy as np

        self.window.close_dialog()
        if not os.path.isfile(path):
            self.warning_info(f'{path} is not a valid file')
            return 
        try:
            trajs = np.loadtxt(path)
        except Exception as e:
            self.warning_info('Load traj failed.')
            return 

        if trajs.shape[1] > 5 or trajs.shape[1] < 4:
            self.warning_info("Tracking trajs must contains: 'x y z frameid' in every line!!!")
            return 

        return trajs

    def _on_menu_scene(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose pcd/ply/obj to load",
                self.window.theme)
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, .pcd, .pts)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")

        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self.load_scene)
        self.window.show_dialog(dlg)

    def _on_WINDOW_toggle_settings_panel(self):
        self._scene_traj.visible = not self._scene_traj.visible
        gui.Application.instance.menubar.set_checked(
            Menu.MENU_SHOW_WINDOWS, self._scene_traj.visible)

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

    def pop_up_text(self, info, func, type='Info'):
        """
        It creates a dialog box with a text box and an OK button. 
        The text box is used to enter a value. 
        The OK button is used to close the dialog box. 
        The function takes three arguments: 
        
        1. info: This is the text that appears in the dialog box. 
        2. func: This is the function that is called when the OK button is clicked. 
        3. type: This is the type of dialog box. 
        
        The function returns nothing. 
        
        Args:
          info: The text that will be displayed in the pop-up window.
          func: the function to be called when the text is changed
          type: The type of message you want to display. Defaults to Info
        """
        em = self.window.theme.font_size
        dlg = gui.Dialog(f'[{type}]')
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(f'[{type}] \n {info}'))
        text_step = gui.TextEdit()
        text_step.set_on_value_changed(func)
        
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(text_step)
        h.add_stretch()
        dlg_layout.add_child(h)
        dlg_layout.add_child(ok)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def dlg_yes_or_no(self, info, func, type='Info'):
        em = self.window.theme.font_size
        dlg = gui.Dialog(f'[{type}]')
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        if info == None:
            info = f"Are you sure to delete '{self._selected_geo}' ? "
        dlg_layout.add_child(gui.Label(f'[{type}] \n {info}'))
        yes = gui.Button("Yes")
        no = gui.Button("No")

        yes.set_on_clicked(func)
        no.set_on_clicked(self._on_about_ok)

        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(yes)
        h.add_child(no)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def warning_info(self, info, type='Warning'):
        em = self.window.theme.font_size
        dlg = gui.Dialog(f'[{type}]')
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(f'[{type}] \n {info}'))
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        # for gui in guis:
        #     dlg_layout.add_child(gui)
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def show_ImageWidget(self, info, image_path):
        em = self.window.theme.font_size
        dlg = gui.Dialog(f'Image')
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(f'{info}'))
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        try:
            dlg_layout.add_child(gui.ImageWidget(image_path))
        except:
            dlg_layout.add_child(gui.Lable('False image path!'))
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

    def _on_material_setting(self, name):
        self._selected_geo = name
        if 'mat' not in self._geo_list[name]:
            self._geo_list[name]['mat'] = Settings()

        settings = self._geo_list[name]['mat']

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

    def _on_load_smpl_done(self, filename):
        self.warning_info('Function not implemented yet')

    def load_scene(self, scene_path):
        self.warning_info('Function not implemented yet')

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            Menu.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def load_traj(self, traj_path):
        self.warning_info('Function not implemented yet')

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _change_remote_info(self, text, key=None):
        self.remote_info[key] = text

    def _remote_setting(self):
        em = self.window.theme.font_size
        separation_height = int(round(0.5 * em))
        remote = gui.Vert(0.15 * em)

        remote_layout = gui.VGrid(2, 0.15 * em)
        tlist = ['username', 'hostname', 'port', 'folder', 'pwd']
        for key in tlist:
            text_edit = gui.TextEdit()
            text_edit.set_on_value_changed(functools.partial(self._change_remote_info, key=key))
            text_edit.text_value = self.remote_info[key]
            remote_layout.add_child(gui.Label(key))
            remote_layout.add_child(text_edit)

        remote.add_child(remote_layout)
        
        return remote
        
    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        try:
            self.load_scene(filename)
        except Exception as e:
            self.load(filename)

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)   
        
    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Copyright"))
        text = gui.Vert(em, gui.Margins(em, em, em, em))
        text.preferred_width = 30 * em
        text.add_child(gui.Label("This is a visualization tool for LiDAR Human Motion Program (http://lidarhumanmotion.net)."))
        text.add_child(gui.Label("The code is realeased on \n'https://github.com/climbingdaily/vis_lidar_human_scene'."))
        text.add_child(gui.Label("The codebase is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License. You must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. Contact us if you are interested in commercial usage. \n\nyudidai@stu.xmu.edu.cn \nCopyright@Yudi Dai. "))
        dlg_layout.add_child(text)

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz(em)
        h.add_stretch()
        h.add_child(ok)
        # h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

def main():
    gui.Application.instance.initialize()

    w = Menu(1280, 720)

    gui.Application.instance.run()

if __name__ == "__main__":
    main()