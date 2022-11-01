# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
import sys
import cv2
import time

from .gui_material import Settings

isMacOS = (platform.system() == "Darwin")

def creat_btn(name, func, color=None):
    btn = gui.Button(name)
    btn.horizontal_padding_em = 0.2
    btn.vertical_padding_em = 0
    if color is not None:
        btn.background_color = gui.Color(r=color[0], b=color[1], g=color[2])
    btn.set_on_clicked(func)
    return btn

class AppWindow:
    
    COOR_INIT = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    DEFAULT_IBL = "default"

    MATERIAL_NAMES = ["Lit", "Unlit", "Normals", "Depth", "Transparency", "LitSSR"]
    MATERIAL_SHADERS = [
        Settings.LIT, Settings.UNLIT, Settings.NORMALS, Settings.DEPTH, Settings.Transparency, Settings.LitSSR
    ]

    def __init__(self, width, height, name='Open3D'):
        self.settings = Settings()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

        self._make_remote_gui()
        self.window = gui.Application.instance.create_window(
            name, width, height)
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_on_sun_direction_changed(self._on_sun_dir)

        self._scene_traj = gui.SceneWidget()
        self._scene_traj.scene = rendering.Open3DScene(w.renderer)
        self._scene_traj.set_on_sun_direction_changed(self._on_sun_dir)
        self._scene_traj.visible = False

        # geometry type list
        self.geo_list = {}

        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row. All layouts take a spacing parameter, which is the spacing
        # between items in the widget, and a margins parameter, which specifies
        # the spacing of the left, top, right, bottom margins. (This acts like
        # the 'padding' property in CSS.)
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # Create a collapsable vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use.

        _setting_tabs = gui.TabControl()

        # view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
        #                                  gui.Margins(em, 0, 0, 0))
        view_ctrls = gui.Vert()

        self._arcball_button = gui.Button("Arcball")
        self._arcball_button.horizontal_padding_em = 0.5
        self._arcball_button.vertical_padding_em = 0
        self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)
        self._fly_button = gui.Button("Fly")
        self._fly_button.horizontal_padding_em = 0.5
        self._fly_button.vertical_padding_em = 0
        self._fly_button.set_on_clicked(self._set_mouse_mode_fly)
        self._model_button = gui.Button("Model")
        self._model_button.horizontal_padding_em = 0.5
        self._model_button.vertical_padding_em = 0
        self._model_button.set_on_clicked(self._set_mouse_mode_model)
        self._sun_button = gui.Button("Sun")
        self._sun_button.horizontal_padding_em = 0.5
        self._sun_button.vertical_padding_em = 0
        self._sun_button.set_on_clicked(self._set_mouse_mode_sun)
        self._ibl_button = gui.Button("Environment")
        self._ibl_button.horizontal_padding_em = 0.5
        self._ibl_button.vertical_padding_em = 0
        self._ibl_button.set_on_clicked(self._set_mouse_mode_ibl)
        view_ctrls.add_child(gui.Label("Mouse controls"))
        # We want two rows of buttons, so make two horizontal layouts. We also
        # want the buttons centered, which we can do be putting a stretch item
        # as the first and last item. Stretch items take up as much space as
        # possible, and since there are two, they will each take half the extra
        # space, thus centering the buttons.
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._arcball_button)
        h.add_child(self._fly_button)
        h.add_child(self._sun_button)
        h.add_stretch()
        view_ctrls.add_child(h)
        # h = gui.Horiz(0.25 * em)  # row 2
        # h.add_stretch()
        # h.add_child(self._model_button)
        # h.add_child(self._ibl_button)
        # h.add_stretch()
        # view_ctrls.add_child(h)

        self._show_skybox = gui.Checkbox("skymap")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        
        self._show_axes = gui.Checkbox("axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        # view_ctrls.add_fixed(separation_height)
        self._show_ground_plane = gui.Checkbox("ground")
        self._show_ground_plane.set_on_checked(self._on_show_ground_plane)
        view_ctrls.add_fixed(separation_height)
        # h = gui.Horiz(0.25 * em)
        # h.add_child(self._show_skybox)
        # h.add_child(self._show_axes)
        # h.add_child(self._show_ground_plane)
        # view_ctrls.add_child(h)
        # h.add_fixed(separation_height)

        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        view_ctrls.add_child(grid)

        
        self._profiles = gui.Combobox()
        for name in sorted(Settings.LIGHTING_PROFILES.keys()):
            self._profiles.add_item(name)
        self._profiles.add_item(Settings.CUSTOM_PROFILE_NAME)
        self._profiles.set_on_selection_changed(self._on_lighting_profile)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(gui.Label("Lighting profiles"))
        view_ctrls.add_child(self._profiles)
        view_ctrls.add_fixed(separation_height)
        # self._settings_panel.add_fixed(separation_height)
        _setting_tabs.add_tab('Lighting', view_ctrls)
        
        # self._settings_panel.add_child(view_ctrls)

        advanced = gui.CollapsableVert("Advanced lighting", 0,
                                       gui.Margins(em, 0, 0, 0))
        # advanced.set_is_open(False)
        # advanced = gui.Vert()


        self._use_ibl = gui.Checkbox("HDR map")
        self._use_ibl.set_on_checked(self._on_use_ibl)
        self._use_sun = gui.Checkbox("Sun")
        self._use_sun.set_on_checked(self._on_use_sun)
        advanced.add_child(gui.Label("Light sources"))
        h = gui.Horiz(em)
        h.add_child(self._use_ibl)
        h.add_child(self._use_sun)
        advanced.add_child(h)

        self._ibl_map = gui.Combobox()
        for ibl in glob.glob(gui.Application.instance.resource_path +
                             "/*_ibl.ktx"):

            self._ibl_map.add_item(os.path.basename(ibl[:-8]))
        self._ibl_map.selected_text = AppWindow.DEFAULT_IBL
        self._ibl_map.set_on_selection_changed(self._on_new_ibl)
        self._ibl_intensity = gui.Slider(gui.Slider.INT)
        self._ibl_intensity.set_limits(0, 200000)
        self._ibl_intensity.set_on_value_changed(self._on_ibl_intensity)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("HDR map"))
        grid.add_child(self._ibl_map)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._ibl_intensity)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Environment"))
        advanced.add_child(grid)

        self._sun_intensity = gui.Slider(gui.Slider.INT)
        self._sun_intensity.set_limits(0, 200000)
        self._sun_intensity.set_on_value_changed(self._on_sun_intensity)
        self._sun_dir = gui.VectorEdit()
        self._sun_dir.set_on_value_changed(self._on_sun_dir)
        self._sun_color = gui.ColorEdit()
        self._sun_color.set_on_value_changed(self._on_sun_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._sun_intensity)
        grid.add_child(gui.Label("Direction"))
        grid.add_child(self._sun_dir)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._sun_color)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Sun (Directional light)"))
        advanced.add_child(grid)

        # self._settings_panel.add_fixed(separation_height)
        # _setting_tabs.add_tab('Lighting', advanced)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(advanced)
        # self._settings_panel.add_child(advanced)

        material_settings = gui.CollapsableVert("Material settings", 0,
                                                gui.Margins(em, 0, 0, 0))
        # material_settings = gui.Vert()

        self._shader = gui.Combobox()
        for shader in AppWindow.MATERIAL_NAMES:
            self._shader.add_item(shader)

        self._shader.set_on_selection_changed(self._on_shader)
        self._material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(self._on_material_prefab)
        self._material_color = gui.ColorEdit()
        self._material_color.set_on_value_changed(self._on_material_color)
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        def empty(show):
            pass

        self.point_box = gui.Checkbox("Point")
        self.point_box.set_on_checked(empty)
        self.point_box.checked = True
        self.mesh_box = gui.Checkbox("Mesh")
        self.mesh_box.set_on_checked(empty)
        self.mesh_box.checked = True
        
        grid = gui.Horiz(0.25 * em)
        grid.add_child(gui.Label("Geometry type"))
        grid.add_child(self.mesh_box)
        grid.add_child(self.point_box)
        material_settings.add_child(grid)
        material_settings.add_fixed(separation_height)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Shader"))
        grid.add_child(self._shader)
        grid.add_child(gui.Label("Material"))
        grid.add_child(self._material_prefab)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._material_color)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        material_settings.add_child(grid)


        # view_ctrls.add_fixed(separation_height)
        h = gui.Horiz(0.25 * em)
        h.add_child(self._show_skybox)
        h.add_child(self._show_axes)
        h.add_child(self._show_ground_plane)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(material_settings)

        # self._settings_panel.add_fixed(separation_height)
        # _setting_tabs.add_tab('Material', material_settings)
        self._settings_panel.add_child(_setting_tabs)
        # ----

        # Normally our user interface can be children of all one layout (usually
        # a vertical layout), which is then the only child of the window. In our
        # case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout
        # callback. The on_layout callback should set the frame
        # (position + size) of every child correctly. After the callback is
        # done the window will layout the grandchildren.
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._scene_traj)
        w.add_child(self._settings_panel)

        self._apply_settings()

    def _apply_settings(self):
        self._apply_settings_on_scene(self._scene.scene)
        self._apply_settings_on_scene(self._scene_traj.scene)

    def _apply_settings_on_scene(self, scene):
        
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]

        scene.set_background(bg_color)
        scene.show_skybox(self.settings.show_skybox)
        scene.show_ground_plane(self.settings.show_ground_plane, rendering.Scene.GroundPlane(0))
        scene.show_axes(self.settings.show_axes)
        if self.settings.new_ibl_name is not None:
            scene.scene.set_indirect_light(
                self.settings.new_ibl_name)
            # Clear new_ibl_name, so we don't keep reloading this image every
            # time the settings are applied.
            self.settings.new_ibl_name = None
        scene.scene.enable_indirect_light(self.settings.use_ibl)
        scene.scene.set_indirect_light_intensity(
            self.settings.ibl_intensity)
        sun_color = [
            self.settings.sun_color.red, self.settings.sun_color.green,
            self.settings.sun_color.blue
        ]
        scene.scene.set_sun_light(self.settings.sun_dir, sun_color,
                                              self.settings.sun_intensity)
        scene.scene.enable_sun_light(self.settings.use_sun)

        if self.settings.apply_material:
            # self._scene.scene.update_material(self.settings.material)
            # update material for point cloud
            if self.point_box.checked:
                for name in self.geo_list:
                    if 'point' == self.geo_list[name]['type']:
                        scene.modify_geometry_material(name, self.settings.material)

            # update material for mesh
            if self.mesh_box.checked:
                for name in self.geo_list:
                    if 'mesh' == self.geo_list[name]['type']:
                        scene.modify_geometry_material(name, self.settings.material)

            self.settings.apply_material = False

        self._bg_color.color_value = self.settings.bg_color
        self._show_skybox.checked = self.settings.show_skybox
        self._show_ground_plane.checked = self.settings.show_ground_plane
        self._show_axes.checked = self.settings.show_axes
        self._use_ibl.checked = self.settings.use_ibl
        self._use_sun.checked = self.settings.use_sun
        self._ibl_intensity.int_value = self.settings.ibl_intensity
        self._sun_intensity.int_value = self.settings.sun_intensity
        self._sun_dir.vector_value = self.settings.sun_dir
        self._sun_color.color_value = self.settings.sun_color
        self._material_prefab.enabled = (
            self.settings.material.shader in [Settings.LIT, Settings.Transparency, Settings.LitSSR])
        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])
        self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        self._scene_traj.frame = gui.Rect(r.width * 3/4, r.y, r.width/4, r.height/4)
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

    def _set_mouse_mode_rotate(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self._scene_traj.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_fly(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)
        self._scene_traj.set_view_controls(gui.SceneWidget.Controls.FLY)

    def _set_mouse_mode_sun(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)
        self._scene_traj.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _set_mouse_mode_ibl(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)
        self._scene_traj.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)

    def _set_mouse_mode_model(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)
        self._scene_traj.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_ground_plane(self, show):
        self.settings.show_ground_plane = show
        self._apply_settings()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_use_ibl(self, use):
        self.settings.use_ibl = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_use_sun(self, use):
        self.settings.use_sun = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_lighting_profile(self, name, index):
        if name != Settings.CUSTOM_PROFILE_NAME:
            self.settings.apply_lighting_profile(name)
            self._apply_settings()

    def _on_new_ibl(self, name, index):
        self.settings.new_ibl_name = gui.Application.instance.resource_path + "/" + name
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_ibl_intensity(self, intensity):
        self.settings.ibl_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_intensity(self, intensity):
        self.settings.sun_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_dir(self, sun_dir):
        self.settings.sun_dir = sun_dir
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_color(self, color):
        self.settings.sun_color = color
        self._apply_settings()

    def _on_shader(self, name, index):
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_material_prefab(self, name, index):
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_color(self, color):
        self.settings.material.base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        if self.settings.material.shader in [Settings.Transparency, Settings.LitSSR]:
            self.settings.material.absorption_color = [
                color.red, color.green, color.blue]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_about_ok(self):
        self.window.close_dialog()

    def load(self, path, name='__model__'):
        # self._scene.scene.clear_geometry()

        geometry = None
        geometry_type = o3d.io.read_file_geometry_type(path)

        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_mesh(path)

        if mesh is not None:
            if len(mesh.triangles) == 0:
                print(
                    "[WARNING] Contains 0 triangles, will read as point cloud")
                mesh = None
            else:
                mesh.compute_vertex_normals()
                if len(mesh.vertex_colors) == 0:
                    mesh.paint_uniform_color([1, 1, 1])
                geometry = mesh
                self.geo_list[name] = {'geometry': geometry, 'type': 'mesh'}
            # Make sure the mesh has texture coordinates
            if not mesh.has_triangle_uvs():
                uv = np.array([[0.0, 0.0]] * (3 * len(mesh.triangles)))
                mesh.triangle_uvs = o3d.utility.Vector2dVector(uv)
        else:
            print("[Info]", path, "appears to be a point cloud")

        if geometry is None:
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
                self.geo_list[name] = {'geometry': geometry, 'type': 'point'}
            else:
                print("[WARNING] Failed to read points", path)

        if geometry is not None:
            # geometry.transform(self.COOR_INIT)

            try:
                self._scene.scene.add_geometry(name, geometry, self.settings.material)
                self._scene.scene.set_geometry_transform(name, self.COOR_INIT)
                bounds = geometry.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)

    def remove_geometry(self, name):
        if self._scene.scene.has_geometry(name):
            self._scene.scene.remove_geometry(name)
        if self._scene_traj.scene.has_geometry(name):
            self._scene_traj.scene.remove_geometry(name)

    def export_image(self, path, width, height):

        def on_image(image):
            # img = image
            img = cv2.resize(np.asarray(image), (width, height))
            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            # o3d.io.write_image(path, img, quality)
            cv2.imwrite(path, img[..., [2,1,0]])
            # time.sleep(0.005)
            # cv2.waitKey(5)

        # x = self.window.content_rect.width
        # y = self.window.content_rect.height
        # ratio = 1280/720
        # if x/y > ratio:
        #     x = int(y * ratio)
        # else:
        #     y = int(x / ratio)
        # self._scene.scene.set_view_size(x, y)
        self._scene.scene.scene.render_to_image(on_image)

    def update_geometry(self, geometry, name, mat=None, reset_bounding_box=False, archive=False, freeze=False):
        self.add_geometry(geometry, name, mat, reset_bounding_box, archive, freeze) 


    def _make_remote_gui(self):
        # folder = gui.TextEdit()
        # username = gui.TextEdit()
        # hostname = gui.TextEdit()
        # port = gui.TextEdit()
        # pwd = gui.TextEdit()

        folder = '/hdd/dyd/lidarhumanscene/data/0417003/lidar_data/lidar_frames_rot'
        username = 'dyd'
        hostname = '10.24.80.241'
        port = '911'
        pwd = ''
        
        self.remote_info = {}
        self.remote_info['username'] = username
        self.remote_info['hostname'] = hostname
        self.remote_info['port'] = port
        self.remote_info['folder'] = folder
        self.remote_info['pwd'] = pwd

def main():
    # We need to initalize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = AppWindow(1024, 768)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path)
        else:
            w.window.show_message_box("Error",
                                      "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()