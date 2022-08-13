import open3d.visualization.gui as gui
import sys
from scipy.spatial.transform import Rotation as R

sys.path.append('.')
sys.path.append('..')

from base_gui import AppWindow as GUI_BASE

class Menu(GUI_BASE):
    MENU_SCENE = 31
    MENU_SMPL = 32
    MENU_VIS = 33

    def __init__(self, width=1280, height=768):
        super(Menu, self).__init__(width, height)
        self.add_menu()

    def add_menu(self):
        file_menu = gui.Menu()
        if gui.Application.instance.menubar is not None:
            menu = gui.Application.instance.menubar
            smpl_menu = gui.Menu()
            smpl_menu.add_item("Open scene PCD", Menu.MENU_SCENE)
            smpl_menu.add_item("Open SMPL pkl", Menu.MENU_SMPL)
            smpl_menu.add_item("Show humans and scene", Menu.MENU_VIS)
            menu.add_menu("SMPL", smpl_menu)

        self.window.set_on_menu_item_activated(Menu.MENU_SCENE, self._on_menu_scene)
        self.window.set_on_menu_item_activated(Menu.MENU_SMPL, self._on_menu_smpl)
        self.window.set_on_menu_item_activated(Menu.MENU_VIS, self._on_menu_show)

    def _on_menu_scene(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose pcd/ply/obj to load",
                self.window.theme)
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")

        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self.load_scene)
        self.window.show_dialog(dlg)

    def _on_menu_smpl(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose pkl file to load",
                             self.window.theme)
        dlg.add_filter(
            ".pkl",
            "SMPL files")

        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_smpl_done)
        self.window.show_dialog(dlg)

    def _on_menu_show(self):
        em = self.window.theme.font_size
        dlg = gui.Dialog("Warning")
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("[Warning]: Please load the scene first"))
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
        pass

    def load_scene(self, scene_path):
        pass

def main():
    gui.Application.instance.initialize()

    w = Menu(1080, 720)

    gui.Application.instance.run()

if __name__ == "__main__":
    main()