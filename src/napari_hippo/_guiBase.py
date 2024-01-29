from qtpy.QtWidgets import QWidget, QVBoxLayout, QGroupBox


# base class for GUI
class GUIBase(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()
        self.subwidgets = []
        self.viewer = napari_viewer

        self.qvl = QVBoxLayout()
        self.setLayout(self.qvl)

    def _add(self, elements, groupname):
        bx = QGroupBox(groupname)
        vbox = QVBoxLayout()

        for e in elements:
            vbox.addWidget(e.root_native_widget)
            self.subwidgets.append(e)

        bx.setLayout(vbox)
        self.qvl.addWidget(bx)

    def reset_choices(self, event=None):
        for e in self.subwidgets:
            e.reset_choices()


