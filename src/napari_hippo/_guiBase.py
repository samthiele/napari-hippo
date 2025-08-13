from qtpy.QtWidgets import QWidget, QVBoxLayout, QGroupBox
from qtpy.QtWidgets import QScrollArea, QLabel
from magicgui.widgets import Label


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
            if hasattr(e, "root_native_widget"):
                vbox.addWidget(e.root_native_widget)
            else:
                vbox.addWidget(e)
            self.subwidgets.append(e)

        bx.setLayout(vbox)
        self.qvl.addWidget(bx)

    def reset_choices(self, event=None):
        for e in self.subwidgets:
            e.reset_choices()

    def add_tutorial(self, tutorial_text):
        tutorial_label = QLabel()
        tutorial_label.setTextFormat(1)  # Qt.RichText
        tutorial_label.setText(tutorial_text)
        tutorial_label.setWordWrap(True)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(tutorial_label)
        self._add([scroll_area], 'Tutorial')
