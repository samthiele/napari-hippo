from qtpy.QtWidgets import QWidget, QVBoxLayout, QGroupBox
from qtpy.QtWidgets import QScrollArea, QLabel
from magicgui.widgets import Label


# base class for GUI
class GUIBase(QWidget):
    # Singleton ToolManager for all GUIBase widgets
    _tool_manager = None

    def __init__(self, napari_viewer):
        super().__init__()
        self.subwidgets = []
        self.viewer = napari_viewer

        # Ensure only one tool widget is open at a time
        if GUIBase._tool_manager is None:
            from napari_hippo import ToolManager
            GUIBase._tool_manager = ToolManager(napari_viewer)
        else:
            # Remove previous tool widget if it exists
            if GUIBase._tool_manager.active_tool_widget is not None:
                try:
                    napari_viewer.window.remove_dock_widget(GUIBase._tool_manager.active_tool_widget)
                except Exception:
                    pass
        GUIBase._tool_manager.active_tool_widget = self

    def _add(self, elements, groupname, tutorialstring=None):
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

    def add_scrollable_sections(self, function_widgets, tutorial_text, function_labels=None, stretch=(2,1)):
        """
        Adds two scrollable sections to the layout: one for function widgets, one for tutorial text.
        function_widgets: list of widgets (magicgui or Qt)
        tutorial_text: str, rich text for tutorial
        function_labels: list of str, optional, labels for each function widget (bold, large)
        stretch: tuple, stretch factors for (functions, tutorial)
        """
        from qtpy.QtWidgets import QVBoxLayout, QScrollArea, QLabel, QWidget, QGroupBox
        from qtpy.QtCore import Qt

        main_layout = QVBoxLayout(self)
        # self.qvl.addLayout(main_layout)

        # --- Function scrollable area ---
        function_widget = QWidget()
        function_layout = QVBoxLayout(function_widget)
        if function_labels is None:
            function_labels = [None] * len(function_widgets)
        for w, label in zip(function_widgets, function_labels):
            if label:
                lbl = QLabel(f"<span style='font-size:16pt; font-weight:bold'>{label}</span>")
                lbl.setTextFormat(Qt.RichText)
                function_layout.addWidget(lbl)
            if hasattr(w, "root_native_widget"):
                function_layout.addWidget(w.root_native_widget)
            else:
                function_layout.addWidget(w)
        function_layout.addStretch()
        function_scroll = QScrollArea()
        function_scroll.setWidget(function_widget)
        function_scroll.setWidgetResizable(True)
        function_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # --- Tutorial scrollable area ---
        tutorial_widget = QWidget()
        tutorial_layout = QVBoxLayout(tutorial_widget)
        tutorial_group = QGroupBox("Tutorial")
        tutorial_group.setStyleSheet("QGroupBox { font-size:16pt; font-weight: bold; }")
        tutorial_inner_layout = QVBoxLayout()
        tutorial_group.setLayout(tutorial_inner_layout)
        tutorial_label = QLabel()
        tutorial_label.setTextFormat(Qt.RichText)
        tutorial_label.setText(f"<span style='font-weight:normal'>{tutorial_text}</span>")
        tutorial_label.setWordWrap(True)
        tutorial_inner_layout.addWidget(tutorial_label)
        # tutorial_inner_layout.addStretch()
        tutorial_layout.addWidget(tutorial_group)
        tutorial_scroll = QScrollArea()
        tutorial_scroll.setWidget(tutorial_widget)
        tutorial_scroll.setWidgetResizable(True)
        tutorial_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Add both scroll areas to main layout with stretch factors
        main_layout.addWidget(function_scroll, stretch=stretch[0])
        main_layout.addWidget(tutorial_scroll, stretch=stretch[1])
        self.setLayout(main_layout)
