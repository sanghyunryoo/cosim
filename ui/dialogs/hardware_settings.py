from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QDialogButtonBox,
    QScrollArea, QWidget, QPushButton, QHBoxLayout, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from ui.dialogs.action_scale_settings import ActionScaleSettingsDialog


class HardwareSettingsDialog(QDialog):
    def __init__(self, hardware_settings, parent):
        super().__init__(parent)
        self.hardware_settings = (hardware_settings or {}).copy()
        self.setWindowTitle("Hardware Settings")
        self._setup_ui()

    def _setup_ui(self):
        # Main vertical layout for the dialog
        main_layout = QVBoxLayout(self)

        # Add action_scale settings button at the top
        action_scale_btn_layout = QHBoxLayout()

        self.action_scale_btn = QPushButton("Configure Action Scales")
        self.action_scale_btn.setToolTip("Open dialog to configure action_scale values for each joint/actuator")
        self.action_scale_btn.clicked.connect(self._open_action_scale_dialog)
        
        action_scale_btn_layout.addWidget(self.action_scale_btn)
        action_scale_btn_layout.addStretch()
        
        main_layout.addLayout(action_scale_btn_layout)

        # Add separator
        separator = QLabel()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #ccc; margin: 10px 0;")
        main_layout.addWidget(separator)

        # Create a scroll area to allow scrolling when height exceeds limit
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)  # Allow resizing of the scroll area content
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Hide horizontal scrollbar

        # Create an inner widget that will hold the form layout
        inner_widget = QWidget()
        form_layout = QFormLayout(inner_widget)
        self.fields = {}

        # Separate action_scales from other hardware settings
        action_scales = {}
        other_settings = {}
        
        for key, value in self.hardware_settings.items():
            if key == "action_scales":
                # Store action_scales separately
                if isinstance(value, dict):
                    action_scales = value.copy()
            else:
                other_settings[key] = value

        # Store action_scales reference
        self.action_scales = action_scales

        # Build editable fields for each hardware setting (excluding action_scales)
        for key, value in other_settings.items():
            label = QLabel(key)
            le = QLineEdit(str(value))
            le.setValidator(QDoubleValidator())  # Allow only floating-point input
            form_layout.addRow(label, le)
            self.fields[key] = le

        # Set the inner widget inside the scroll area
        scroll.setWidget(inner_widget)
        main_layout.addWidget(scroll)

        # OK / Cancel buttons at the bottom
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)   # Close dialog with "Accepted" state
        buttons.rejected.connect(self.reject)   # Close dialog with "Rejected" state
        main_layout.addWidget(buttons)

        # Set maximum height to 600px; scrolling will be enabled if exceeded
        self.setMaximumHeight(600)

    def _open_action_scale_dialog(self):
        """Open the ActionScaleSettingsDialog to configure action_scales."""
        dialog = ActionScaleSettingsDialog(self.action_scales, self)
        
        if dialog.exec_() == QDialog.Accepted:
            # Validate settings before accepting
            is_valid, error_msg = dialog.validate_settings()
            if not is_valid:
                QMessageBox.warning(self, "Validation Error", error_msg)
                return
            
            # Update action_scales with new values
            self.action_scales = dialog.get_settings()

    def get_settings(self):
        # Return current field values as a dictionary, including action_scales
        result = {key: le.text() for key, le in self.fields.items()}
        
        # Add action_scales back to the result
        if self.action_scales:
            result["action_scales"] = self.action_scales
        
        return result