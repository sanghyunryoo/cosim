from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QDialogButtonBox,
    QScrollArea, QWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator


class ActionScaleSettingsDialog(QDialog):
    """
    Dialog for configuring action_scale values for different joint/actuator types.
    
    This dialog dynamically creates input fields based on the action_scale keys
    present in the hardware settings (e.g., hip, shoulder, leg, wheel for wheeled robots,
    or hip_pitch, torso, knee, etc. for humanoid robots).
    """
    
    def __init__(self, action_scale_settings, parent=None):
        """
        Initialize the Action Scale Settings Dialog.
        
        Args:
            action_scale_settings (dict): Dictionary containing action_scale values
                                         for different joints/actuators
            parent: Parent widget
        """
        super().__init__(parent)
        self.action_scale_settings = (action_scale_settings or {}).copy()
        self.setWindowTitle("Action Scale Settings")
        self.setMinimumWidth(400)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface components."""
        # Main vertical layout for the dialog
        main_layout = QVBoxLayout(self)

        # Add description label at the top
        description = QLabel(
            "Configure action scale values for each joint/actuator.\n"
            "These values scale the policy output before sending to actuator."
        )
        description.setWordWrap(True)
        description.setStyleSheet("QLabel { padding: 10px; background-color: #f0f0f0; border-radius: 5px; }")
        main_layout.addWidget(description)

        # Create a scroll area to allow scrolling when height exceeds limit
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create an inner widget that will hold the form layout
        inner_widget = QWidget()
        form_layout = QFormLayout(inner_widget)
        form_layout.setLabelAlignment(Qt.AlignRight)
        form_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        
        self.fields = {}

        # Build editable fields for each action_scale setting
        if self.action_scale_settings:
            for key in sorted(self.action_scale_settings.keys()):
                value = self.action_scale_settings[key]
                
                # Create label with formatted text
                label = QLabel(f"{key}:")
                label.setMinimumWidth(120)
                
                # Create line edit with validator
                le = QLineEdit(str(value))
                le.setValidator(QDoubleValidator())
                le.setMinimumWidth(100)
                le.setToolTip(f"Set action scale for {key}")
                
                form_layout.addRow(label, le)
                self.fields[key] = le
        else:
            # Show message if no action_scale settings found
            no_data_label = QLabel("No action scale settings available for this environment.")
            no_data_label.setAlignment(Qt.AlignCenter)
            no_data_label.setStyleSheet("QLabel { padding: 20px; color: #666; }")
            form_layout.addRow(no_data_label)

        # Set the inner widget inside the scroll area
        scroll.setWidget(inner_widget)
        main_layout.addWidget(scroll)

        # OK / Cancel buttons at the bottom
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        # Set maximum height to 600px; scrolling will be enabled if exceeded
        self.setMaximumHeight(600)

    def get_settings(self):
        """
        Get the current action_scale settings from the dialog.
        
        Returns:
            dict: Dictionary containing action_scale values for each joint/actuator
        """
        return {key: float(le.text()) if le.text() else 0.0 
                for key, le in self.fields.items()}

    def validate_settings(self):
        """
        Validate that all fields contain valid numeric values.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        for key, le in self.fields.items():
            text = le.text().strip()
            if not text:
                return False, f"'{key}' field is empty"
            try:
                float(text)
            except ValueError:
                return False, f"'{key}' contains invalid numeric value"
        return True, ""
