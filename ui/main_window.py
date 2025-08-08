import os
import yaml
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QPushButton, QLabel, QMessageBox, QMainWindow,
    QFileDialog, QGroupBox, QScrollArea, QComboBox, QLineEdit, QSlider, QApplication, QCheckBox, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt, QEvent, QUrl
from PyQt5.QtGui import QDesktopServices, QIcon, QDoubleValidator, QIntValidator
from core.tester import Tester

# Custom QComboBox that ignores mouse wheel events
class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()

# Custom QSlider that ignores mouse wheel events
class NoWheelSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()

# Button that ignores mouse clicks (only responds to keyboard input)
class NonClickableButton(QPushButton):
    def mousePressEvent(self, event):
        event.ignore()
    def mouseReleaseEvent(self, event):
        event.ignore()

# New dialog class for hardware settings
class HardwareSettingsDialog(QDialog):
    def __init__(self, hardware_settings, parent=None):
        super().__init__(parent)
        self.hardware_settings = hardware_settings.copy()
        self.setWindowTitle("Hardware Settings")
        self._setup_ui()

    def _setup_ui(self):
        layout = QFormLayout(self)
        self.fields = {}
        for key, value in self.hardware_settings.items():
            label = QLabel(key)
            le = QLineEdit(str(value))
            le.setValidator(QDoubleValidator())
            layout.addRow(label, le)
            self.fields[key] = le
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_settings(self):
        return {key: le.text() for key, le in self.fields.items()}

class TesterWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    def __init__(self, tester: Tester):
        super().__init__()
        self.tester = tester
    def run(self):
        try:
            self.tester.init_user_command()
            self.tester.test()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        cur_file_path = os.path.abspath(__file__)
        config_path = os.path.join(os.path.dirname(cur_file_path), "../config/env_table.yaml")
        config_path = os.path.abspath(config_path)
        with open(config_path) as f:
            self.env_config = yaml.full_load(f)
        self._init_window()
        self._init_variables()
        self._setup_ui()
        self._init_default_command_values()
        self.status_label.setText("대기 중")
        self.env_id_cb.currentTextChanged.connect(self.update_defaults)
        self.update_defaults(self.env_id_cb.currentText())

    def _init_window(self):
        app_logo_path = os.path.join(os.path.dirname(__file__), "icon", "main_logo_128_128.png")
        self.setWindowIcon(QIcon(app_logo_path))
        self.setWindowTitle("cosim - v1.5.0")
        self.resize(1060, 500)
        # 기본적으로 메인 윈도우에 이벤트 필터를 설치
        self.installEventFilter(self)

    def _init_variables(self):
        self.key_mapping = {}
        self.active_keys = {}
        self.thread = None
        self.worker = None
        self.tester = None
        self.current_command_values = [0.0] * 6
        # Lists for command-related QLineEdit/QLabel widgets
        self.command_sensitivity_le_list = []
        self.max_command_value_le_list = []
        self.command_initial_value_le_list = []
        self.command_timer = None
        # === Added: Observation Scales inputs ===
        self.obs_scales_le = {} # dict to hold QLineEdit for obs_scales
        # New variable for hardware settings
        self.hardware_settings = {}

    def _init_default_command_values(self):
        try:
            self.current_command_values = [float(widget.text()) for widget in self.command_initial_value_le_list]
        except Exception:
            self.current_command_values = [0.0] * 6

    def update_defaults(self, new_env_id):
        settings = self.env_config.get(new_env_id)
        # Update hardware settings dictionary instead of QLineEdit fields
        self.hardware_settings = settings["hardware"].copy()
        # Optional env settings
        env_settings = settings.get("env")
        if isinstance(env_settings, dict):
            observation_dim = env_settings["observation_dim"]
            action_dim = env_settings["action_dim"]
            command_dim = env_settings["command_dim"]
            command_0_max = "1.5"
            command_1_max = "1.5"
            if command_dim is not None:
                self.command_dim_le.setText(str(command_dim))
            if action_dim is not None:
                self.action_dim_le.setText(str(action_dim))
            if observation_dim is not None:
                self.observation_dim_le.setText(str(observation_dim))
            if command_0_max is not None:
                self.max_command_value_le_list[0].setText(command_0_max)
            if command_1_max is not None:
                self.max_command_value_le_list[2].setText(command_1_max)
        # Update command[3] initial value
        if isinstance(self.command_initial_value_le_list[3], QLineEdit):
            self.command_initial_value_le_list[3].setText(env_settings["command_3_initial"])
        # === Added: Update Observation Scales on env change ===
        obs_scales = settings.get("obs_scales", {})
        for key, le in self.obs_scales_le.items():
            if key in obs_scales:
                le.setText(str(obs_scales[key]))
        scale_commands = obs_scales.get("scale_commands", True)
        if scale_commands:
            self.scale_commands_cb.setCurrentText("True")
        else:
            self.scale_commands_cb.setCurrentText("False")

    def showEvent(self, event):
        self.centralWidget().setFocus()
        super().showEvent(event)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            self.handle_key_press(event)
            return True
        elif event.type() == QEvent.KeyRelease:
            self.handle_key_release(event)
            return True
        return super().eventFilter(obj, event)

    def handle_key_press(self, event):
        if event.isAutoRepeat():
            return
        key = event.key()
        if key in self.key_mapping and key not in self.active_keys:
            btn, cmd_index, direction = self.key_mapping[key]
            btn.setChecked(True)
            self.active_keys[key] = {"cmd_index": cmd_index, "direction": direction}

    def handle_key_release(self, event):
        if event.isAutoRepeat():
            return
        key = event.key()
        if key in self.key_mapping:
            btn, cmd_index, _ = self.key_mapping[key]
            btn.setChecked(False)
            if key in self.active_keys:
                self.active_keys.pop(key)
            default_value = self._get_default_command_value(cmd_index)
            self.current_command_values[cmd_index] = default_value
            self._update_command_button(cmd_index, default_value)

    def _get_default_command_value(self, index):
        try:
            return float(self.command_initial_value_le_list[index].text())
        except Exception:
            return 0.0

    def _update_status_label(self):
        html_text = (
            "<html><head><style>"
            "h3 { margin: 0 0 8px 0; }"
            "table { border-collapse: collapse; }"
            "td { padding: 4px 8px; border: 1px solid #ddd; }"
            "</style></head><body>"
            "<h4> Current Command Values</h4><table>"
        )
        for i, value in enumerate(self.current_command_values):
            if i % 6 == 0:
                if i != 0:
                    html_text += "</tr>"
                html_text += "<tr>"
            html_text += f"<td>[{i}] = {value:.3f}</td>"
        html_text += "</tr></table></body></html>"
        self.status_label.setText(html_text)

    def _update_command_button(self, index, value):
        self.current_command_values[index] = value
        self._update_status_label()

    def send_current_command(self):
        for key_info in self.active_keys.values():
            cmd_index = key_info["cmd_index"]
            direction = key_info["direction"]
            step = self._parse_float(self.command_sensitivity_le_list[cmd_index].text(), 0.1)
            max_command_value = self._parse_float(self.max_command_value_le_list[cmd_index].text(), 2.0)
            current_value = self.current_command_values[cmd_index]
            new_value = current_value + direction * step
            if direction > 0:
                new_value = min(new_value, max_command_value)
            else:
                new_value = max(new_value, -max_command_value)
            self.current_command_values[cmd_index] = new_value
            self._update_command_button(cmd_index, new_value)
        if self.tester:
            for i, value in enumerate(self.current_command_values):
                self.tester.update_command(i, value)
        self._update_status_label()

    def _parse_float(self, text, default):
        try:
            return float(text)
        except Exception:
            return default

    # ---------------------------------------------------------------------
    # UI SETUP
    # ---------------------------------------------------------------------

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        # Top area: Left (configuration) and right (command settings/input)
        top_h_layout = QHBoxLayout()
        top_h_layout.setSpacing(15)
        main_layout.addLayout(top_h_layout)
        # Left: Configuration settings in a scrollable area
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        top_h_layout.addWidget(config_scroll, 3)
        config_widget = QWidget()
        config_scroll.setWidget(config_widget)
        self.config_layout = QVBoxLayout(config_widget)
        self.config_layout.setContentsMargins(10, 10, 10, 10)
        self.config_layout.setSpacing(15)
        self._create_top_config_groups()
        # Removed: self._create_hardware_group()
        self._create_random_group()
        # Right: Command settings and key input visual buttons
        right_v_layout = QVBoxLayout()
        top_h_layout.addLayout(right_v_layout, 1)
        self._create_command_settings_group(right_v_layout)
        self._setup_key_visual_buttons(right_v_layout)
        self._create_event_input_group(right_v_layout) # Added for push event
        self.status_label = QLabel("대기 중")
        self.status_label.setStyleSheet("font-size: 14px;")
        main_layout.addWidget(self.status_label)
        # Bottom: Start/Stop buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.start_button = QPushButton("Start Test")
        self.start_button.setFixedWidth(120)
        self.start_button.clicked.connect(self.start_test)
        btn_layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop Test")
        self.stop_button.setFixedWidth(120)
        self.stop_button.clicked.connect(self.stop_test)
        self.stop_button.setEnabled(False)
        btn_layout.addWidget(self.stop_button)
        main_layout.addLayout(btn_layout)
        self._apply_styles()

    # Added method for push event UI
    def _create_event_input_group(self, parent_layout):
        event_group = QGroupBox("Event Input")
        event_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        event_layout = QFormLayout()
        event_layout.setLabelAlignment(Qt.AlignRight)
        event_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        event_layout.setSpacing(8)
        event_group.setLayout(event_layout)
        push_vel_layout = QHBoxLayout()
        self.push_vel_x_le = QLineEdit("0.0")
        self.push_vel_x_le.setPlaceholderText("x")
        self.push_vel_x_le.setFixedWidth(50)
        self.push_vel_y_le = QLineEdit("0.0")
        self.push_vel_y_le.setPlaceholderText("y")
        self.push_vel_y_le.setFixedWidth(50)
        self.push_vel_z_le = QLineEdit("0.0")
        self.push_vel_z_le.setPlaceholderText("z")
        self.push_vel_z_le.setFixedWidth(50)
        push_vel_layout.addWidget(self.push_vel_x_le)
        push_vel_layout.addWidget(self.push_vel_y_le)
        push_vel_layout.addWidget(self.push_vel_z_le)
        event_layout.addRow("Push Velocity (x, y, z):", push_vel_layout)
        self.push_button = QPushButton("Push")
        self.push_button.pressed.connect(self.activate_push_trigger)
        self.push_button.released.connect(self.deactivate_push_trigger)
        event_layout.addRow(self.push_button)
        parent_layout.addWidget(event_group)

    # Added method to handle push event activation
    def activate_push_trigger(self):
        if self.tester:
            try:
                push_vel = [
                    float(self.push_vel_x_le.text()),
                    float(self.push_vel_y_le.text()),
                    float(self.push_vel_z_le.text())
                ]
                self.tester.activate_push_event(push_vel)
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Push velocity must be numeric values.")

    # Added method to handle push event deactivation
    def deactivate_push_trigger(self):
        if self.tester:
            self.tester.deactivate_push_event()

    # ---------------------------------------------------------------------
    # CONFIG GROUPS (env, policy, obs_scales are now separate)
    # ---------------------------------------------------------------------

    def _create_top_config_groups(self):
        top_layout = QHBoxLayout()
        top_layout.setSpacing(15)
        self.config_layout.addLayout(top_layout)
        # Environment group on the left
        self._create_env_group(top_layout)
        # Container (vertical) on the right for Policy + Observation Scales
        policy_container = QVBoxLayout()
        policy_container.setSpacing(10)
        top_layout.addLayout(policy_container, 1)
        # Policy Settings group
        self._create_policy_group(policy_container)
        # Observation Scales group (now OUTSIDE Policy Settings)
        obs_group = self._create_obs_scales_group()
        policy_container.addWidget(obs_group, 0)

    def _create_env_group(self, parent_layout):
        env_group = QGroupBox("Environment Settings")
        env_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        env_layout = QFormLayout()
        env_layout.setLabelAlignment(Qt.AlignRight)
        env_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        env_layout.setSpacing(8)
        env_group.setLayout(env_layout)
        self.env_id_cb = NoWheelComboBox()
        self.env_id_cb.addItems(self.env_config.keys())
        self.env_id_cb.setCurrentText("flamingo_v1_5_1")
        env_layout.addRow("ID:", self.env_id_cb)
        # Add Hardware Settings button
        settings_btn = QPushButton("Hardware Settings")
        settings_btn.clicked.connect(self.open_hardware_settings)
        env_layout.addRow("Hardware:", settings_btn)
        self.terrain_id_cb = NoWheelComboBox()
        self.terrain_id_cb.addItems([
            'flat', 'rocky_easy', 'rocky_hard',
            'slope_easy', 'slope_hard',
            'stairs_up_easy', 'stairs_up_normal', 'stairs_up_hard'
        ])
        self.terrain_id_cb.setCurrentText("flat")
        env_layout.addRow("Terrain:", self.terrain_id_cb)
        self.max_duration_le = QLineEdit("120.0")
        env_layout.addRow("Max Duration (s):", self.max_duration_le)
        self.observation_dim_le = QLineEdit("20")
        env_layout.addRow("Observation Dim:", self.observation_dim_le)
        self.command_dim_le = QLineEdit("6")
        env_layout.addRow("Command Dim:", self.command_dim_le)
        self.action_dim_le = QLineEdit("8")
        env_layout.addRow("Action Dim:", self.action_dim_le)
        self.action_in_state_cb = NoWheelComboBox()
        self.action_in_state_cb.addItems(["True", "False"])
        self.action_in_state_cb.setCurrentText("True")
        env_layout.addRow("Action in State:", self.action_in_state_cb)
        self.num_stack_cb = NoWheelComboBox()
        self.num_stack_cb.addItems([str(i) for i in range(1, 16)])
        self.num_stack_cb.setCurrentText("3")
        env_layout.addRow("Num Stack:", self.num_stack_cb)
        self.external_sensors_cb = NoWheelComboBox()
        self.external_sensors_cb.addItems(["height_map", "base_lin_vel", "All", "None"])
        self.external_sensors_cb.setCurrentText("None")
        env_layout.addRow("External Sensors:", self.external_sensors_cb)
        self.external_sensors_Hz_cb = NoWheelComboBox()
        self.external_sensors_Hz_cb.addItems(["10", "25"])
        self.external_sensors_Hz_cb.setCurrentText("10")
        env_layout.addRow("External Sensors Hz:", self.external_sensors_Hz_cb)
        self.height_size_x_le = QLineEdit()
        self.height_size_x_le.setFixedWidth(50)
        self.height_size_x_le.setPlaceholderText("x (m)")
        double_validator = QDoubleValidator(0.000001, 1e6, 4)
        double_validator.setNotation(QDoubleValidator.StandardNotation)
        self.height_size_x_le.setValidator(double_validator)
        self.height_size_x_le.setText("1.0")
        self.height_size_y_le = QLineEdit()
        self.height_size_y_le.setFixedWidth(50)
        self.height_size_y_le.setPlaceholderText("y (m)")
        self.height_size_y_le.setValidator(double_validator)
        self.height_size_y_le.setText("0.6")
        self.height_res_x_le = QLineEdit()
        self.height_res_x_le.setFixedWidth(50)
        self.height_res_x_le.setPlaceholderText("x_res")
        int_validator = QIntValidator(1, 10000)
        self.height_res_x_le.setValidator(int_validator)
        self.height_res_x_le.setText("15")
        self.height_res_y_le = QLineEdit()
        self.height_res_y_le.setFixedWidth(50)
        self.height_res_y_le.setPlaceholderText("y_res")
        self.height_res_y_le.setValidator(int_validator)
        self.height_res_y_le.setText("9")
        size_res_layout = QHBoxLayout()
        size_res_layout.setSpacing(4)
        size_res_layout.addWidget(self.height_size_x_le)
        size_res_layout.addWidget(QLabel("×"))
        size_res_layout.addWidget(self.height_size_y_le)
        size_res_layout.addWidget(QLabel(" Res:"))
        size_res_layout.addWidget(self.height_res_x_le)
        size_res_layout.addWidget(QLabel("×"))
        size_res_layout.addWidget(self.height_res_y_le)
        env_layout.addRow("Height Map Size (m):", size_res_layout)
        parent_layout.addWidget(env_group, 1)

    def _create_policy_group(self, parent_layout):
        policy_group = QGroupBox("Policy Settings")
        policy_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        policy_layout = QFormLayout()
        policy_layout.setLabelAlignment(Qt.AlignRight)
        policy_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        policy_layout.setSpacing(8)
        policy_group.setLayout(policy_layout)
        self.use_lstm_cb = NoWheelComboBox()
        self.use_lstm_cb.addItems(["True", "False"])
        self.use_lstm_cb.setCurrentText("False")
        policy_layout.addRow("Use LSTM:", self.use_lstm_cb)
        self.h_in_dim_le = QLineEdit("256")
        policy_layout.addRow("h_in Dim:", self.h_in_dim_le)
        self.c_in_dim_le = QLineEdit("256")
        policy_layout.addRow("c_in Dim:", self.c_in_dim_le)
        # ONNX File
        self.policy_file_le = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_policy_file)
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.policy_file_le)
        file_layout.addWidget(browse_btn)
        policy_layout.addRow("ONNX File:", file_layout)
        parent_layout.addWidget(policy_group, 0)

    # ---- Observation Scales Group ----
    def _create_obs_scales_group(self):
        obs_group = QGroupBox("Observation Scales")
        obs_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        obs_form = QFormLayout()
        obs_form.setLabelAlignment(Qt.AlignLeft)
        obs_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        obs_form.setSpacing(8)
        obs_group.setLayout(obs_form)
        settings = self.env_config[self.env_id_cb.currentText()]
        obs_scales = settings.get("obs_scales", {})
        for key in ["lin_vel", "ang_vel", "dof_pos", "dof_vel"]:
            default = obs_scales.get(key, 1.0)
            le = QLineEdit(str(default))
            le.setFixedWidth(50)
            obs_form.addRow(f"{key}:", le)
            self.obs_scales_le[key] = le
        self.scale_commands_cb = NoWheelComboBox()
        self.scale_commands_cb.addItems(["True", "False"])
        scale_commands = obs_scales.get("scale_commands", True)
        if scale_commands:
            self.scale_commands_cb.setCurrentText("True")
        else:
            self.scale_commands_cb.setCurrentText("False")
        obs_form.addRow("scale_commands:", self.scale_commands_cb)
        return obs_group

    def _create_random_group(self):
        random_group = QGroupBox("Random Settings")
        random_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)
        form_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form_layout.setSpacing(8)
        random_group.setLayout(form_layout)
        self.precision_cb = NoWheelComboBox()
        self.precision_cb.addItems(["low", "medium", "high", "ultra", "extreme"])
        self.precision_cb.setCurrentText("medium")
        form_layout.addRow("Precision:", self.precision_cb)
        self.sensor_noise_cb = NoWheelComboBox()
        self.sensor_noise_cb.addItems(["none", "low", "medium", "high", "ultra", "extreme"])
        self.sensor_noise_cb.setCurrentText("low")
        form_layout.addRow("Sensor Noise:", self.sensor_noise_cb)
        def create_slider_row(slider, min_val, max_val, init_val, scale, decimals):
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(init_val)
            value_label = QLabel(f"{init_val / scale:.{decimals}f}")
            slider.valueChanged.connect(lambda v: value_label.setText(f"{v / scale:.{decimals}f}"))
            h_layout = QHBoxLayout()
            h_layout.addWidget(slider)
            h_layout.addWidget(value_label)
            return h_layout
        self.init_noise_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Init Noise:", create_slider_row(self.init_noise_slider, 0, 100, 5, 100, 2))
        self.sliding_friction_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Sliding Friction:", create_slider_row(self.sliding_friction_slider, 0, 100, 80, 100, 2))
        self.torsional_friction_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Torsional Friction:", create_slider_row(self.torsional_friction_slider, 0, 10, 2, 100, 2))
        self.rolling_friction_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Rolling Friction:", create_slider_row(self.rolling_friction_slider, 0, 10, 1, 100, 2))
        self.friction_loss_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Friction Loss:", create_slider_row(self.friction_loss_slider, 0, 100, 10, 100, 2))
        self.action_delay_prob_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Action Delay Prob.:", create_slider_row(self.action_delay_prob_slider, 0, 100, 5, 100, 2))
        self.mass_noise_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Mass Noise:", create_slider_row(self.mass_noise_slider, 0, 50, 5, 100, 2))
        self.load_slider = NoWheelSlider(Qt.Horizontal)
        form_layout.addRow("Load:", create_slider_row(self.load_slider, 0, 200, 0, 10, 1))
        self.config_layout.addWidget(random_group)

    def _create_command_settings_group(self, parent_layout):
        command_group = QGroupBox("Command Settings")
        command_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
        )
        grid_layout = QGridLayout(command_group)
        grid_layout.addWidget(QLabel("Index"), 0, 0)
        grid_layout.addWidget(QLabel("Sensitivity"), 0, 1)
        grid_layout.addWidget(QLabel("Max Value"), 0, 2)
        grid_layout.addWidget(QLabel("Initial Value"), 0, 3)
        settings = self.env_config.get(self.env_id_cb.currentText())
        for i in range(6):
            label = QLabel(f"command[{i}]")
            sensitivity_le = QLineEdit("0.02")
            max_value_le = QLineEdit("1.5" if i in [0, 1, 2] else "1")
            init_value_widget = QLineEdit(settings["env"]["command_3_initial"]) if i == 3 else QLabel("0.0")
            grid_layout.addWidget(label, i + 1, 0)
            grid_layout.addWidget(sensitivity_le, i + 1, 1)
            grid_layout.addWidget(max_value_le, i + 1, 2)
            grid_layout.addWidget(init_value_widget, i + 1, 3)
            self.command_sensitivity_le_list.append(sensitivity_le)
            self.max_command_value_le_list.append(max_value_le)
            self.command_initial_value_le_list.append(init_value_widget)
        self.position_command_cb = QCheckBox("Position Command")
        self.position_command_cb.setChecked(False)
        row_position = 6 + 1 # Add the checkbox below the 6 command rows (index 1–6 used)
        grid_layout.addWidget(self.position_command_cb, row_position, 0, 1, 4, Qt.AlignLeft)
        parent_layout.addWidget(command_group)

    def _setup_key_visual_buttons(self, parent_layout):
        button_style = (
            "NonClickableButton { background-color: #3C3F41; border: none; color: #FFFFFF; "
            "font-size: 11px; padding: 10px; border-radius: 10px; min-width: 50px; min-height: 50px; }"
            "NonClickableButton:checked { background-color: #4E94D4; }"
        )
        key_group = QGroupBox("Command Input")
        key_layout = QVBoxLayout(key_group)
        key_layout.setSpacing(10)
        # Direction keys (W, A, S, D)
        dir_group = QGroupBox("command[0], command[2]")
        dir_layout = QGridLayout(dir_group)
        self.btn_up = NonClickableButton("W")
        self.btn_up.setStyleSheet(button_style)
        self.btn_up.setCheckable(True)
        dir_layout.addWidget(self.btn_up, 0, 1)
        self.btn_left = NonClickableButton("A")
        self.btn_left.setStyleSheet(button_style)
        self.btn_left.setCheckable(True)
        dir_layout.addWidget(self.btn_left, 1, 0)
        self.btn_right = NonClickableButton("D")
        self.btn_right.setStyleSheet(button_style)
        self.btn_right.setCheckable(True)
        dir_layout.addWidget(self.btn_right, 1, 2)
        self.btn_down = NonClickableButton("S")
        self.btn_down.setStyleSheet(button_style)
        self.btn_down.setCheckable(True)
        dir_layout.addWidget(self.btn_down, 1, 1)
        key_layout.addWidget(dir_group)
        # Other keys (I, O, P, J, K, L)
        other_group = QGroupBox("command[3], command[4], command[5]")
        other_layout = QGridLayout(other_group)
        self.btn_i = NonClickableButton("I")
        self.btn_i.setStyleSheet(button_style)
        self.btn_i.setCheckable(True)
        other_layout.addWidget(self.btn_i, 0, 0)
        self.btn_o = NonClickableButton("O")
        self.btn_o.setStyleSheet(button_style)
        self.btn_o.setCheckable(True)
        other_layout.addWidget(self.btn_o, 0, 1)
        self.btn_p = NonClickableButton("P")
        self.btn_p.setStyleSheet(button_style)
        self.btn_p.setCheckable(True)
        other_layout.addWidget(self.btn_p, 0, 2)
        self.btn_j = NonClickableButton("J")
        self.btn_j.setStyleSheet(button_style)
        self.btn_j.setCheckable(True)
        other_layout.addWidget(self.btn_j, 1, 0)
        self.btn_k = NonClickableButton("K")
        self.btn_k.setStyleSheet(button_style)
        self.btn_k.setCheckable(True)
        other_layout.addWidget(self.btn_k, 1, 1)
        self.btn_l = NonClickableButton("L")
        self.btn_l.setStyleSheet(button_style)
        self.btn_l.setCheckable(True)
        other_layout.addWidget(self.btn_l, 1, 2)
        key_layout.addWidget(other_group)
        # ZX group (command[1])
        zx_group = QGroupBox("command[1]")
        zx_layout = QHBoxLayout(zx_group)
        zx_style = (
            "NonClickableButton { background-color: #3C3F41; border: none; color: #FFFFFF; "
            "font-size: 11px; padding: 4px; border-radius: 10px; min-width: 30px; min-height: 30px; }"
            "NonClickableButton:checked { background-color: #4E94D4; }"
        )
        self.btn_z = NonClickableButton("Z")
        self.btn_z.setStyleSheet(zx_style)
        self.btn_z.setCheckable(True)
        zx_layout.addWidget(self.btn_z)
        self.btn_x = NonClickableButton("X")
        self.btn_x.setStyleSheet(zx_style)
        self.btn_x.setCheckable(True)
        zx_layout.addWidget(self.btn_x)
        key_layout.addWidget(zx_group)
        parent_layout.addWidget(key_group, 1)
        # Set key mapping
        self.key_mapping = {
            Qt.Key_W: (self.btn_up, 0, +1.0),
            Qt.Key_S: (self.btn_down, 0, -1.0),
            Qt.Key_A: (self.btn_left, 2, +1.0),
            Qt.Key_D: (self.btn_right, 2, -1.0),
            Qt.Key_Z: (self.btn_z, 1, -1.0),
            Qt.Key_X: (self.btn_x, 1, +1.0),
            Qt.Key_I: (self.btn_i, 3, +1.0),
            Qt.Key_J: (self.btn_j, 3, -1.0),
            Qt.Key_O: (self.btn_o, 4, +1.0),
            Qt.Key_K: (self.btn_k, 4, -1.0),
            Qt.Key_P: (self.btn_p, 5, +1.0),
            Qt.Key_L: (self.btn_l, 5, -1.0)
        }

    def _apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }
            QLineEdit, QComboBox, QSlider {
                padding: 4px;
            }
            QPushButton {
                background-color: #007ACC;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #4E94D4;
            }
            QPushButton:disabled {
                background-color: #A0A0A0;
            }
            QPushButton:hover:!disabled {
                background-color: #005999;
            }
        """)

    def browse_policy_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Policy ONNX File", os.path.join(os.getcwd(), "weights"),
            "ONNX Files (*.onnx)"
        )
        if file_path:
            self.policy_file_le.setText(file_path)

    def open_hardware_settings(self):
        dialog = HardwareSettingsDialog(self.hardware_settings, self)
        if dialog.exec_() == QDialog.Accepted:
            self.hardware_settings = dialog.get_settings()

    def start_test(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("테스트 실행 중...")
        self._update_status_label()
        # Disable the Position Command checkbox
        self.position_command_cb.setEnabled(False)
        config = self._gather_config()
        if config is None:
            return
        policy_file_path = self.policy_file_le.text().strip()
        if not policy_file_path or not os.path.isfile(policy_file_path):
            QMessageBox.critical(self, "Error", "유효한 ONNX 파일을 선택해주세요.")
            self.position_command_cb.setEnabled(True)
            self._reset_ui_after_test()
            return
        self.tester = Tester()
        self.tester.load_config(config)
        self.tester.load_policy(policy_file_path)
        self._init_default_command_values()
        for i, value in enumerate(self.current_command_values):
            self.tester.update_command(i, value)
        self.tester.stepFinished.connect(self.send_current_command)
        self.thread = QThread()
        self.worker = TesterWorker(self.tester)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_test_finished)
        self.worker.error.connect(self.on_test_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _gather_config(self):
        try:
            config = {
                "env": {
                    "id": self.env_id_cb.currentText(),
                    "terrain": self.terrain_id_cb.currentText(),
                    "action_in_state": self.action_in_state_cb.currentText() == "True",
                    "max_duration": float(self.max_duration_le.text().strip()),
                    "observation_dim": int(self.observation_dim_le.text().strip()),
                    "command_dim": int(self.command_dim_le.text().strip()),
                    "action_dim": int(self.action_dim_le.text().strip()),
                    "num_stack": int(self.num_stack_cb.currentText()),
                    "external_sensors": self.external_sensors_cb.currentText(),
                    "external_sensors_Hz": self.external_sensors_Hz_cb.currentText(),
                    "height_map": {"x_size": float(self.height_size_x_le.text().strip()),
                                   "y_size": float(self.height_size_y_le.text().strip()),
                                   "x_res": int(self.height_res_x_le.text().strip()),
                                   "y_res": int(self.height_res_y_le.text().strip()),}
                },
                "policy": {
                    "use_lstm": self.use_lstm_cb.currentText() == "True",
                    "h_in_dim": int(self.h_in_dim_le.text().strip()),
                    "c_in_dim": int(self.c_in_dim_le.text().strip()),
                    "onnx_file": os.path.basename(self.policy_file_le.text())
                },
                "obs_scales": {
                    key: float(le.text())
                    for key, le in self.obs_scales_le.items()
                },
                "command":{
                    "position_command": self.position_command_cb.isChecked(),
                },
                "random": {
                    "precision": self.precision_cb.currentText(),
                    "sensor_noise": self.sensor_noise_cb.currentText(),
                    "init_noise": self.init_noise_slider.value() / 100.0,
                    "sliding_friction": self.sliding_friction_slider.value() / 100.0,
                    "torsional_friction": self.torsional_friction_slider.value() / 100.0,
                    "rolling_friction": self.rolling_friction_slider.value() / 100.0,
                    "friction_loss": self.friction_loss_slider.value() / 100.0,
                    "action_delay_prob": self.action_delay_prob_slider.value() / 100.0,
                    "mass_noise": self.mass_noise_slider.value() / 100.0,
                    "load": self.load_slider.value() / 10.0
                },
                "hardware": {k: float(v) for k, v in self.hardware_settings.items()}
            }
            config["obs_scales"]["scale_commands"] = self.scale_commands_cb.currentText() == "True"
            # Load random_table
            cur_file_path = os.path.abspath(__file__)
            config_path = os.path.join(os.path.dirname(cur_file_path), "../config/random_table.yaml")
            config_path = os.path.abspath(config_path)
            with open(config_path) as f:
                random_config = yaml.full_load(f)
            config["random_table"] = random_config["random_table"]
            return config
        except Exception as e:
            QMessageBox.critical(self, "Error", f"파라미터 설정 오류: {e}")
            self._reset_ui_after_test()
            return None

    def _reset_ui_after_test(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("대기 중")

    def reset_command_buttons(self):
        for key in list(self.active_keys.keys()):
            btn, cmd_index, _ = self.key_mapping[key]
            btn.setChecked(False)
            default_value = self._get_default_command_value(cmd_index)
            self._update_command_button(cmd_index, default_value)
            self.active_keys.pop(key)

    def on_test_finished(self):
        self.reset_command_buttons()
        self.status_label.setText("테스트 완료")
        self._reset_ui_after_test()
        # Re-enable the Position Command checkbox
        self.position_command_cb.setEnabled(True)
        reply = QMessageBox.question(
            self,
            "Report 확인",
            "테스트가 종료되었습니다. 리포트를 열람하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            policy_file_path = self.policy_file_le.text().strip()
            report_path = os.path.join(os.path.dirname(policy_file_path), "report.pdf")
            if os.path.isfile(report_path):
                QDesktopServices.openUrl(QUrl.fromLocalFile(report_path))
            else:
                QMessageBox.warning(self, "Warning", "리포트 파일(report.pdf)이 존재하지 않습니다.")

    def on_test_error(self, error_msg):
        QMessageBox.critical(self, "Test Error", error_msg)
        self.status_label.setText("오류 발생")
        self._reset_ui_after_test()

    def stop_test(self):
        if self.tester:
            try:
                self.tester.stop()
                self.status_label.setText("테스트 중지 요청")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"테스트 중지 오류: {e}")
        self.reset_command_buttons()
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)