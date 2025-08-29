import os
import yaml

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QPushButton, QLabel, QMessageBox, QMainWindow,
    QFileDialog, QGroupBox, QScrollArea, QLineEdit, QCheckBox, QDialog
)
from PyQt5.QtCore import QThread, Qt, QEvent, QUrl
from PyQt5.QtGui import QDesktopServices, QIcon
from core.tester import Tester
from ui.utils import to_float, to_int, normalize_numkey_float_values
from ui.custom_widgets import NoWheelComboBox, NoWheelSlider, NonClickableButton
from ui.dialogs.hardware_settings import HardwareSettingsDialog
from ui.dialogs.observation_settings import ObservationSettingsDialog
from ui.workers import TesterWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        cur_file_path = os.path.abspath(__file__)
        config_path = os.path.join(os.path.dirname(cur_file_path), "../config/env_table.yaml")
        config_path = os.path.abspath(config_path)
        with open(config_path) as f:
            self.env_config = yaml.full_load(f)

        self.obs_types = ["dof_pos", "dof_vel", "lin_vel_x", "lin_vel_y", "lin_vel_z", "ang_vel", "projected_gravity", "height_map", "last_action"]

        # Per-environment observation settings cache
        self.obs_settings_by_env = {}

        self._init_window()
        self._init_variables()
        self._setup_ui()
        self._init_default_command_values()
        self.status_label.setText("Waiting ...")
        self.env_id_cb.currentTextChanged.connect(self.update_defaults)
        self.update_defaults(self.env_id_cb.currentText())
        self._last_run_had_error = False

    def _init_window(self):
        app_icon_path = os.path.join(os.path.dirname(__file__), "icon", "window_icon.png")
        self.setWindowIcon(QIcon(app_icon_path))
        self.setWindowTitle("cosim - v1.5.0")
        self.resize(750, 970)
        self.installEventFilter(self)

    def _init_variables(self):
        self.key_mapping = {}
        self.active_keys = {}
        self.thread = None
        self.worker = None
        self.tester = None
        self.current_command_values = [0.0] * 6
        self.command_sensitivity_le_list = []
        self.max_command_value_le_list = []
        self.command_initial_value_le_list = []
        self.command_timer = None
        self.hardware_settings = {}
        self.hardware_settings_by_env = {}

        # Whether the user manually changed observation settings via dialog (kept for reference; cache now used)
        self.observation_overridden_by_user = False

        # Initial observation_settings (will be overridden by update_defaults for the first env)
        self.observation_settings = {
            "stacked_obs_order": [],
            "non_stacked_obs_order": [],
            "stack_size": 3,
            "command_dim": 6,
            "command_scales": {"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0},
            "height_map": {"size_x": 1.0, "size_y": 0.6, "res_x": 15, "res_y": 9, "freq": 50, "scale": 1.0},
            "dof_pos": None,
            "dof_vel": None,
            "lin_vel_x": None,
            "lin_vel_y": None,
            "lin_vel_z": None,
            "ang_vel": None,
            "projected_gravity": None,
            "last_action": None,
        }

    def _init_default_command_values(self):
        """Initialize current_command_values from the UI 'Initial Value' fields."""
        try:
            vals = []
            for widget in self.command_initial_value_le_list:
                if isinstance(widget, QLineEdit):
                    vals.append(float(widget.text()))
                elif isinstance(widget, QLabel):
                    vals.append(float(widget.text()))
                else:
                    vals.append(0.0)
            self.current_command_values = vals if len(vals) == 6 else [0.0] * 6
        except Exception:
            self.current_command_values = [0.0] * 6

    # -------- observation defaults/caching --------
    def _make_observation_defaults(self, env_id: str):
        env_cfg = self.env_config.get(env_id, {}) or {}
        cmd_cfg_raw = env_cfg.get("command", {}) if isinstance(env_cfg.get("command", {}), dict) else {}
        obs_scales = env_cfg.get("obs_scales", {}) or {}
        command_scales_cfg = normalize_numkey_float_values(env_cfg.get("command_scales", {}))
        stacked_list = env_cfg.get("stacked_obs_order", []) or []
        non_stacked_list = env_cfg.get("non_stacked_obs_order", []) or []
        stack_size_yaml = to_int(env_cfg.get("stack_size", 3), 3)

        # Apply default frequency and scale
        obs_dict = {}
        for obs in stacked_list:
            obs_dict[obs] = {"freq": 50, "scale": to_float(obs_scales.get(obs, 1.0), 1.0)}

        for obs in non_stacked_list:
            obs_dict[obs] = {"freq": 50, "scale": to_float(obs_scales.get(obs, 1.0), 1.0)}

        for obs in self.obs_types:
            if obs not in obs_dict:
                obs_dict[obs] = None

        cmd_dim = to_int(cmd_cfg_raw.get("command_dim", 6), 6)

        merged_command_scales = {}
        for i in range(cmd_dim):
            key = str(i)
            merged_command_scales[key] = to_float(command_scales_cfg.get(key, 1.0), 1.0)

        height_in_order = ("height_map" in stacked_list) or ("height_map" in non_stacked_list)
        if height_in_order:
            height_map_yaml = env_cfg.get("height_map", {}) if isinstance(env_cfg.get("height_map", {}), dict) else {}
            height_map_val = {
                "size_x": to_float(height_map_yaml.get("size_x", 1.0)),
                "size_y": to_float(height_map_yaml.get("size_y", 0.6)),
                "res_x": to_int(height_map_yaml.get("res_x", 15)),
                "res_y": to_int(height_map_yaml.get("res_y", 9)),
                "freq": 50,
                "scale": 1.0,
            }
        else:
            height_map_val = None

        return {
            "stacked_obs_order": stacked_list,
            "non_stacked_obs_order": non_stacked_list,
            "stack_size": stack_size_yaml,
            "command_dim": cmd_dim,
            "command_scales": merged_command_scales,
            "height_map": height_map_val,
            **obs_dict
        }

    def _ensure_observation_defaults(self):
        # If not in cache, create defaults for the current env
        env_id = self.env_id_cb.currentText()
        if env_id not in self.obs_settings_by_env:
            self.obs_settings_by_env[env_id] = self._make_observation_defaults(env_id)
        # Sync current observation_settings with latest cache
        self.observation_settings = (self.obs_settings_by_env[env_id]).copy()

    # ---------------- per-env hardware helpers (like observation) ----------------
    def _make_hardware_defaults(self, env_id: str):
        """Build default hardware settings for the env from YAML (shallow copy)."""
        env_cfg = self.env_config.get(env_id, {}) or {}
        hw = env_cfg.get("hardware", {}) or {}
        # Keep string values (editable in dialog). Numeric conversion is done in _gather_config.
        return hw.copy()

    def _ensure_hardware_defaults(self):
        """Ensure current env has cached hardware settings and sync self.hardware_settings."""
        env_id = self.env_id_cb.currentText()
        if env_id not in self.hardware_settings_by_env:
            self.hardware_settings_by_env[env_id] = self._make_hardware_defaults(env_id)
        self.hardware_settings = (self.hardware_settings_by_env[env_id]).copy()

    def update_defaults(self, new_env_id):
        settings = self.env_config.get(new_env_id, {}) or {}
        if new_env_id in self.hardware_settings_by_env:
            self.hardware_settings = (self.hardware_settings_by_env[new_env_id]).copy()
        else:
            self.hardware_settings = self._make_hardware_defaults(new_env_id)
            self.hardware_settings_by_env[new_env_id] = (self.hardware_settings).copy()

        cmd_cfg = settings.get("command", {}) if isinstance(settings.get("command", {}), dict) else {}

        # UI upper bounds (example retained)
        command_0_max = "1.5"
        command_2_max = "1.5"
        if self.max_command_value_le_list:
            self.max_command_value_le_list[0].setText(command_0_max)
            self.max_command_value_le_list[2].setText(command_2_max)

        # command[3] initial value (accept float/int)
        if self.command_initial_value_le_list and isinstance(self.command_initial_value_le_list[3], QLineEdit):
            c3 = cmd_cfg.get("command_3_initial", 0.0)
            self.command_initial_value_le_list[3].setText(str(to_float(c3, 0.0)))

        # On environment change: observation settings via cache or defaults
        if new_env_id in self.obs_settings_by_env:
            self.observation_settings = (self.obs_settings_by_env[new_env_id]).copy()
        else:
            self.observation_settings = self._make_observation_defaults(new_env_id)
            self.obs_settings_by_env[new_env_id] = (self.observation_settings).copy()

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
            widget = self.command_initial_value_le_list[index]
            if isinstance(widget, (QLineEdit, QLabel)):
                return float(widget.text())
            return 0.0
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
        # Apply key-driven deltas within bounds, update tester and status
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

    # ---------------- UI SETUP ----------------

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        top_h_layout = QHBoxLayout()
        top_h_layout.setSpacing(15)
        main_layout.addLayout(top_h_layout)

        # Left: scroll area (Policy placed below Environment)
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        top_h_layout.addWidget(config_scroll, 3)
        config_widget = QWidget()
        config_scroll.setWidget(config_widget)
        self.config_layout = QVBoxLayout(config_widget)
        self.config_layout.setContentsMargins(10, 10, 10, 10)
        self.config_layout.setSpacing(15)

        # Vertical: Policy under Environment
        self._create_env_group(self.config_layout)
        self._create_policy_group(self.config_layout)

        # Random Settings group
        self._create_random_group()

        # Place Event Input on the left (under Random Settings)
        self._create_event_input_group(self.config_layout)

        # Right: Command Settings / Command Input
        right_v_layout = QVBoxLayout()
        top_h_layout.addLayout(right_v_layout, 1)
        self._create_command_settings_group(right_v_layout)
        self._setup_key_visual_buttons(right_v_layout)

        self.status_label = QLabel("대기 중")
        self.status_label.setStyleSheet("font-size: 14px;")
        main_layout.addWidget(self.status_label)

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

    def deactivate_push_trigger(self):
        if self.tester:
            self.tester.deactivate_push_event()

    # --------- CONFIG GROUPS ---------

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

        default_env = list(self.env_config.keys())[0]
        self.env_id_cb.setCurrentText(default_env)
        env_layout.addRow("ID:", self.env_id_cb)

        self.max_duration_le = QLineEdit("120.0")
        env_layout.addRow("Max Duration (s):", self.max_duration_le)

        settings_btn = QPushButton("Hardware Settings")
        settings_btn.clicked.connect(self.open_hardware_settings)
        env_layout.addRow("Hardware:", settings_btn)

        obs_settings_btn = QPushButton("Observation Settings")
        obs_settings_btn.clicked.connect(self.open_observation_settings)
        env_layout.addRow("Observation:", obs_settings_btn)

        self.terrain_id_cb = NoWheelComboBox()
        self.terrain_id_cb.addItems([
            'flat', 'rocky_easy', 'rocky_hard',
            'slope_easy', 'slope_hard',
            'stairs_up_easy', 'stairs_up_normal', 'stairs_up_hard'
        ])
        self.terrain_id_cb.setCurrentText("flat")
        env_layout.addRow("Terrain:", self.terrain_id_cb)
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
        self.policy_file_le = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_policy_file)
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.policy_file_le)
        file_layout.addWidget(browse_btn)
        policy_layout.addRow("ONNX File:", file_layout)
        parent_layout.addWidget(policy_group, 0)

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

        # command[3] initial value is taken from the env's 'command' section
        settings = self.env_config.get(self.env_id_cb.currentText(), {}) or {}
        cmd_cfg = settings.get("command", {}) if isinstance(settings.get("command", {}), dict) else {}
        cmd3_init = str(to_float(cmd_cfg.get("command_3_initial", 0.0), 0.0))

        for i in range(6):  # indices 0~5
            label = QLabel(f"command[{i}]")
            sensitivity_le = QLineEdit("0.02")
            max_value_le = QLineEdit("1.5" if i in [0, 1, 2] else "1")
            init_value_widget = QLineEdit(cmd3_init) if i == 3 else QLabel("0.0")
            grid_layout.addWidget(label, i + 1, 0)
            grid_layout.addWidget(sensitivity_le, i + 1, 1)
            grid_layout.addWidget(max_value_le, i + 1, 2)
            grid_layout.addWidget(init_value_widget, i + 1, 3)
            self.command_sensitivity_le_list.append(sensitivity_le)
            self.max_command_value_le_list.append(max_value_le)
            self.command_initial_value_le_list.append(init_value_widget)
        self.position_command_cb = QCheckBox("Position Command")
        self.position_command_cb.setChecked(False)
        row_position = 6 + 1
        grid_layout.addWidget(self.position_command_cb, row_position, 0, 1, 4, Qt.AlignLeft)
        parent_layout.addWidget(command_group)

    def _setup_key_visual_buttons(self, parent_layout):
        button_style = (
            "NonClickableButton { background-color: #3C3F41; border: none; color: #FFFFFF; "
            "font-size: 11px; padding: 10px; border-radius: 10px; min-width: 36px; min-height: 36px; }"
            "NonClickableButton:checked { background-color: #4E94D4; }"
        )
        key_group = QGroupBox("Command Input")
        key_layout = QVBoxLayout(key_group)
        key_layout.setSpacing(8)

        dir_group = QGroupBox("command[0], command[2]")
        dir_layout = QGridLayout(dir_group)
        self.btn_up = NonClickableButton("W"); self.btn_up.setStyleSheet(button_style); self.btn_up.setCheckable(True); dir_layout.addWidget(self.btn_up, 0, 1)
        self.btn_left = NonClickableButton("A"); self.btn_left.setStyleSheet(button_style); self.btn_left.setCheckable(True); dir_layout.addWidget(self.btn_left, 1, 0)
        self.btn_right = NonClickableButton("D"); self.btn_right.setStyleSheet(button_style); self.btn_right.setCheckable(True); dir_layout.addWidget(self.btn_right, 1, 2)
        self.btn_down = NonClickableButton("S"); self.btn_down.setStyleSheet(button_style); self.btn_down.setCheckable(True); dir_layout.addWidget(self.btn_down, 1, 1)
        key_layout.addWidget(dir_group)

        other_group = QGroupBox("command[3], command[4], command[5]")
        other_layout = QGridLayout(other_group)
        self.btn_i = NonClickableButton("I"); self.btn_i.setStyleSheet(button_style); self.btn_i.setCheckable(True); other_layout.addWidget(self.btn_i, 0, 0)
        self.btn_o = NonClickableButton("O"); self.btn_o.setStyleSheet(button_style); self.btn_o.setCheckable(True); other_layout.addWidget(self.btn_o, 0, 1)
        self.btn_p = NonClickableButton("P"); self.btn_p.setStyleSheet(button_style); self.btn_p.setCheckable(True); other_layout.addWidget(self.btn_p, 0, 2)
        self.btn_j = NonClickableButton("J"); self.btn_j.setStyleSheet(button_style); self.btn_j.setCheckable(True); other_layout.addWidget(self.btn_j, 1, 0)
        self.btn_k = NonClickableButton("K"); self.btn_k.setStyleSheet(button_style); self.btn_k.setCheckable(True); other_layout.addWidget(self.btn_k, 1, 1)
        self.btn_l = NonClickableButton("L"); self.btn_l.setStyleSheet(button_style); self.btn_l.setCheckable(True); other_layout.addWidget(self.btn_l, 1, 2)
        key_layout.addWidget(other_group)

        zx_group = QGroupBox("command[1]")
        zx_layout = QHBoxLayout(zx_group)
        zx_style = (
            "NonClickableButton { background-color: #3C3F41; border: none; color: #FFFFFF; "
            "font-size: 11px; padding: 4px; border-radius: 10px; min-width: 22px; min-height: 22px; }"
            "NonClickableButton:checked { background-color: #4E94D4; }"
        )
        self.btn_z = NonClickableButton("Z"); self.btn_z.setStyleSheet(zx_style); self.btn_z.setCheckable(True); zx_layout.addWidget(self.btn_z)
        self.btn_x = NonClickableButton("X"); self.btn_x.setStyleSheet(zx_style); self.btn_x.setCheckable(True); zx_layout.addWidget(self.btn_x)
        key_layout.addWidget(zx_group)
        parent_layout.addWidget(key_group, 1)

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
        env_id = self.env_id_cb.currentText()
        self._ensure_hardware_defaults()
        dialog = HardwareSettingsDialog((self.hardware_settings).copy(), self)
        if dialog.exec_() == QDialog.Accepted:
            self.hardware_settings = dialog.get_settings()
            # Save back to per-env cache so it persists after env switches
            self.hardware_settings_by_env[env_id] = (self.hardware_settings).copy()

    def open_observation_settings(self):
        # Open the dialog with the latest settings for the current env
        env_id = self.env_id_cb.currentText()
        self._ensure_observation_defaults()  # Sync cache
        dialog = ObservationSettingsDialog((self.observation_settings).copy(), self)
        if dialog.exec_() == QDialog.Accepted:
            self.observation_settings = dialog.get_settings()
            # Save current env settings back into the cache (so they restore next time)
            self.obs_settings_by_env[env_id] = (self.observation_settings).copy()
            # Mark that user manually changed settings (for reference)
            self.observation_overridden_by_user = True

    # ---------------- Run / Gather Config ----------------

    def start_test(self):
        # Ensure latest settings for the current env
        self._last_run_had_error = False
        self._ensure_observation_defaults()
        self._ensure_hardware_defaults()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Test running...")
        self._update_status_label()
        self.position_command_cb.setEnabled(False)
        config = self._gather_config()
        if config is None:
            return
        policy_file_path = self.policy_file_le.text().strip()
        if not policy_file_path or not os.path.isfile(policy_file_path):
            QMessageBox.critical(self, "Error", "Please select a valid ONNX file.")
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
            self._ensure_hardware_defaults()
            # hardware: convert numeric strings to float where applicable
            hardware_numeric = {k: to_float(v, v) for k, v in self.hardware_settings.items()}

            # observation: copy latest settings for the current env
            env_id = self.env_id_cb.currentText()
            self._ensure_observation_defaults()
            observation = (self.observation_settings).copy()

            # height_map patching with env YAML defaults
            env_cfg = self.env_config.get(env_id, {}) or {}
            yaml_hm = env_cfg.get("height_map", {}) if isinstance(env_cfg.get("height_map", {}), dict) else {}
            yaml_hm_defaults = {
                "size_x": to_float(yaml_hm.get("size_x", 1.0)),
                "size_y": to_float(yaml_hm.get("size_y", 0.6)),
                "res_x": to_int(yaml_hm.get("res_x", 15)),
                "res_y": to_int(yaml_hm.get("res_y", 9)),
            }

            hm_val = observation.get("height_map", None)
            if isinstance(hm_val, dict):
                hm_val.setdefault("size_x", yaml_hm_defaults["size_x"])
                hm_val.setdefault("size_y", yaml_hm_defaults["size_y"])
                hm_val.setdefault("res_x", yaml_hm_defaults["res_x"])
                hm_val.setdefault("res_y", yaml_hm_defaults["res_y"])
                hm_val.setdefault("freq", 50)
                hm_val.setdefault("scale", 1.0)
                observation["height_map"] = hm_val
            elif hm_val is None:
                observation["height_map"] = None
            else:
                observation["height_map"] = None

            config = {
                "env": {
                    "id": env_id,
                    "terrain": self.terrain_id_cb.currentText(),
                    "max_duration": float(self.max_duration_le.text().strip()),
                    "position_command": self.position_command_cb.isChecked()
                },
                "observation": observation,
                "policy": {
                    "use_lstm": self.use_lstm_cb.currentText() == "True",
                    "h_in_dim": int(self.h_in_dim_le.text().strip()),
                    "c_in_dim": int(self.c_in_dim_le.text().strip()),
                    "onnx_file": os.path.basename(self.policy_file_le.text())
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
                "hardware": hardware_numeric
            }

            # random_table (only if present)
            cur_file_path = os.path.abspath(__file__)
            random_path = os.path.join(os.path.dirname(cur_file_path), "../config/random_table.yaml")
            random_path = os.path.abspath(random_path)
            if os.path.isfile(random_path):
                with open(random_path) as f:
                    random_config = yaml.full_load(f)
                if isinstance(random_config, dict) and "random_table" in random_config:
                    config["random_table"] = random_config["random_table"]
            return config
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Parameter setting error: {e}")
            self._reset_ui_after_test()
            return None

    def _reset_ui_after_test(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Waiting ...")

    def reset_command_buttons(self):
        for key in list(self.active_keys.keys()):
            btn, cmd_index, _ = self.key_mapping[key]
            btn.setChecked(False)
            default_value = self._get_default_command_value(cmd_index)
            self._update_command_button(cmd_index, default_value)
            self.active_keys.pop(key)

    def on_test_finished(self):
        self.reset_command_buttons()
        the_text = "Test complete"
        self.status_label.setText(the_text)
        self._reset_ui_after_test()
        self.position_command_cb.setEnabled(True)

        if not self._last_run_had_error:
            reply = QMessageBox.question(
                self,
                "Check Report",
                "Test has finished. Would you like to view the report?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                policy_file_path = self.policy_file_le.text().strip()
                report_path = os.path.join(os.path.dirname(policy_file_path), "report.pdf")
                if os.path.isfile(report_path):
                    QDesktopServices.openUrl(QUrl.fromLocalFile(report_path))
                else:
                    QMessageBox.warning(self, "Warning", "Report file (report.pdf) does not exist.")
        
    def on_test_error(self, error_msg):
        self._last_run_had_error = True
        QMessageBox.critical(self, "Test Error", error_msg)
        self.status_label.setText("Error occurred")
        self._reset_ui_after_test()

    def stop_test(self):
        if self.tester:
            try:
                self.tester.stop()
                self.status_label.setText("Test stop requested")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Test stop error: {e}")
        self.reset_command_buttons()
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)