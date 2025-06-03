import matplotlib
matplotlib.use('Agg')  # Suppress GUI backend warning
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import math
import time
from cycler import cycler

# Use seaborn whitegrid style and set a clean color palette (excluding yellow)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 1.5,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    # Subplot margins
    'figure.subplot.left': 0.1,
    'figure.subplot.right': 0.9,
    'figure.subplot.bottom': 0.05,
    'figure.subplot.top': 0.95,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    # Color palette (excluding yellow)
    'axes.prop_cycle': cycler('color', [
        '#E15759',  # red
        '#4E79A7',  # blue
        '#B07AA1',  # purple
        '#59A14F',  # green
        '#9C755F'   # brown
    ])
})

# Default marker size for plots
DEFAULT_MARKER_SIZE = 1


class Reporter:
    def __init__(self, report_path, config):
        """
        Initializes the reporter with a PDF output path and configuration.
        Parameters:
            report_path (str): Path to save the PDF report.
            config (dict): Dictionary containing configuration values.
        """
        self.report_path = report_path
        self.config = config
        self.history = {}
        self.timesteps = 0

    def write_info(self, info):
        """
        Logs a dictionary of key-value pairs at each timestep.
        Parameters:
            info (dict): Logged data such as {'dt': 0.1, 'action': [...], ...}
        """
        self.timesteps += 1
        for key, value in info.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def _build_config_rows(self, config, indent=0):
        """
        Converts a nested config dictionary into flat rows for a parameter table.
        Applies indentation for nested keys.
        """
        rows = []
        indent_str = "    " * indent  # 들여쓰기는 공백 4칸씩
        for key, value in config.items():
            if isinstance(value, dict):
                rows.append([f"{indent_str}{key}", ""])
                rows.extend(self._build_config_rows(value, indent + 1))
            elif isinstance(value, list):
                joined = ", ".join(map(str, value))
                rows.append([f"{indent_str}{key}", joined])
            else:
                rows.append([f"{indent_str}{key}", str(value)])
        return rows

    def generate_report(self):
        """
        Generates a multi-page PDF report summarizing logged data and configuration.
        The report includes plots and tables formatted for A4 portrait pages.
        """
        PAGE_SIZE = (8.27, 11.69)
        dt = float(self.history.get('dt', [1])[0])
        times = np.arange(self.timesteps) * dt

        with PdfPages(self.report_path) as pdf:
            # ===============================
            # Cover Page
            # ===============================
            fig_cover = plt.figure(figsize=PAGE_SIZE)
            plt.axis('off')

            logo_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "ui", "icon", 'cocelo_logo_large.png'))
            try:
                logo = plt.imread(logo_path)
                dpi = fig_cover.get_dpi()
                fig_width_px = fig_cover.get_figwidth() * dpi
                logo_width = logo.shape[1]
                xo = int((fig_width_px - logo_width) / 2)
                fig_cover.figimage(logo, xo=xo, yo=50, alpha=0.8, zorder=1)
            except FileNotFoundError:
                print(f"Warning: Logo file not found at {logo_path}")

            # Title
            plt.text(0.5, 0.72,
                     "Test Report",
                     ha='center', va='center',
                     fontsize=38, fontweight='bold')

            # Timestamp
            plt.text(0.5, 0.63,
                     f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                     ha='center', va='center',
                     fontsize=14)

            # Description
            description = ("This report provides an in-depth overview of key performance metrics,\n"
                           " encompassing the following analyses:")
            plt.text(0.5, 0.57,
                     description,
                     ha='center', va='center',
                     fontsize=12)

            # Bullet list
            bullet_x = 0.35
            bullet_y_positions = [0.51, 0.47, 0.43, 0.39]
            bullets = [
                "•   Set Points vs. Actual State",
                "•   Command vs. Actual Values",
                "•   Action oscillation",
                "•   Actual Torques",
            ]
            for bullet, y in zip(bullets, bullet_y_positions):
                plt.text(bullet_x, y,
                         bullet,
                         ha='left', va='center',
                         fontsize=12)

            pdf.savefig(fig_cover)
            plt.close(fig_cover)

            # ===============================
            # 1. Set Points vs. Actual State
            # ===============================
            if ('set_points' in self.history and 'cur_state' in self.history):
                set_points = np.array(self.history['set_points'], dtype=float)
                cur_state = np.array(self.history['cur_state'], dtype=float)
                if set_points.ndim == 1:
                    set_points = set_points.reshape(-1, 1)
                    cur_state = cur_state.reshape(-1, 1)
                n_dims = set_points.shape[1]
                n_cols = 1 if n_dims == 1 else 2
                n_rows = math.ceil(n_dims / n_cols)
                fig, axes = plt.subplots(n_rows, n_cols, figsize=PAGE_SIZE)
                fig.suptitle("Set Points vs. Actual State", fontsize=16, fontweight='bold')

                if n_rows * n_cols == 1:
                    axes = np.array([axes])
                else:
                    axes = np.array(axes).flatten()

                for idx in range(n_dims):
                    ax = axes[idx]
                    ax.plot(times, set_points[:, idx], marker='o', linestyle='-', markersize=DEFAULT_MARKER_SIZE,
                            label="Set Point")
                    ax.plot(times, cur_state[:, idx], marker='^', linestyle='-.', markersize=DEFAULT_MARKER_SIZE,
                            label="Current State")
                    ax.set_xlabel("Time (s)", fontsize=10)
                    ax.set_ylabel(f"Dimension {idx}", fontsize=10)
                    ax.set_title(f"Dimension {idx}", fontsize=12)
                    ax.legend(fontsize=8)
                    ax.grid(True)

                for idx in range(n_dims, len(axes)):
                    fig.delaxes(axes[idx])

                fig.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

            # ===============================
            # 2. Command vs. Actual Values
            # ===============================
            command_keys = [
                ("lin_vel_x_command", "lin_vel_x", "Linear Velocity X"),
                ("lin_vel_y_command", "lin_vel_y", "Linear Velocity Y"),
                ("ang_vel_z_command", "ang_vel_z", "Angular Velocity Z")
            ]
            optional_keys = [
                ("pos_z_command", "pos_z", "Position Z"),
                ("ang_roll_command", "ang_roll", "Angular Roll"),
                ("ang_pitch_command", "ang_pitch", "Angular Pitch")
            ]
            plot_keys = []
            for cmd_key, actual_key, label in command_keys + optional_keys:
                if cmd_key in self.history and actual_key in self.history:
                    plot_keys.append((cmd_key, actual_key, label))

            if plot_keys:
                n_plots = len(plot_keys)
                if n_plots <= 2:
                    n_cols = n_plots
                    n_rows = 1
                else:
                    n_cols = 2
                    n_rows = math.ceil(n_plots / 2)
                fig, axes = plt.subplots(n_rows, n_cols, figsize=PAGE_SIZE)
                fig.suptitle("Command vs. Actual Values", fontsize=16, fontweight='bold')

                if n_rows * n_cols == 1:
                    axes = np.array([axes])
                else:
                    axes = np.array(axes).flatten()

                for idx, (cmd_key, actual_key, label) in enumerate(plot_keys):
                    cmd_values = np.array(self.history[cmd_key], dtype=float)
                    actual_values = np.array(self.history[actual_key], dtype=float)
                    ax = axes[idx]
                    ax.plot(times, cmd_values, marker='o', linestyle='-', markersize=DEFAULT_MARKER_SIZE,
                            label=f"{label} Command")
                    if label == 'Position Z':
                        ax.plot(times, actual_values, marker='x', linestyle='--', markersize=DEFAULT_MARKER_SIZE,
                                label=f"{label} Actual (Absolute)")
                    else:
                        ax.plot(times, actual_values, marker='x', linestyle='--', markersize=DEFAULT_MARKER_SIZE,
                                label=f"{label} Actual")
                    ax.set_xlabel("Time (s)", fontsize=10)
                    ax.set_ylabel(label, fontsize=10)
                    ax.set_title(f"{label}", fontsize=12)
                    ax.legend(fontsize=8)
                    ax.grid(True)

                for idx in range(len(plot_keys), len(axes)):
                    fig.delaxes(axes[idx])

                fig.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

            # ===============================
            # 3. Action Oscillation and Torques
            # ===============================
            if ('torque' in self.history and 'action_diff_RMS' in self.history):
                fig, axes = plt.subplots(2, 1, figsize=PAGE_SIZE)
                fig.suptitle("Action Difference (Oscillation) and Torques", fontsize=16, fontweight='bold')

                diffs = np.array(self.history['action_diff_RMS'], dtype=float)
                axes[0].plot(times, diffs, marker='s', linestyle='-', color="#E15759", 
                             markersize=DEFAULT_MARKER_SIZE, label="Δa (RMS)")
                axes[0].set_xlabel("Time (s)", fontsize=10)
                axes[0].set_ylabel("Action Difference:= Δa (RMS)", fontsize=10)
                axes[0].set_title("Action Difference (Oscillation)", fontsize=12)
                axes[0].legend(fontsize=8)
                axes[0].grid(True)

                torque_arr = np.array(self.history['torque'])
                if torque_arr.ndim == 1:
                    axes[1].plot(times, torque_arr, marker='o', linestyle='-', markersize=DEFAULT_MARKER_SIZE,
                                 label="Torque")
                else:
                    n_torques = torque_arr.shape[1]
                    for i in range(n_torques):
                        axes[1].plot(times, torque_arr[:, i], marker='o', linestyle='-', markersize=DEFAULT_MARKER_SIZE,
                                     label=f"Torque {i}")
                axes[1].set_xlabel("Time (s)", fontsize=10)
                axes[1].set_ylabel("Torque", fontsize=10)
                axes[1].set_title("Torque of Each Joint", fontsize=12)
                axes[1].legend(fontsize=8)
                axes[1].grid(True)

                fig.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

            # ===============================
            # Final Section: Configuration Table
            # ===============================
            filtered_config = {k: v for k, v in self.config.items() if k in ["env", "policy", "random", "hardware", "obs_scales"]}
            table_data = self._build_config_rows(filtered_config)
            MAX_ROWS_PER_PAGE = 50
            total_rows = len(table_data)
            n_pages = math.ceil(total_rows / MAX_ROWS_PER_PAGE)

            for page in range(n_pages):
                start_idx = page * MAX_ROWS_PER_PAGE
                end_idx = start_idx + MAX_ROWS_PER_PAGE
                page_data = table_data[start_idx:end_idx]

                fig_config = plt.figure(figsize=PAGE_SIZE)
                ax = fig_config.add_subplot(111)
                ax.axis('tight')
                ax.axis('off')
                fig_config.suptitle("Configuration", fontsize=20, fontweight='bold', y=0.98)

                table = ax.table(cellText=page_data,
                                 colLabels=["Parameter", "Value"],
                                 loc="center",
                                 cellLoc='left')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.2)

                for (row, col), cell in table.get_celld().items():
                    if row == 0:
                        cell.set_text_props(fontweight='bold', color='white')
                        cell.set_facecolor("#40466e")
                    else:
                        cell.set_facecolor("#f1f1f2")

                        if col == 0:
                            text_str = cell.get_text().get_text().strip()
                            #  Bold key group headers (e.g., env, policy) for readability
                            if text_str in ["env", "policy", "random", "hardware", "obs_scales"]:
                                cell.set_text_props(fontweight='bold')

                pdf.savefig(fig_config)
                plt.close(fig_config)

        print(f"Report successfully saved to {self.report_path}")