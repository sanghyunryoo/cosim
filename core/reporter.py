import math
import time
import textwrap

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PDF export
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import to_rgb
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import matplotlib.patheffects as pe
import numpy as np
from cycler import cycler


# -------------------------------
# Modern report theme (design only)
# -------------------------------
THEME_INK = "#0B1220"        # near-slate-950
THEME_INK_SOFT = "#111827"   # gray-900
THEME_MUTED = "#64748B"      # slate-500
THEME_MUTED_2 = "#94A3B8"    # slate-400
THEME_BORDER = "#E2E8F0"     # slate-200
THEME_GRID = "#E5E7EB"       # gray-200
THEME_SURFACE = "#FFFFFF"
THEME_SURFACE_ALT = "#F8FAFC"  # slate-50
THEME_ACCENT = "#6366F1"     # indigo-500
THEME_ACCENT_2 = "#06B6D4"   # cyan-500
THEME_DANGER = "#EF4444"     # red-500
THEME_SUCCESS = "#22C55E"    # green-500

palette = [
    "#6366F1",  # Indigo
    "#06B6D4",  # Cyan
    "#22C55E",  # Green
    "#F97316",  # Orange
    "#EC4899",  # Pink
    "#A855F7",  # Violet
    "#0EA5E9",  # Sky
    "#111827",  # Gray-900
]

# Global Matplotlib style (clean, modern, PDF-friendly)
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans'],

    # Figure / Axes
    'figure.facecolor': THEME_SURFACE,
    'savefig.facecolor': THEME_SURFACE,
    'axes.facecolor': THEME_SURFACE,
    'axes.edgecolor': THEME_BORDER,
    'axes.linewidth': 0.9,
    'axes.titlecolor': THEME_INK,
    'axes.labelcolor': "#334155",   # slate-700
    'axes.titleweight': 'semibold',
    'axes.titlesize': 15,
    'axes.labelsize': 11,
    'axes.titlepad': 8,
    'axes.axisbelow': True,

    # Spines
    'axes.spines.top': False,
    'axes.spines.right': False,

    # Ticks
    'xtick.color': "#475569",        # slate-600
    'ytick.color': "#475569",
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    # Lines
    'lines.linewidth': 1.7,
    'lines.solid_capstyle': 'round',
    'lines.solid_joinstyle': 'round',

    # Grid
    'grid.color': THEME_GRID,
    'grid.linestyle': '-',
    'grid.linewidth': 0.6,
    'grid.alpha': 1.0,

    # Legend
    'legend.fontsize': 9,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': THEME_BORDER,
    'legend.borderpad': 0.6,
    'legend.labelspacing': 0.4,
    'legend.handlelength': 2.2,

    # Subplot margins (keep overall logic; just slightly refined spacing)
    'figure.subplot.left': 0.08,
    'figure.subplot.right': 0.92,
    'figure.subplot.bottom': 0.06,
    'figure.subplot.top': 0.94,

    # PDF font embedding
    'pdf.fonttype': 42,
    'ps.fonttype': 42,

    # Color palette
    'axes.prop_cycle': cycler('color', palette),
})

# Default marker size for time-series plots
DEFAULT_MARKER_SIZE = 1


def add_section_header(fig, title, *, banner_height=0.06, page_num=None):
    """
    Draw a slim banner on top of the figure with a white, bold title.
    (Design updated only)
    """
    # Soft shadow
    fig.add_artist(FancyBboxPatch(
        (0.04, 1.0 - banner_height - 0.006), 0.92, banner_height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=fig.transFigure,
        linewidth=0,
        facecolor="black",
        alpha=0.10,
        zorder=999
    ))

    # Main banner
    fig.add_artist(FancyBboxPatch(
        (0.04, 1.0 - banner_height), 0.92, banner_height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=fig.transFigure,
        linewidth=1.0,
        edgecolor="#111827",
        facecolor=THEME_INK,
        zorder=1000
    ))

    # Gradient accent line
    accent_x, accent_y = 0.04, 1.0 - banner_height
    accent_w, accent_h = 0.92, 0.003

    c0 = np.array(to_rgb("#06B6D4"))  # cyan-500
    c1 = np.array(to_rgb("#A855F7"))  # violet-500

    # (optional) subtle glow under the line
    fig.add_artist(Rectangle(
        (accent_x, accent_y - accent_h * 1.0), accent_w, accent_h * 2.8,
        transform=fig.transFigure,
        facecolor=(c0 * 0.5 + c1 * 0.5),
        edgecolor="none",
        alpha=0.10,
        zorder=1000.8
    ))

    # gradient image data (small but smooth)
    grad_w, grad_h = 512, 10
    row = np.linspace(c0, c1, grad_w)[None, :, :]   # 1 x W x 3
    grad = np.repeat(row, grad_h, axis=0)           # H x W x 3

    # place the image in figure coordinates
    bbox = TransformedBbox(
        Bbox.from_bounds(accent_x, accent_y, accent_w, accent_h),
        fig.transFigure
    )
    bi = BboxImage(bbox, interpolation="bicubic", origin="lower", zorder=1001)
    bi.set_data(grad)
    bi.set_clip_on(False)
    fig.add_artist(bi)

    fig.text(
        0.065, 1.0 - banner_height / 2.0, title,
        ha="left", va="center",
        fontsize=14.5, fontweight="bold", color="white",
        path_effects=[pe.withStroke(linewidth=2, foreground="black", alpha=0.18)],
        zorder=1002
    )

    if page_num is not None:
        chip_w = 0.042                  
        chip_x = 0.04 + 0.92 - chip_w - 0.020 

        fig.text(
            chip_x + chip_w/2.0, 1.0 - banner_height / 2.0, str(page_num),
            ha="center", va="center",
            fontsize=7.6,  # was 8.5
            color="white",
            path_effects=[pe.withStroke(linewidth=1, foreground="black", alpha=0.20)],
            zorder=1004
        )


class Reporter:
    def __init__(self, report_path, config):
        """
        Initialize a report generator.
        Args:
            report_path: Output PDF path.
            config: Dict of configuration for rendering the 'Configuration' table.
        """
        self.report_path = report_path
        self.config = config
        self.history = {}
        self.timesteps = 0

    def write_info(self, info):
        """
        Append one timestep of logged info.
        """
        self.timesteps += 1
        for key, value in info.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def _build_config_rows(self, config, indent=0):
        """
        Flatten nested dict into 2-column rows: [Parameter, Value].
        Indentation is represented by 4 spaces per nesting level.
        """
        rows = []
        indent_str = "    " * indent
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

    def _wrap_long_text(self, s: str, *, width=38, max_lines=8):
        """
        Wrap a long string to a fixed width. If it exceeds max_lines,
        trim and add a tail marker like: '… (+N more)'.
        """
        if not isinstance(s, str):
            s = str(s)
        wrapped_lines = textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False)
        if len(wrapped_lines) <= max_lines:
            return "\n".join(wrapped_lines), len(wrapped_lines)
        hidden = len(wrapped_lines) - max_lines
        trimmed = wrapped_lines[:max_lines] + [f"… (+{hidden} more)"]
        return "\n".join(trimmed), len(trimmed)

    def generate_report(self):
        """
        Build the multi-page PDF report.
        """
        PAGE_SIZE = (8.27, 11.69)  # A4 portrait
        dt = float(self.history.get('dt', [1])[0])
        times = np.arange(self.timesteps) * dt

        # Safe defaults for page counts
        PAGE_NUM_IN_SECTION_1 = 0
        PAGE_NUM_IN_SECTION_2 = 0
        PAGE_NUM_IN_SECTION_3 = 0

        with PdfPages(self.report_path) as pdf:
            # -------------------------------
            # Cover page
            # -------------------------------
            fig_cover = plt.figure(figsize=PAGE_SIZE)
            bg_ax = fig_cover.add_axes([0, 0, 1, 1], zorder=0)
            bg_ax.axis("off")

            # Soft vertical gradient background (refined)
            h, w = 900, 1400
            top_rgb = np.array([255, 255, 255]) / 255.0        # white
            bottom_rgb = np.array([244, 244, 245]) / 255.0     # zinc-100 (#F4F4F5)

            t = np.linspace(0.0, 1.0, h)[:, None]
            grad = (top_rgb * (1.0 - t) + bottom_rgb * t)
            grad = np.repeat(grad[None, ...], w, axis=0).transpose(1, 0, 2)
            yy, xx = np.mgrid[0:h, 0:w]
            x = (xx / (w - 1)) - 0.5
            y = (yy / (h - 1)) - 0.45
            r = np.sqrt(x * x + y * y)
            v = np.clip((r - 0.05) / 0.95, 0.0, 1.0)  # 0~1

            grad = np.clip(grad * (1.0 - 0.02 * v[..., None]), 0.0, 1.0)

            bg_ax.imshow(grad, aspect="auto", extent=[0, 1, 0, 1], origin="lower")

            # Header ribbon (rounded, with accent)
            fig_cover.add_artist(FancyBboxPatch(
                (0.06, 0.85), 0.88, 0.125,
                boxstyle="round,pad=0.012,rounding_size=0.03",
                linewidth=0,
                facecolor=THEME_INK,
                zorder=1,
                transform=fig_cover.transFigure
            ))

            fig_cover.text(
                0.10, 0.915, "Test Report",
                ha="left", va="center",
                fontsize=34, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground="black", alpha=0.15)]
            )
            fig_cover.text(
                0.10, 0.875, "Performance Summary",
                ha="left", va="center",
                fontsize=13, color="#CBD5E1"
            )

            generated_on = time.strftime("%Y-%m-%d %H:%M:%S")
            duration_sec = int(self.timesteps * dt)
            mins, secs = divmod(duration_sec, 60)
            hours, mins = divmod(mins, 60)
            duration_str = f"{hours:02d}:{mins:02d}:{secs:02d}"
            env_id = (self.config.get("env", {}) or {}).get("id", "N/A")

            meta_items = [
                ("Env ID", str(env_id)),
                ("Duration", duration_str),
                ("Generated", generated_on),
            ]

            chip_y = 0.66
            chip_w = 0.22
            chip_h = 0.085
            chip_gap = 0.045

            total_w = 3 * chip_w + 2 * chip_gap
            x_start = 0.5 - total_w / 2.0
            chip_xs = [x_start, x_start + chip_w + chip_gap, x_start + 2 * (chip_w + chip_gap)]

            for (label, value), cx in zip(meta_items, chip_xs):
                # shadow
                fig_cover.add_artist(FancyBboxPatch(
                    (cx, chip_y - 0.006), chip_w, chip_h,
                    boxstyle="round,pad=0.013,rounding_size=0.025",
                    linewidth=0, facecolor="black", alpha=0.08,
                    zorder=1.5, transform=fig_cover.transFigure
                ))
                # card
                chip = FancyBboxPatch(
                    (cx, chip_y), chip_w, chip_h,
                    boxstyle="round,pad=0.013,rounding_size=0.025",
                    linewidth=1.0, edgecolor=THEME_BORDER, facecolor="white",
                    zorder=2, transform=fig_cover.transFigure
                )
                fig_cover.add_artist(chip)

                fig_cover.text(cx + 0.016, chip_y + chip_h*0.66, label,
                               ha="left", va="center", fontsize=9.5, color=THEME_MUTED)
                fig_cover.text(cx + 0.016, chip_y + chip_h*0.30, value,
                               ha="left", va="center", fontsize=12.5, fontweight="bold", color=THEME_INK)

            # Divider
            fig_cover.add_artist(Rectangle((0.10, 0.605), 0.80, 0.002, color=THEME_BORDER, zorder=1))

            description = "COCELO Sim-to-Sim Framework v1.5"
            desc_y = 0.56
            fig_cover.text(0.5, desc_y, description,
                           ha="center", va="top",
                           fontsize=16, fontweight="bold", color=THEME_INK)

            # Footer
            fig_cover.add_artist(Rectangle((0.10, 0.08), 0.80, 0.0015, color=THEME_BORDER, zorder=1))
            current_year = time.strftime("%Y")
            fig_cover.text(
                0.10, 0.05, f"{current_year} COCELO Inc. - Automatically Generated",
                ha="left", va="center", fontsize=9, color="#6B7280"
            )

            pdf.savefig(fig_cover)
            plt.close(fig_cover)

            # -------------------------------
            # 1) Set Points vs. States
            # -------------------------------
            if ('set_points' in self.history and 'state' in self.history):
                set_points = np.array(self.history['set_points'], dtype=float)
                state = np.array(self.history['state'], dtype=float)
                if set_points.ndim == 1:
                    set_points = set_points.reshape(-1, 1)
                    state = state.reshape(-1, 1)

                n_dims = set_points.shape[1]
                MAX_PLOTS_PER_PAGE_IN_SECTION_1 = 8
                N_COLS = 2

                START_PAGE_NO = 1
                PAGE_NUM_IN_SECTION_1 = len(list(range(0, n_dims, MAX_PLOTS_PER_PAGE_IN_SECTION_1)))
                for i, start in enumerate(range(0, n_dims, MAX_PLOTS_PER_PAGE_IN_SECTION_1)):
                    end = min(start + MAX_PLOTS_PER_PAGE_IN_SECTION_1, n_dims)
                    dims_this_page = end - start
                    n_rows = math.ceil(dims_this_page / N_COLS)

                    fig, axes = plt.subplots(n_rows, N_COLS, figsize=PAGE_SIZE)
                    add_section_header(fig, "Set Points vs. States", page_num=START_PAGE_NO + i)

                    if isinstance(axes, np.ndarray):
                        axes = axes.flatten()
                    else:
                        axes = np.array([axes])

                    for local_idx, dim_idx in enumerate(range(start, end)):
                        ax = axes[local_idx]
                        ax.plot(times, set_points[:, dim_idx], linestyle='-',
                                markersize=DEFAULT_MARKER_SIZE, label="Set Point")
                        ax.plot(times, state[:, dim_idx], linestyle='-.',
                                markersize=DEFAULT_MARKER_SIZE, label="State")
                        ax.set_xlabel("Time (s)", fontsize=10)
                        ax.set_ylabel(f"Set Point", fontsize=10)
                        ax.set_title(f"Dimension {dim_idx}", fontsize=12)
                        ax.legend(fontsize=8)
                        ax.grid(True)

                    for j in range(dims_this_page, len(axes)):
                        fig.delaxes(axes[j])

                    fig.tight_layout(rect=[0, 0, 1, 0.92])
                    pdf.savefig(fig)
                    plt.close(fig)

            # -------------------------------
            # 2) Command Inputs vs. Measured Outputs
            # -------------------------------
            # Find all user_command_i keys dynamically
            user_command_keys = sorted([k for k in self.history.keys() if k.startswith('user_command_')])
            n_commands = len(user_command_keys)
            measured_outputs = [
                ("lin_vel_x", "Linear Velocity X", "m/s"),
                ("lin_vel_y", "Linear Velocity Y", "m/s"),
                ("ang_vel_yaw", "Angular Velocity Yaw", "rad/s"),
            ]

            # Filter measured outputs that exist in history
            plot_data = [
                (measured_key, label, unit)
                for (measured_key, label, unit) in measured_outputs
                if measured_key in self.history
            ]

            # Define distinct line styles for commands
            COMMAND_LINESTYLES = ["-", "--", "-.", ":"]

            MAX_PLOTS_PER_PAGE_IN_SECTION_2 = 3

            START_PAGE_NO = PAGE_NUM_IN_SECTION_1 + 1
            PAGE_NUM_IN_SECTION_2 = len(list(range(0, len(plot_data), MAX_PLOTS_PER_PAGE_IN_SECTION_2)))

            if plot_data and n_commands > 0:
                for i, start in enumerate(range(0, len(plot_data), MAX_PLOTS_PER_PAGE_IN_SECTION_2)):
                    page_data = plot_data[start:start + MAX_PLOTS_PER_PAGE_IN_SECTION_2]
                    n_rows = len(page_data)

                    fig, axes = plt.subplots(n_rows, 1, figsize=PAGE_SIZE)
                    add_section_header(fig, "Command Inputs vs. Measured Outputs", page_num=START_PAGE_NO + i)

                    if not isinstance(axes, (list, np.ndarray)):
                        axes = [axes]

                    for idx, (measured_key, label, unit) in enumerate(page_data):
                        measured_values = np.array(self.history[measured_key], dtype=float)
                        ax = axes[idx]

                        # Plot all user_command_i on this graph with distinct colors and styles
                        for cmd_idx, cmd_key in enumerate(user_command_keys):
                            cmd_values = np.array(self.history[cmd_key], dtype=float)
                            cmd_index = cmd_key.replace('user_command_', '')

                            # Use palette colors and cycle through line styles
                            color = palette[cmd_idx % len(palette)]
                            linestyle = COMMAND_LINESTYLES[cmd_idx % len(COMMAND_LINESTYLES)]

                            ax.plot(times, cmd_values,
                                    linestyle=linestyle,
                                    color=color,
                                    linewidth=1.5,
                                    markersize=DEFAULT_MARKER_SIZE,
                                    label=f"Command {cmd_index}")

                        # Plot the measured output with thick black line
                        ax.plot(times, measured_values,
                                linestyle='-',
                                linewidth=2.5,
                                color='#000000',
                                markersize=DEFAULT_MARKER_SIZE,
                                label=f"{label} ({unit})")

                        ax.set_xlabel("Time (s)", fontsize=10)
                        ax.set_ylabel("value", fontsize=10)
                        ax.set_title(f"{label} ({unit})", fontsize=12)
                        ax.legend(fontsize=8, loc='best')
                        ax.grid(True)

                    fig.tight_layout(rect=[0, 0, 1, 0.92])
                    pdf.savefig(fig)
                    plt.close(fig)

            # -------------------------------
            # 3) Action Oscillation and Applied Torques
            # -------------------------------
            START_PAGE_NO = PAGE_NUM_IN_SECTION_1 + PAGE_NUM_IN_SECTION_2 + 1
            PAGE_NUM_IN_SECTION_3 = 0
            if ('torque' in self.history and 'action_diff_RMSE' in self.history):
                diffs = np.array(self.history['action_diff_RMSE'], dtype=float)
                torque_arr = np.array(self.history['torque'], dtype=float)
                if torque_arr.ndim == 1:
                    torque_arr = torque_arr.reshape(-1, 1)

                fig, axes = plt.subplots(3, 1, figsize=PAGE_SIZE)
                add_section_header(fig, "Action Oscillation and Applied Torques", page_num=START_PAGE_NO)
                PAGE_NUM_IN_SECTION_3 = 1

                # (a) Δa (RMSE) time-series
                axes[0].plot(times, diffs, linestyle='-', label="Δa (RMSE)")

                # --- Moving Average (same graph) ---
                # window from config if provided; default = 20
                ma_win = max(1, min(20, self.timesteps//2))

                # compute simple moving average and align to times
                kernel = np.ones(ma_win, dtype=float) / float(ma_win)
                ma_vals = np.convolve(diffs, kernel, mode='valid')
                # pad the front with NaNs so length matches times
                ma_aligned = np.concatenate([np.full(ma_win - 1, np.nan), ma_vals])

                axes[0].plot(times, ma_aligned, linestyle='--', label=f"Δa (RMSE) Moving Average (window={ma_win})")
                # -----------------------------------

                axes[0].set_xlabel("Time (s)", fontsize=10)
                axes[0].set_ylabel("Δa (RMSE)", fontsize=10)
                axes[0].set_title("Action Oscillation", fontsize=12)
                axes[0].legend(fontsize=8)
                axes[0].grid(True)

                # (b) Torques per joint
                n_torques = torque_arr.shape[1]
                LINESTYLES = ["-", "--", ":", "-."]

                for i in range(n_torques):
                    color = palette[i % 8]
                    ls = LINESTYLES[(i // 8) % len(LINESTYLES)]
                    axes[1].plot(times, torque_arr[:, i], linestyle=ls, color=color, label=f"Torque {i}")

                axes[1].set_xlabel("Time (s)", fontsize=10)
                axes[1].set_ylabel("Torque", fontsize=10)
                axes[1].set_title("Applied Torque of Each Joint", fontsize=12)

                if n_torques <= 8:
                    axes[1].legend(fontsize=7, ncol=2)
                elif n_torques <= 16:
                    axes[1].legend(fontsize=6, ncol=4)
                else:
                    axes[1].legend(fontsize=6, ncol=6)

                axes[1].grid(True)

                # (c) Torque distribution
                all_torque = torque_arr.ravel()
                p5, p95 = np.percentile(all_torque, [5, 95])
                axes[2].set_xlabel("Torque (N·m)", fontsize=10)
                axes[2].set_ylabel("Count", fontsize=10)
                axes[2].set_title("Torque Distribution of All Joints", fontsize=12)
                axes[2].hist(all_torque, color="#BFDBFE", bins=60, alpha=0.95, edgecolor=THEME_INK_SOFT)
                axes[2].axvline(p5, color=THEME_DANGER, linestyle='--', linewidth=1.2, label=f"5th: {p5:.3g}")
                axes[2].axvline(p95, color=THEME_DANGER, linestyle=':', linewidth=1.2, label=f"95th: {p95:.3g}")

                mean_val = np.mean(all_torque)
                mean_abs_val = np.mean(np.abs(all_torque))

                axes[2].axvline(mean_val, color=THEME_SUCCESS, linestyle='-.', linewidth=1.2, label=f"Average: {mean_val:.3g}")
                axes[2].axvline(mean_abs_val, color=THEME_INK_SOFT, linestyle='-.', linewidth=1.2, label=f"|Average|: {mean_abs_val:.3g}")

                axes[2].legend(fontsize=8)
                axes[2].grid(True)

                fig.tight_layout(rect=[0, 0, 1, 0.92])
                pdf.savefig(fig)
                plt.close(fig)

            # -------------------------------
            # Final) Configuration Table
            # -------------------------------
            filtered_config = {k: v for k, v in self.config.items() if k in ["env", "policy", "observation", "random", "hardware"]}
            table_data = self._build_config_rows(filtered_config)
            MAX_ROWS_PER_PAGE = 50
            total_rows = len(table_data)
            n_pages = math.ceil(total_rows / MAX_ROWS_PER_PAGE) if total_rows > 0 else 1

            START_PAGE_NO = PAGE_NUM_IN_SECTION_1 + PAGE_NUM_IN_SECTION_2 + PAGE_NUM_IN_SECTION_3 + 1
            for i, page in enumerate(range(n_pages)):
                start_idx = page * MAX_ROWS_PER_PAGE
                end_idx = start_idx + MAX_ROWS_PER_PAGE
                page_data = table_data[start_idx:end_idx]

                fig_config = plt.figure(figsize=PAGE_SIZE)
                ax = fig_config.add_subplot(111)
                ax.axis('tight')
                ax.axis('off')
                add_section_header(fig_config, "Configuration", page_num=START_PAGE_NO + i)

                table = ax.table(
                    cellText=page_data,
                    colLabels=["Parameter", "Value"],
                    loc="upper center",
                    cellLoc='left',
                )
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.2)  # Keep original scale; DO NOT change column widths

                # Header/body base styling; detect a base row height
                base_h = None
                for (row, col), cell in table.get_celld().items():
                    # unified borders (design only)
                    cell.set_edgecolor(THEME_BORDER)
                    cell.set_linewidth(0.6)

                    if row == 0:
                        cell.set_text_props(fontweight='bold', color='white')
                        cell.set_facecolor(THEME_INK)
                        try:
                            cell.PAD = 0.03
                        except Exception:
                            pass
                    else:
                        if base_h is None:
                            base_h = cell.get_height()

                        # subtle zebra striping
                        cell.set_facecolor(THEME_SURFACE if (row % 2 == 1) else THEME_SURFACE_ALT)

                        # slightly tighter padding
                        try:
                            cell.PAD = 0.02
                        except Exception:
                            pass

                        if col == 0:
                            left_text = cell.get_text().get_text().strip()
                            if left_text in ["env", "policy", "observation", "random", "hardware"]:
                                cell.set_text_props(fontweight='bold', color=THEME_INK)
                                # section header row highlight (design only)
                                cell.set_facecolor("#EEF2FF")
                                if (row, 1) in table.get_celld():
                                    table.get_celld()[(row, 1)].set_facecolor("#EEF2FF")

                if base_h is None:
                    base_h = 0.05  # Fallback if table implementation changes

                # Make left-column padding and alignment UNIFORM for all body rows
                for (row, col), cell in table.get_celld().items():
                    if row > 0 and col == 0:
                        # Set consistent left padding and alignment so all labels start at same x
                        try:
                            cell.PAD = 0.02
                        except Exception:
                            pass
                        cell.get_text().set_ha("left")
                        cell.get_text().set_va("center")

                # (1) Force the "observation" section header's Value cell to be empty
                obs_header_row = None
                for (row, col), cell in table.get_celld().items():
                    if row == 0:
                        continue
                    if col == 0:
                        left_key = cell.get_text().get_text().strip()
                        if left_key == "observation":
                            obs_header_row = row
                            if (row, 1) in table.get_celld():
                                vcell = table.get_celld()[(row, 1)]
                                vcell.get_text().set_text("")  # Keep blank (never show text)

                # (2) Wrap ONLY the two target keys; center-align; adjust row height; keep others untouched
                target_keys = {"stacked_obs_order", "non_stacked_obs_order"}
                for (row, col), cell in list(table.get_celld().items()):
                    # Only body rows, Value column
                    if row == 0 or col != 1:
                        continue
                    # Never touch the observation header's Value cell
                    if obs_header_row is not None and row == obs_header_row:
                        continue

                    # Read the left key for this row (strip indentation)
                    if row - 1 < len(page_data):
                        left_key = page_data[row - 1][0].lstrip()
                    else:
                        left_key = ""

                    if left_key not in target_keys:
                        continue

                    value_text = cell.get_text().get_text()
                    if not value_text:
                        continue

                    # Wrap text and set it back
                    wrapped, n_lines = self._wrap_long_text(value_text, width=50, max_lines=8)
                    vtxt = cell.get_text()
                    vtxt.set_text(wrapped)
                    vtxt.set_ha("left")
                    vtxt.set_va("center")  # vertical center for consistent look

                    # Row height with generous growth and safety for multi-line
                    growth = 0.70                                # per extra line multiplier
                    safety = 0.012 if n_lines > 1 else 0.0       # absolute safety margin
                    new_h = base_h * (1.0 + growth * (n_lines - 1)) + safety

                    # Apply same height to Parameter cell and keep its text vertically centered
                    if (row, 0) in table.get_celld():
                        lcell = table.get_celld()[(row, 0)]
                        lcell.set_height(new_h)
                        lcell.get_text().set_va("center")
                        lcell.get_text().set_ha("left")  # ensure left-aligned label

                    cell.set_height(new_h)

                fig_config.tight_layout(rect=[0, 0, 1, 0.92])
                pdf.savefig(fig_config)
                plt.close(fig_config)

        print(f"Report successfully saved to {self.report_path}")
