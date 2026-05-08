# logger.py

import logging
import io
import numpy as np
from contextlib import redirect_stdout
from uniplot import plot as uplot

DEFAULT_WIDTH_SINGLE = 80
DEFAULT_HEIGHT_SINGLE = 25
DEFAULT_WIDTH_SIDE = 50
DEFAULT_HEIGHT_SIDE = 15
# 50 (width) + 10 (labels/padding) = 60
LEFT_COLUMN_TOTAL_WIDTH = DEFAULT_WIDTH_SIDE + 10
RIGHT_COLUMN_TOTAL_WIDTH = DEFAULT_WIDTH_SIDE + 6
SIDE_BY_SIDE_SEPARATOR = " " * 4
SIDE_BY_SIDE_TOTAL_WIDTH = (
    LEFT_COLUMN_TOTAL_WIDTH
    + len(SIDE_BY_SIDE_SEPARATOR)
    + RIGHT_COLUMN_TOTAL_WIDTH
)


class StageFormatter(logging.Formatter):
    def format(self, record):
        # Provide a default stage so formatting never fails on plain log calls.
        if not hasattr(record, 'stage'):
            record.stage = "GENERAL"
        return super().format(record)

# Logger setup
logger = logging.getLogger("AppCore")
handler = logging.StreamHandler()
level = None

# Use the structured `[LEVEL][STAGE] message` scheme everywhere.
formatter = StageFormatter('[%(levelname)s][%(stage)s] %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)

def get_level():
    """
    Read the current log level from `config.json`.

    Returns:
        The `logging` level implied by the `debug_mode` flag.
    """
    global level
    with open("config.json") as f:
        import json
        config = json.load(f)
    if config.get("debug_mode", False):
        level = logging.DEBUG
    else:
        level = logging.INFO
    return level

logger.setLevel(level or get_level())

def _log(msg, stage, level):
    logger.log(level, msg, extra={'stage': stage})

# Public logging helpers
def log_info(msg, stage="GENERAL"):
    _log(msg, stage, logging.INFO)

def log_warn(msg, stage="GENERAL"):
    _log(msg, stage, logging.WARNING)

def log_error(msg, stage="GENERAL"):
    _log(msg, stage, logging.ERROR)

def log_debug(msg, stage="GENERAL"):
    _log(msg, stage, logging.DEBUG)

def setup_file_logging(log_path):
    """
    Attach a file handler that mirrors console logging format and level.

    Args:
        log_path: Destination log file path for the current experiment run.
    """
    # Add a dedicated file handler for the experiment log.
    fh = logging.FileHandler(log_path)
    fh.setLevel(level or get_level())
    
    # Reuse the same stage-aware formatter as the console logger.
    formatter = StageFormatter('[%(levelname)s][%(stage)s] %(message)s')
    fh.setFormatter(formatter)
    
    # Attach the handler to the shared project logger.
    logger.addHandler(fh)


def get_plot_line_width() -> int:
    return SIDE_BY_SIDE_TOTAL_WIDTH


def get_divider(char: str = "=", width: int | None = None) -> str:
    width = width or get_plot_line_width()
    return char * max(1, width)


def get_stage_banner(title: str, width: int | None = None, fill_char: str = "=") -> str:
    """
    Build a centered stage banner aligned with the ASCII dashboard width.

    Args:
        title: Human-readable stage title such as `DATASET` or `TRAINING`.
        width: Total banner width. Defaults to the dashboard width.
        fill_char: Character used to pad both sides of the banner.

    Returns:
        A centered banner string.
    """
    width = width or get_plot_line_width()
    label = f" *** STAGE: {title} *** "
    if len(label) >= width:
        return label
    remaining = width - len(label)
    left = remaining // 2
    right = remaining - left
    return f"{fill_char * left}{label}{fill_char * right}"

def get_ascii_plot(data, title,
                   width=DEFAULT_WIDTH_SINGLE,
                   height=DEFAULT_HEIGHT_SINGLE,
                   lines=False,
                   force_diagonal=False,
                   stage="SUMMARY"):
    """
    Render a single ASCII plot through `uniplot`.

    This helper is used for the console dashboard and supports both standard
    line plots and the small scatter / histogram special cases used in the
    evaluator summary.
    """
    if data is None: return ["No data"]
    
    buf = io.StringIO()
    # Arguments passed through to `uniplot.plot`.
    uplot_args = {
        "title": title,
        "width": width,
        "height": height,
        "color": False,
        "lines": lines
    }

    # Special handling for scatter inputs such as Actual vs Predicted.
    if isinstance(data, list) and len(data) == 2 and isinstance(data[0], (list, np.ndarray)):
        xs_raw, ys_raw = np.array(data[0]), np.array(data[1])
        if xs_raw.size == 0 or ys_raw.size == 0:
            return [f"{title}: no data"]

        finite_mask = np.isfinite(xs_raw) & np.isfinite(ys_raw)
        xs_raw = xs_raw[finite_mask]
        ys_raw = ys_raw[finite_mask]
        if xs_raw.size == 0 or ys_raw.size == 0:
            return [f"{title}: no finite data"]
        
        if force_diagonal:
            low = min(xs_raw.min(), ys_raw.min())
            high = max(xs_raw.max(), ys_raw.max())
            
            # Force a square scatter domain when the diagonal should be visible.
            uplot_args.update({
                "x_min": low, "x_max": high,
                "y_min": low, "y_max": high,
                "xs": [xs_raw, np.array([low, high])],
                "ys": [ys_raw, np.array([low, high])],
                "lines": [False, True]  # data points + diagonal line
            })
        else:
            uplot_args.update({"xs": xs_raw, "ys": ys_raw})
            
    # Standard handling for line-series and histogram-like inputs.
    else:
        # Histogram inputs are passed as `[bin_centers, counts]`.
        if isinstance(data, list) and len(data) == 2 and isinstance(data[0], (list, np.ndarray)):
             xs = np.array(data[0])
             ys = np.array(data[1])
             if xs.size == 0 or ys.size == 0:
                 return [f"{title}: no data"]
             finite_mask = np.isfinite(xs) & np.isfinite(ys)
             xs = xs[finite_mask]
             ys = ys[finite_mask]
             if xs.size == 0 or ys.size == 0:
                 return [f"{title}: no finite data"]
             uplot_args.update({"xs": xs, "ys": ys})
        else:
             ys = np.array(data)
             if ys.size == 0:
                 return [f"{title}: no data"]
             ys = ys[np.isfinite(ys)]
             if ys.size == 0:
                 return [f"{title}: no finite data"]
             uplot_args.update({"ys": ys})

    try:
        with redirect_stdout(buf):
            uplot(**uplot_args)
        return buf.getvalue().splitlines()
    except Exception as e:
        log_error(f"{title} Plot Error: {e}", stage=stage)
        return [f"{title} Plot Error: {e}"]


def get_overlay_ascii_plot(
    series_list,
    title,
    width=DEFAULT_WIDTH_SINGLE,
    height=DEFAULT_HEIGHT_SINGLE,
):
    """
    Render a simple multi-series ASCII plot without relying on uniplot's
    multi-series support.

    Each series item is a tuple of (label, values, char).
    """
    valid_series = []
    for label, values, char in series_list:
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        valid_series.append((label, arr, char))

    if not valid_series:
        return [f"{title}: no finite data"]

    plot_width = max(10, width - 2)
    plot_height = max(5, height - 2)

    global_min = min(float(arr.min()) for _, arr, _ in valid_series)
    global_max = max(float(arr.max()) for _, arr, _ in valid_series)
    if global_min == global_max:
        global_min -= 1.0
        global_max += 1.0

    canvas = [[" " for _ in range(plot_width)] for _ in range(plot_height)]
    tick_count = min(5, plot_height)
    tick_rows = {}
    for idx in range(tick_count):
        if tick_count == 1:
            row = 0
            value = global_max
        else:
            row = int(round(idx * (plot_height - 1) / (tick_count - 1)))
            value = global_max - (global_max - global_min) * (row / max(1, plot_height - 1))
        tick_rows[row] = value

    for _, arr, char in valid_series:
        if arr.size == 1:
            x_positions = [0]
        else:
            x_positions = np.linspace(0, plot_width - 1, num=arr.size)

        for x_f, value in zip(x_positions, arr):
            x = int(round(float(x_f)))
            norm = (float(value) - global_min) / (global_max - global_min)
            y = plot_height - 1 - int(round(norm * (plot_height - 1)))
            current = canvas[y][x]
            canvas[y][x] = char if current == " " else "▘"

    lines = [title]
    legend = "  ".join([f"{char}={label}" for label, _, char in valid_series])
    lines.append(legend)
    lines.append("+" + "-" * plot_width + "+")
    for row_idx, row in enumerate(canvas):
        label = tick_rows.get(row_idx)
        suffix = f" {label:.4f}" if label is not None else ""
        lines.append("|" + "".join(row) + "|" + suffix)
    lines.append("+" + "-" * plot_width + "+")

    max_points = max(arr.size for _, arr, _ in valid_series)
    if max_points > 1:
        lines.append(f"1{' ' * max(1, plot_width - len(str(max_points)) - 1)}{max_points}")
    else:
        lines.append("1")

    return lines
def get_residuals_hist_data(y_true, y_pred, bins=20):
    """
    Convert residuals into histogram coordinates for ASCII plotting.

    Args:
        y_true: Reference values.
        y_pred: Predicted values.
        bins: Number of histogram bins.

    Returns:
        Tuple of `(bin_centers, counts)`.
    """
    if y_true is None or y_pred is None:
        return np.array([]), np.array([])
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.size == 0 or y_pred.size == 0:
        return np.array([]), np.array([])
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite_mask]
    y_pred = y_pred[finite_mask]
    if y_true.size == 0 or y_pred.size == 0:
        return np.array([]), np.array([])
    errors = y_pred - y_true
    counts, bin_edges = np.histogram(errors, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, counts

def log_side_by_side(data_left, title_left, data_right, title_right, 
                     is_scatter=False, is_hist=False, stage="SUMMARY"):
    """
    Render two ASCII plots side by side using a shared text layout.

    Args:
        data_left: Left-panel payload.
        title_left: Left-panel title.
        data_right: Right-panel payload.
        title_right: Right-panel title.
        is_scatter: Whether the left panel is a scatter plot.
        is_hist: Whether the right panel is a histogram plot.
        stage: Logging stage for downstream plot errors.

    Returns:
        Combined multi-line string with both panels.
    """
    # Force the diagonal for left-side scatter panels.
    if (
        isinstance(data_left, list)
        and data_left
        and isinstance(data_left[0], tuple)
        and len(data_left[0]) == 3
    ):
        left_lines = get_overlay_ascii_plot(
            data_left,
            title_left,
            width=DEFAULT_WIDTH_SIDE,
            height=DEFAULT_HEIGHT_SIDE,
        )
    else:
        left_lines = get_ascii_plot(data_left, title_left,
                                    width=DEFAULT_WIDTH_SIDE, height=DEFAULT_HEIGHT_SIDE,
                                    force_diagonal=is_scatter, stage=stage)
    
    if (
        isinstance(data_right, list)
        and data_right
        and isinstance(data_right[0], tuple)
        and len(data_right[0]) == 3
    ):
        right_lines = get_overlay_ascii_plot(
            data_right,
            title_right,
            width=DEFAULT_WIDTH_SIDE,
            height=DEFAULT_HEIGHT_SIDE,
        )
    else:
        # Histograms are drawn as line plots in the right-hand panel.
        right_lines = get_ascii_plot(data_right, title_right,
                                     width=DEFAULT_WIDTH_SIDE, height=DEFAULT_HEIGHT_SIDE,
                                     lines=is_hist, stage=stage)

    max_len = max(len(left_lines), len(right_lines))
    left_lines += [""] * (max_len - len(left_lines))
    right_lines += [""] * (max_len - len(right_lines))

    max_w_left = max(len(line) for line in left_lines) if left_lines else DEFAULT_WIDTH_SIDE

    combined = [""]
    separator = SIDE_BY_SIDE_SEPARATOR
    for left, right in zip(left_lines, right_lines):
        left_padded = left.ljust(LEFT_COLUMN_TOTAL_WIDTH)
        combined.append(f"{left_padded}{separator}{right}")
    
    return "\n".join(combined)

def console_plots(trainer_history, side_by_side=True, stage="SUMMARY"):
    """
    Emit the ASCII training dashboard to the experiment log.

    Args:
        trainer_history: History dictionary produced by `Trainer.train`.
        side_by_side: Whether to render the compact two-column layout.
        stage: Logging stage under which the dashboard should be written.
    """
    if side_by_side:
        log_info("Generating Multi-column ASCII Dashboard.", stage=stage)

        dashboard_loss_rmse = log_side_by_side(
            [
                ("Train Loss", trainer_history["train_loss"], "▘"),
                ("Val Loss", trainer_history.get("val_loss", []), "*"),
            ], "Learning Curve (Loss)",
            [
                ("Train RMSE", trainer_history.get("train_rmse", []), "▘"),
                ("Val RMSE", trainer_history["val_rmse"], "*"),
            ], "Validation RMSE"
        )
        log_info(dashboard_loss_rmse, stage=stage)

        dashboard_r_ci = log_side_by_side(
            [
                ("Train Pearson", trainer_history.get("train_pearson", []), "▘"),
                ("Val Pearson", trainer_history["val_pearson"], "*"),
            ], "Correlation (Pearson R)",
            [
                ("Train CI", trainer_history.get("train_ci", []), "▘"),
                ("Val CI", trainer_history["val_ci"], "*"),
            ], "Ranking Accuracy (CI)"
        )
        log_info(dashboard_r_ci, stage=stage)

        y_true = trainer_history.get('best_y_true')
        y_pred = trainer_history.get('best_y_pred')
        if y_true is None or y_pred is None:
            log_info("Final Performance Analytics: best predictions are not available yet.", stage=stage)
        else:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            hist_x, hist_y = get_residuals_hist_data(y_true, y_pred)

            dashboard_final = log_side_by_side(
                [y_true, y_pred], "Actual vs Predicted",
                [hist_x, hist_y], "Residuals Distribution",
                is_scatter=True,
                is_hist=True,
                stage=stage
            )
            
            log_info("Final Performance Analytics:" + dashboard_final, stage=stage)

    else:
        loss_chart = get_overlay_ascii_plot(
            [
                ("Train Loss", trainer_history["train_loss"], "▘"),
                ("Val Loss", trainer_history.get("val_loss", []), "*"),
            ],
            title="Learning Curve (Loss)"
        )
        log_info(f"Loss Curve:\n" + "\n".join(loss_chart), stage=stage)

        rmse_chart = get_overlay_ascii_plot(
            [
                ("Train RMSE", trainer_history.get("train_rmse", []), "▘"),
                ("Val RMSE", trainer_history["val_rmse"], "*"),
            ],
            title="Validation RMSE"
        )
        log_info(f"RMSE Curve:\n" + "\n".join(rmse_chart), stage=stage)

        r_chart = get_overlay_ascii_plot(
            [
                ("Train Pearson", trainer_history.get("train_pearson", []), "▘"),
                ("Val Pearson", trainer_history["val_pearson"], "*"),
            ],
            title="Correlation (Pearson R)"
        )
        log_info(f"Pearson (R) Curve:\n" + "\n".join(r_chart), stage=stage)

        ci_chart = get_overlay_ascii_plot(
            [
                ("Train CI", trainer_history.get("train_ci", []), "▘"),
                ("Val CI", trainer_history["val_ci"], "*"),
            ],
            title="Ranking Accuracy (CI)"
        )
        log_info(f"Concordancy Index Curve:\n" + "\n".join(ci_chart), stage=stage)

        y_true = trainer_history.get('best_y_true')
        y_pred = trainer_history.get('best_y_pred')
        if y_true is None or y_pred is None:
            log_info("Best predictions are not available yet; skipping final scatter/hist plots.", stage=stage)
        else:
            act_vs_pred_chart = get_ascii_plot([y_true, y_pred], title="Predicted = f(Actual)", force_diagonal=True)
            log_info(f"Actual vs Predicted Scatter Plot:\n" + "\n".join(act_vs_pred_chart), stage=stage)

            hist_x, hist_y = get_residuals_hist_data(y_true, y_pred)

            resid_chart = get_ascii_plot([hist_x, hist_y], title="Residuals Distribution", lines=True)
            log_info(f"Errors Distribution (Residuals):\n" + "\n".join(resid_chart), stage=stage)
