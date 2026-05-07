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


class StageFormatter(logging.Formatter):
    def format(self, record):
        # Если префикс не передан, ставим пустую заглушку, 
        # чтобы форматтер не выкинул ошибку
        if not hasattr(record, 'stage'):
            record.stage = "GENERAL"
        return super().format(record)

# Настройка
logger = logging.getLogger("AppCore")
handler = logging.StreamHandler()
level = None

# Вот здесь задаем твою схему [LEVEL][STAGE]
formatter = StageFormatter('[%(levelname)s][%(stage)s] %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)
# подтягиваем debug_mode из конфига и устанавливаем соответствующий уровень логов
def get_level():
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

# Публичный API твоего логгера
def log_info(msg, stage="GENERAL"):
    _log(msg, stage, logging.INFO)

def log_warn(msg, stage="GENERAL"):
    _log(msg, stage, logging.WARNING)

def log_error(msg, stage="GENERAL"):
    _log(msg, stage, logging.ERROR)

def log_debug(msg, stage="GENERAL"):
    _log(msg, stage, logging.DEBUG)

def setup_file_logging(log_path):
    # Создаем обработчик для файла
    fh = logging.FileHandler(log_path)
    fh.setLevel(level or get_level())
    
    # Применяем тот же форматтер
    formatter = StageFormatter('[%(levelname)s][%(stage)s] %(message)s')
    fh.setFormatter(formatter)
    
    # Добавляем к существующему логгеру
    logger.addHandler(fh)

def get_ascii_plot(data, title,
                   width=DEFAULT_WIDTH_SINGLE,
                   height=DEFAULT_HEIGHT_SINGLE,
                   lines=False,
                   force_diagonal=False,
                   stage="SUMMARY"):
    if data is None: return ["No data"]
    
    buf = io.StringIO()
    # Словарь аргументов для uplot
    uplot_args = {
        "title": title,
        "width": width,
        "height": height,
        "color": False,
        "lines": lines
    }

    # 1. Логика для Scatter Plot (Actual vs Predicted)
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
            
            # Устанавливаем жесткие границы квадрата
            uplot_args.update({
                "x_min": low, "x_max": high,
                "y_min": low, "y_max": high,
                "xs": [xs_raw, np.array([low, high])],
                "ys": [ys_raw, np.array([low, high])],
                "lines": [False, True] # Точки для данных, линия для диагонали
            })
        else:
            uplot_args.update({"xs": xs_raw, "ys": ys_raw})
            
    # 2. Логика для обычных графиков (Loss, RMSE и Histogram)
    else:
        # Если это гистограмма, пришедшая как [bins, counts]
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
            uplot(**uplot_args) # Распаковываем только нужные аргументы
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
    """Готовит данные для гистограммы"""
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
    # Если слева scatter, включаем диагональ
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
    
    # Если справа гистограмма, рисуем ее линиями
    right_lines = get_ascii_plot(data_right, title_right,
                                 width=DEFAULT_WIDTH_SIDE, height=DEFAULT_HEIGHT_SIDE,
                                 lines=is_hist, stage=stage)

    max_len = max(len(left_lines), len(right_lines))
    left_lines += [""] * (max_len - len(left_lines))
    right_lines += [""] * (max_len - len(right_lines))

    max_w_left = max(len(line) for line in left_lines) if left_lines else DEFAULT_WIDTH_SIDE

    combined = [""]
    separator = " " * 4
    for left, right in zip(left_lines, right_lines):
        left_padded = left.ljust(LEFT_COLUMN_TOTAL_WIDTH)
        combined.append(f"{left_padded}{separator}{right}")
        # combined.append(f"{left.ljust(DEFAULT_WIDTH_SIDE + 5)}{separator}{right}")
    
    return "\n".join(combined)

def console_plots(trainer_history, side_by_side=True, stage="SUMMARY"):
    if side_by_side:
        log_info("Generating Multi-column ASCII Dashboard.", stage=stage)

        dashboard_loss_rmse = log_side_by_side(
            [
                ("Train Loss", trainer_history["train_loss"], "▘"),
                ("Val Loss", trainer_history.get("val_loss", []), "*"),
            ], "Learning Curve (Loss)",
            trainer_history["val_rmse"], "Validation RMSE"
        )
        log_info(dashboard_loss_rmse, stage=stage)

        dashboard_r_ci = log_side_by_side(
            trainer_history["val_pearson"], "Correlation (Pearson R)",
            trainer_history["val_ci"], "Ranking Accuracy (CI)"
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

        rmse_chart = get_ascii_plot(trainer_history["val_rmse"], title="Validation RMSE")
        log_info(f"RMSE Curve:\n" + "\n".join(rmse_chart), stage=stage)

        r_chart = get_ascii_plot(trainer_history["val_pearson"], title="Correlation (Pearson R)")
        log_info(f"Pearson (R) Curve:\n" + "\n".join(r_chart), stage=stage)

        ci_chart = get_ascii_plot(trainer_history["val_ci"], title="Ranking Accuracy (CI)")
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
