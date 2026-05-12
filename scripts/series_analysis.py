#!/usr/bin/env python3
"""Lightweight helpers for post-hoc experiment-series analysis.

This module is intentionally Colab-friendly:

- no torch import
- no project runtime dependencies
- works from `runs/experiment_registry.csv`
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "duration_sec",
    "splitter_seed",
    "batch_run_index",
    "batch_n_times",
    "best_epoch",
    "epochs_completed",
    "test_rmse",
    "test_pearson",
    "test_ci",
]


@dataclass(frozen=True)
class SeriesSummary:
    experiment_name: str
    observed: int
    success: int
    duration_mean: float | None
    duration_std: float | None
    pearson_mean: float | None
    pearson_std: float | None
    rmse_mean: float | None
    rmse_std: float | None
    ci_mean: float | None
    ci_std: float | None


def load_registry(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def filter_registry(
    df: pd.DataFrame,
    experiment_names: Iterable[str] | None = None,
    prefixes: Iterable[str] | None = None,
    status: str | None = None,
) -> pd.DataFrame:
    result = df.copy()
    if experiment_names:
        names = set(experiment_names)
        result = result[result["experiment_name"].isin(names)]
    if prefixes:
        prefixes = tuple(prefixes)
        result = result[result["experiment_name"].fillna("").str.startswith(prefixes)]
    if status is not None:
        result = result[result["status"].fillna("") == status]
    return result.reset_index(drop=True)


def summarize_series(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[SeriesSummary] = []
    for experiment_name, group in df.groupby("experiment_name", dropna=False):
        success = group["status"].fillna("").eq("success")
        success_group = group[success]
        rows.append(
            SeriesSummary(
                experiment_name=str(experiment_name),
                observed=int(len(group)),
                success=int(success.sum()),
                duration_mean=_nan_or_none(success_group["duration_sec"].mean()),
                duration_std=_nan_or_none(success_group["duration_sec"].std(ddof=0)),
                pearson_mean=_nan_or_none(success_group["test_pearson"].mean()),
                pearson_std=_nan_or_none(success_group["test_pearson"].std(ddof=0)),
                rmse_mean=_nan_or_none(success_group["test_rmse"].mean()),
                rmse_std=_nan_or_none(success_group["test_rmse"].std(ddof=0)),
                ci_mean=_nan_or_none(success_group["test_ci"].mean()),
                ci_std=_nan_or_none(success_group["test_ci"].std(ddof=0)),
            )
        )
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df
    return summary_df.sort_values("experiment_name").reset_index(drop=True)


def plot_metric_summary_scatter(
    df: pd.DataFrame,
    *,
    x_metric: str,
    y_metric: str,
    experiment_names: Iterable[str] | None = None,
    figsize: tuple[float, float] = (7.5, 5.5),
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
) -> plt.Axes:
    data = df.copy()
    if experiment_names:
        names = set(experiment_names)
        data = data[data["experiment_name"].isin(names)]
    data = data[data["status"].fillna("") == "success"]
    data = data.dropna(subset=[x_metric, y_metric])
    if data.empty:
        raise ValueError(f"No successful rows with both {x_metric} and {y_metric}")

    fig, ax = plt.subplots(figsize=figsize)
    for experiment_name, group in data.groupby("experiment_name"):
        x_mean = float(group[x_metric].mean())
        y_mean = float(group[y_metric].mean())
        x_err = float(group[x_metric].std(ddof=0)) if len(group) > 1 else 0.0
        y_err = float(group[y_metric].std(ddof=0)) if len(group) > 1 else 0.0
        ax.errorbar(
            x_mean,
            y_mean,
            xerr=x_err,
            yerr=y_err,
            fmt="o",
            markersize=7,
            capsize=4,
            alpha=0.9,
            label=experiment_name,
        )
        # ax.annotate(
        #     experiment_name,
        #     (x_mean, y_mean),
        #     textcoords="offset points",
        #     xytext=(5, 5),
        #     ha="left",
        #     fontsize=8,
        # )
    ax.set_title(title or f"Series Comparison: mean {x_metric} vs mean {y_metric}")
    ax.set_xlabel(x_label or x_metric)
    ax.set_ylabel(y_label or y_metric)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    return ax


def load_series_histories(
    df: pd.DataFrame,
    experiment_name: str,
    *,
    runs_dir: str | Path = "runs",
    only_success: bool = True,
) -> pd.DataFrame:
    runs_dir = Path(runs_dir)
    series_df = df[df["experiment_name"] == experiment_name].copy()
    if only_success:
        series_df = series_df[series_df["status"].fillna("") == "success"]
    if series_df.empty:
        return pd.DataFrame(
            columns=[
                "experiment_signature",
                "splitter_seed",
                "batch_run_index",
                "epoch",
                "train_rmse",
                "val_rmse",
                "train_loss",
                "val_loss",
                "train_pearson",
                "val_pearson",
            ]
        )

    rows: list[dict] = []
    for _, run_row in series_df.iterrows():
        exp_dir_value = run_row.get("exp_dir", "")
        exp_dir = Path(exp_dir_value) if isinstance(exp_dir_value, str) and exp_dir_value else runs_dir / str(run_row["experiment_signature"])
        history_path = exp_dir / "history.json"
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        epoch_count = max((len(v) for v in history.values() if isinstance(v, list)), default=0)
        for epoch_index in range(epoch_count):
            rows.append(
                {
                    "experiment_signature": run_row.get("experiment_signature"),
                    "splitter_seed": run_row.get("splitter_seed"),
                    "batch_run_index": run_row.get("batch_run_index"),
                    "epoch": epoch_index + 1,
                    "train_rmse": _list_value(history.get("train_rmse"), epoch_index),
                    "val_rmse": _list_value(history.get("val_rmse"), epoch_index),
                    "train_loss": _list_value(history.get("train_loss"), epoch_index),
                    "val_loss": _list_value(history.get("val_loss"), epoch_index),
                    "train_pearson": _list_value(history.get("train_pearson"), epoch_index),
                    "val_pearson": _list_value(history.get("val_pearson"), epoch_index),
                }
            )
    return pd.DataFrame(rows)


def summarize_history_metric_by_epoch(history_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if history_df.empty:
        raise ValueError("No history rows available")
    if metric not in history_df.columns:
        raise ValueError(f"Unknown history metric: {metric}")

    data = history_df.dropna(subset=[metric]).copy()
    if data.empty:
        raise ValueError(f"No history rows with metric {metric}")

    summary = data.groupby("epoch")[metric].agg(["median", "mean", "std", "count", "min", "max"]).reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    quantiles = (
        data.groupby("epoch")[metric]
        .quantile([0.25, 0.75])
        .unstack()
        .reset_index()
        .rename(columns={0.25: "q25", 0.75: "q75"})
    )
    summary = summary.merge(quantiles, on="epoch", how="left")
    summary["lower_std"] = summary["median"] - summary["std"]
    summary["upper_std"] = summary["median"] + summary["std"]
    summary["lower_sem"] = summary["mean"] - summary["sem"]
    summary["upper_sem"] = summary["mean"] + summary["sem"]
    summary["t_crit_95"] = summary["count"].apply(_t_critical_95_two_sided)
    summary["ci95_normal_half_width"] = 1.96 * summary["sem"]
    summary["ci95_t_half_width"] = summary["t_crit_95"] * summary["sem"]
    summary["lower_ci95_normal"] = summary["mean"] - summary["ci95_normal_half_width"]
    summary["upper_ci95_normal"] = summary["mean"] + summary["ci95_normal_half_width"]
    summary["lower_ci95_t"] = summary["mean"] - summary["ci95_t_half_width"]
    summary["upper_ci95_t"] = summary["mean"] + summary["ci95_t_half_width"]
    return summary


def plot_series_panel(
    df: pd.DataFrame,
    experiment_names: Iterable[str],
    metric: str,
    *,
    runs_dir: str | Path = "runs",
    only_success: bool = True,
    figsize: tuple[float, float] = (11, 6),
    title: str | None = None,
    show_curves: bool = False,
    curve_alpha: float = 0.22,
    curve_linewidth: float = 1.0,
    show_band: bool = True,
    band_mode: str = "iqr",
    band_alpha: float = 0.14,
    show_center: bool = True,
    center_linewidth: float = 2.6,
    show_count_drop_ticks: bool = True,
    show_count_labels: bool = True,
) -> plt.Axes:
    experiment_names = list(experiment_names)
    if not experiment_names:
        raise ValueError("No experiment_names provided")
    if "experiment_name" not in df.columns:
        raise ValueError(
            "plot_series_panel() expects a registry-like dataframe with an 'experiment_name' column. "
            "Pass registry_view, not history_df."
        )

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")

    plotted_any = False
    single_series = len(experiment_names) == 1

    for idx, experiment_name in enumerate(experiment_names):
        history_df = load_series_histories(
            df,
            experiment_name,
            runs_dir=runs_dir,
            only_success=only_success,
        )
        if history_df.empty:
            continue

        summary = summarize_history_metric_by_epoch(history_df, metric)
        x, center, lower, upper, center_label, band_label = _resolve_band_mode(summary, band_mode=band_mode)
        color = cmap(idx % 10)
        curve_color = color
        line_label = center_label if single_series else experiment_name
        band_drawn = False

        if show_band:
            ax.fill_between(
                x,
                lower,
                upper,
                color=color,
                alpha=band_alpha,
                label=band_label if single_series else None,
                zorder=1,
            )
            band_drawn = True

        if show_curves:
            curve_data = history_df.dropna(subset=[metric]).copy()
            for _, group in curve_data.groupby("experiment_signature"):
                group = group.sort_values("epoch")
                ax.plot(
                    group["epoch"],
                    group[metric],
                    color=curve_color,
                    alpha=curve_alpha,
                    linewidth=curve_linewidth,
                    zorder=2 if band_drawn else 1,
                )

        if show_center:
            ax.plot(
                x,
                center,
                color=color,
                linewidth=center_linewidth if single_series else max(center_linewidth - 0.2, 1.0),
                label=line_label,
                zorder=3,
            )

        if show_count_drop_ticks:
            _draw_count_drop_ticks(ax, summary, color=color, show_labels=show_count_labels)

        plotted_any = True

    if not plotted_any:
        raise ValueError("No plottable series found for the requested experiment_names and metric")

    mode_bits = []
    if show_center:
        mode_bits.append("center")
    if show_curves:
        mode_bits.append("curves")
    if show_band:
        mode_bits.append(f"{band_mode} band")
    mode_suffix = " + ".join(mode_bits) if mode_bits else "no layers"

    ax.set_title(title or f"{metric}: {mode_suffix}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    if show_center or (single_series and show_band):
        ax.legend(loc="best", fontsize=8)
    elif not single_series:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    return ax


def _draw_count_drop_ticks(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    color: object,
    show_labels: bool,
) -> None:
    if summary.empty or "count" not in summary.columns:
        return

    counts = summary["count"].to_numpy(dtype=float)
    epochs = summary["epoch"].to_numpy(dtype=float)
    medians = summary["median"].to_numpy(dtype=float)
    if len(counts) == 0 or len(epochs) == 0 or len(medians) == 0:
        return

    y_min = float(np.nanmin(summary["lower_std"].to_numpy(dtype=float)))
    y_max = float(np.nanmax(summary["upper_std"].to_numpy(dtype=float)))
    y_span = max(y_max - y_min, 1e-9)
    tick_half_height = 0.018 * y_span
    tick_epochs = [0.0]
    tick_medians = [medians[0]]
    tick_counts = [int(counts[0])]
    prev_count = counts[0]
    for idx in range(1, len(counts)):
        current_count = counts[idx]
        if current_count < prev_count:
            tick_epochs.append(float(epochs[idx]))
            tick_medians.append(float(medians[idx]))
            tick_counts.append(int(current_count))
        prev_count = current_count

    if tick_epochs:
        tick_epochs_arr = np.asarray(tick_epochs, dtype=float)
        tick_medians_arr = np.asarray(tick_medians, dtype=float)
        ax.vlines(
            tick_epochs_arr,
            tick_medians_arr - tick_half_height,
            tick_medians_arr + tick_half_height,
            colors=color,
            linewidth=1.6,
            alpha=0.95,
        )
        if show_labels:
            for epoch, y, count in zip(tick_epochs_arr, tick_medians_arr, tick_counts):
                ax.annotate(
                    f"{count}",
                    (epoch, y + tick_half_height),
                    textcoords="offset points",
                    xytext=(3, 2),
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color=color,
                )


def _resolve_band_mode(
    summary: pd.DataFrame,
    *,
    band_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    x = summary["epoch"].to_numpy(dtype=float)
    if band_mode == "std":
        center = summary["median"].to_numpy(dtype=float)
        lower = summary["lower_std"].to_numpy(dtype=float)
        upper = summary["upper_std"].to_numpy(dtype=float)
        return x, center, lower, upper, "Median", "Median ± 1 std"
    if band_mode == "iqr":
        center = summary["median"].to_numpy(dtype=float)
        lower = summary["q25"].to_numpy(dtype=float)
        upper = summary["q75"].to_numpy(dtype=float)
        return x, center, lower, upper, "Median", "IQR (q25-q75)"
    if band_mode == "sem":
        center = summary["mean"].to_numpy(dtype=float)
        lower = summary["lower_sem"].to_numpy(dtype=float)
        upper = summary["upper_sem"].to_numpy(dtype=float)
        return x, center, lower, upper, "Mean", "Mean ± 1 SEM"
    if band_mode == "ci95_normal":
        center = summary["mean"].to_numpy(dtype=float)
        lower = summary["lower_ci95_normal"].to_numpy(dtype=float)
        upper = summary["upper_ci95_normal"].to_numpy(dtype=float)
        return x, center, lower, upper, "Mean", "Mean 95% CI (normal)"
    if band_mode == "ci95_t":
        center = summary["mean"].to_numpy(dtype=float)
        lower = summary["lower_ci95_t"].to_numpy(dtype=float)
        upper = summary["upper_ci95_t"].to_numpy(dtype=float)
        return x, center, lower, upper, "Mean", "Mean 95% CI (Student t)"
    raise ValueError(
        f"Unsupported band_mode: {band_mode}. "
        "Use 'iqr', 'std', 'sem', 'ci95_normal', or 'ci95_t'."
    )


def _t_critical_95_two_sided(count: float) -> float:
    n = int(count) if not pd.isna(count) else 0
    if n <= 1:
        return 0.0
    df = n - 1
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    if df in table:
        return table[df]
    if df < 40:
        return 2.021
    if df < 60:
        return 2.000
    if df < 120:
        return 1.980
    return 1.960


def _list_value(values: object, index: int) -> float | None:
    if not isinstance(values, list):
        return None
    if index >= len(values):
        return None
    value = values[index]
    if pd.isna(value):
        return None
    return float(value)


def _nan_or_none(value: float) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
