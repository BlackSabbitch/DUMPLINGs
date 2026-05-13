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


def load_series_readout_diagnostics(
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
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, run_row in series_df.iterrows():
        exp_dir_value = run_row.get("exp_dir", "")
        exp_dir = (
            Path(exp_dir_value)
            if isinstance(exp_dir_value, str) and exp_dir_value
            else runs_dir / str(run_row["experiment_signature"])
        )
        test_results_path = exp_dir / "test_results.json"
        if not test_results_path.exists():
            continue

        test_results = json.loads(test_results_path.read_text(encoding="utf-8"))
        readout = test_results.get("readout_diagnostics")
        if not isinstance(readout, dict):
            continue

        row: dict[str, object] = {
            "experiment_name": run_row.get("experiment_name"),
            "experiment_signature": run_row.get("experiment_signature"),
            "splitter_seed": run_row.get("splitter_seed"),
            "batch_run_index": run_row.get("batch_run_index"),
            "status": run_row.get("status"),
            "test_rmse": _coerce_float(test_results.get("RMSE", run_row.get("test_rmse"))),
            "test_pearson": _coerce_float(test_results.get("Pearson_R", run_row.get("test_pearson"))),
            "test_ci": _coerce_float(test_results.get("CI", run_row.get("test_ci"))),
            "readout_type": readout.get("type"),
            "mixer_has_bias": readout.get("mixer_has_bias"),
            "alpha": _coerce_float(readout.get("alpha")),
            "beta": _coerce_float(readout.get("beta")),
            "gamma": _coerce_float(readout.get("gamma")),
            "local_to_global_abs_contribution_ratio": _coerce_float(
                readout.get("local_to_global_abs_contribution_ratio")
            ),
        }

        for prefix in (
            "global_branch_output",
            "local_branch_output",
            "global_contribution",
            "local_contribution",
            "alpha_stats",
            "beta_stats",
            "gamma_stats",
            "local_to_global_abs_contribution_ratio_stats",
        ):
            block = readout.get(prefix)
            if not isinstance(block, dict):
                continue
            for key in ("mean", "std", "min", "max", "mean_abs"):
                if key in block:
                    row[f"{prefix}_{key}"] = _coerce_float(block.get(key))

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    diag_df = pd.DataFrame(rows)
    numeric_cols = [
        col
        for col in diag_df.columns
        if col
        not in {
            "experiment_name",
            "experiment_signature",
            "status",
            "readout_type",
            "mixer_has_bias",
        }
    ]
    for column in numeric_cols:
        diag_df[column] = pd.to_numeric(diag_df[column], errors="coerce")
    return diag_df.sort_values(["batch_run_index", "splitter_seed", "experiment_signature"]).reset_index(drop=True)


def summarize_a3_mixture_diagnostics(diag_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if diag_df.empty:
        raise ValueError("No readout diagnostics rows available")

    required = {
        "alpha",
        "beta",
        "gamma",
        "test_rmse",
        "test_pearson",
        "test_ci",
        "global_contribution_mean_abs",
        "local_contribution_mean_abs",
        "local_to_global_abs_contribution_ratio",
    }
    missing = sorted(required - set(diag_df.columns))
    if missing:
        raise ValueError(f"Missing required diagnostic columns: {', '.join(missing)}")

    parameter_cols = [
        "alpha",
        "beta",
        "gamma",
        "global_branch_output_mean",
        "local_branch_output_mean",
        "global_contribution_mean",
        "local_contribution_mean",
        "global_contribution_mean_abs",
        "local_contribution_mean_abs",
        "local_to_global_abs_contribution_ratio",
        "local_to_global_abs_contribution_ratio_stats_mean",
        "local_to_global_abs_contribution_ratio_stats_std",
        "test_rmse",
        "test_pearson",
        "test_ci",
    ]
    parameter_cols = [col for col in parameter_cols if col in diag_df.columns]

    summary_rows: list[dict[str, object]] = []
    for col in parameter_cols:
        series = pd.to_numeric(diag_df[col], errors="coerce").dropna()
        if series.empty:
            continue
        summary_rows.append(
            {
                "metric": col,
                "n": int(series.shape[0]),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("metric").reset_index(drop=True)

    correlation_pairs = [
        ("alpha", "test_rmse"),
        ("alpha", "test_pearson"),
        ("alpha", "test_ci"),
        ("beta", "test_rmse"),
        ("beta", "test_pearson"),
        ("beta", "test_ci"),
        ("gamma", "test_rmse"),
        ("gamma", "test_pearson"),
        ("gamma", "test_ci"),
        ("local_to_global_abs_contribution_ratio", "test_rmse"),
        ("local_to_global_abs_contribution_ratio", "test_pearson"),
        ("local_to_global_abs_contribution_ratio", "test_ci"),
        ("local_contribution_mean_abs", "test_rmse"),
        ("local_contribution_mean_abs", "test_pearson"),
        ("local_contribution_mean_abs", "test_ci"),
    ]

    corr_rows: list[dict[str, object]] = []
    for left, right in correlation_pairs:
        if left not in diag_df.columns or right not in diag_df.columns:
            continue
        pair = diag_df[[left, right]].dropna()
        if len(pair) < 2:
            corr = None
        else:
            corr = float(pair[left].corr(pair[right]))
        corr_rows.append(
            {
                "x": left,
                "y": right,
                "pearson_corr": corr,
                "n": int(len(pair)),
            }
        )
    corr_df = pd.DataFrame(corr_rows).sort_values(["x", "y"]).reset_index(drop=True)

    ranking_cols = [
        col
        for col in [
            "experiment_signature",
            "splitter_seed",
            "batch_run_index",
            "test_rmse",
            "test_pearson",
            "test_ci",
            "alpha",
            "beta",
            "gamma",
            "global_contribution_mean_abs",
            "local_contribution_mean_abs",
            "local_to_global_abs_contribution_ratio",
        ]
        if col in diag_df.columns
    ]
    by_pearson_df = diag_df[ranking_cols].sort_values(
        ["test_pearson", "test_ci", "test_rmse"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    by_rmse_df = diag_df[ranking_cols].sort_values(
        ["test_rmse", "test_pearson", "test_ci"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return {
        "per_run": diag_df.copy(),
        "summary": summary_df,
        "correlations": corr_df,
        "ranked_by_pearson": by_pearson_df,
        "ranked_by_rmse": by_rmse_df,
    }


def plot_a3_mixture_parameters(
    diag_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (10, 4.8),
    title: str | None = None,
) -> plt.Axes:
    required = ["alpha", "beta", "gamma"]
    missing = [col for col in required if col not in diag_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for parameter plot: {', '.join(missing)}")

    plot_df = diag_df.copy()
    x = np.arange(len(plot_df))
    labels = (
        plot_df.get("splitter_seed")
        if "splitter_seed" in plot_df.columns
        else plot_df.get("batch_run_index")
    )
    if labels is None:
        labels = pd.Series(range(1, len(plot_df) + 1))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, plot_df["alpha"], marker="o", linewidth=1.8, label="alpha")
    ax.plot(x, plot_df["beta"], marker="o", linewidth=1.8, label="beta")
    ax.plot(x, plot_df["gamma"], marker="o", linewidth=1.8, label="gamma (bias)")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in labels], rotation=0)
    ax.set_xlabel("Seed" if "splitter_seed" in plot_df.columns else "Run")
    ax.set_ylabel("Coefficient value")
    ax.set_title(title or "A3 mixer parameters by run")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    return ax


def plot_a3_local_ratio_vs_metric(
    diag_df: pd.DataFrame,
    *,
    metric: str = "test_pearson",
    figsize: tuple[float, float] = (6.5, 5.0),
    title: str | None = None,
) -> plt.Axes:
    x_col = "local_to_global_abs_contribution_ratio"
    required = [x_col, metric]
    missing = [col for col in required if col not in diag_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for ratio plot: {', '.join(missing)}")

    plot_df = diag_df.dropna(subset=required).copy()
    if plot_df.empty:
        raise ValueError(f"No rows available with both {x_col} and {metric}")

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        plot_df[x_col],
        plot_df[metric],
        c=plot_df.get("splitter_seed", pd.Series(np.arange(len(plot_df)))).to_numpy(),
        cmap="viridis",
        s=55,
        alpha=0.9,
    )
    for _, row in plot_df.iterrows():
        label = row.get("splitter_seed", row.get("batch_run_index", ""))
        ax.annotate(
            str(int(label)) if pd.notna(label) and float(label).is_integer() else str(label),
            (row[x_col], row[metric]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            fontsize=8,
        )
    corr = plot_df[x_col].corr(plot_df[metric]) if len(plot_df) >= 2 else np.nan
    ax.set_xlabel("local/global abs contribution ratio")
    ax.set_ylabel(metric)
    corr_text = f" (corr={corr:.3f})" if pd.notna(corr) else ""
    ax.set_title(title or f"{metric} vs local/global ratio{corr_text}")
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("splitter_seed" if "splitter_seed" in plot_df.columns else "run index")
    plt.tight_layout()
    return ax


def plot_a3_alpha_beta_scatter(
    diag_df: pd.DataFrame,
    *,
    color_metric: str = "test_rmse",
    figsize: tuple[float, float] = (6.5, 5.2),
    title: str | None = None,
) -> plt.Axes:
    required = ["alpha", "beta", color_metric]
    missing = [col for col in required if col not in diag_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for alpha/beta scatter: {', '.join(missing)}")

    plot_df = diag_df.dropna(subset=required).copy()
    if plot_df.empty:
        raise ValueError("No rows available for alpha/beta scatter")

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        plot_df["alpha"],
        plot_df["beta"],
        c=plot_df[color_metric],
        cmap="plasma",
        s=65,
        alpha=0.9,
    )
    for _, row in plot_df.iterrows():
        label = row.get("splitter_seed", row.get("batch_run_index", ""))
        ax.annotate(
            str(int(label)) if pd.notna(label) and float(label).is_integer() else str(label),
            (row["alpha"], row["beta"]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            fontsize=8,
        )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.4)
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_title(title or f"alpha vs beta colored by {color_metric}")
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_metric)
    plt.tight_layout()
    return ax


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


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
