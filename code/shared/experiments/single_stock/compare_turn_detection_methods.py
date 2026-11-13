"""Compare completed Lo-style and BOCPD turn-signal diagnostics.

This script is read-only with respect to both experiments.  It never trains a
policy and writes only comparison artifacts under ``results/turn_comparison``.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = EXPERIMENT_DIR / "results"
LO_ROOT = RESULTS_ROOT / "confirmed_turns" / "signal_diagnostic"
BOCPD_ROOT = RESULTS_ROOT / "bocpd" / "signal_diagnostic"
OUTPUT_ROOT = RESULTS_ROOT / "turn_comparison"


def _require(path: Path) -> Path:
    if not path.exists():
        raise RuntimeError(f"Required completed diagnostic is missing: {path}")
    return path


def _signal_rows(
    method: str,
    root: Path,
    parameter_column: str,
    parameter_name: str,
) -> pd.DataFrame:
    holm = pd.read_csv(_require(root / "holm_summary.csv"))
    observations = pd.read_csv(_require(root / "observations.csv"))
    keys = ["ticker", "horizon", parameter_column]
    density = (
        observations.assign(
            is_turn=observations["confirmed_turn_state"].isin([1, 2]).astype(int)
        )
        .groupby(keys, as_index=False)
        .agg(n_observations=("is_turn", "size"), n_confirmed_turns=("is_turn", "sum"))
    )
    density["turns_per_100_trading_days"] = (
        100.0 * density["n_confirmed_turns"] / density["n_observations"]
    )
    merged = holm.merge(density, on=keys, how="left", validate="one_to_one")
    result = pd.DataFrame(
        {
            "method": method,
            "ticker": merged["ticker"],
            "horizon": merged["horizon"].astype(int),
            "parameter_name": parameter_name,
            "parameter_value": merged[parameter_column].astype(float),
            "trough_minus_peak_mean_return": merged["trough_minus_peak_mean_return"],
            "raw_p_value": merged["raw_p_value"],
            "holm_rank": merged["holm_rank"].astype(int),
            "holm_adjusted_threshold": merged["holm_adjusted_threshold"],
            "passes_holm_correction": merged["passes_holm_correction"].astype(bool),
            "positive_folds": merged["positive_folds"].astype(int),
            "n_confirmed_turns": merged["n_confirmed_turns"].astype(int),
            "n_observations": merged["n_observations"].astype(int),
            "turns_per_100_trading_days": merged["turns_per_100_trading_days"],
        }
    )
    return result


def build_comparison() -> tuple[pd.DataFrame, str]:
    rows = pd.concat(
        [
            _signal_rows("lo_causal", LO_ROOT, "k_multiplier", "k_multiplier"),
            _signal_rows("bocpd", BOCPD_ROOT, "hazard_lambda", "hazard_lambda"),
        ],
        ignore_index=True,
    ).sort_values(["method", "ticker", "horizon", "parameter_value"])
    lines = [
        "# Confirmed-turn detection method comparison",
        "",
        "本报告只读取两个实验已经完成的结果，不重新训练。",
        "",
        "| method | Holm显著数 | tests | turns/100d min | median | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, group in rows.groupby("method", sort=False):
        density = group["turns_per_100_trading_days"]
        lines.append(
            f"| {method} | {int(group['passes_holm_correction'].sum())} | {len(group)} | "
            f"{density.min():.3f} | {density.median():.3f} | {density.max():.3f} |"
        )
    lines.extend(["", "## 逐组合信号诊断与转折密度", "",
                  "| method | ticker | h | parameter | value | spread | raw p | Holm | turns/100d |",
                  "|---|---|---:|---|---:|---:|---:|:---:|---:|"])
    for row in rows.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.ticker} | {row.horizon} | {row.parameter_name} | "
            f"{row.parameter_value:g} | {row.trough_minus_peak_mean_return:.6f} | "
            f"{row.raw_p_value:.6f} | {'是' if row.passes_holm_correction else '否'} | "
            f"{row.turns_per_100_trading_days:.3f} |"
        )
    lo_stage2 = RESULTS_ROOT / "confirmed_turns" / "summary" / "stitched_significance.csv"
    bocpd_stage2 = RESULTS_ROOT / "bocpd" / "summary" / "stitched_significance.csv"
    lines.extend(["", "## 第二阶段", ""])
    if lo_stage2.exists() and bocpd_stage2.exists():
        lines.append(
            "两个方法都存在完整模型结果；其原始stitched结果路径已记录，"
            "后续应在相同指标列确认后再生成模型级差值，避免猜测schema。"
        )
    else:
        lines.append("至少一个方法尚未进入第二阶段，因此不比较vs buy-hold模型表现。")
    lines.append("")
    return rows, "\n".join(lines)


def _atomic_write(frame: pd.DataFrame, report: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "turn_detection_method_comparison.csv"
    report_path = OUTPUT_ROOT / "report.md"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=OUTPUT_ROOT, suffix=".tmp", delete=False
    ) as handle:
        frame.to_csv(handle, index=False)
        csv_tmp = Path(handle.name)
    csv_tmp.replace(csv_path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=OUTPUT_ROOT, suffix=".tmp", delete=False
    ) as handle:
        handle.write(report)
        report_tmp = Path(handle.name)
    report_tmp.replace(report_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    frame, report = build_comparison()
    if not np.isfinite(frame["turns_per_100_trading_days"]).all():
        raise RuntimeError("Non-finite turn density in comparison")
    _atomic_write(frame, report)
    print(f"Comparison complete: {OUTPUT_ROOT / 'report.md'}")


if __name__ == "__main__":
    main()
