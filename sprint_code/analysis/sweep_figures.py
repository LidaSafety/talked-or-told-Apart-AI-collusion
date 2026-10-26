"""Figures for the scripted bins / threshold / run-length sweep (results/scripted_sweep/).
Scripted markets only: a power analysis, not evidence about LLM behaviour.

    .venv/bin/python -m sprint_code.analysis.sweep_figures

Reads results/scripted_sweep/power.csv for the main bins and writes to results/scripted_sweep/figures/:
    sample_efficiency.{png,pdf}      detection of a coordinating pair, and false pairs without one, against run length
    coordination_strength.{png,pdf}  detection at T = 150 against the share of rounds the pair follows its scheme
    shared_signal.{png,pdf}          false pairs when two bidders share a signal but have no channel (S2)
    figure_data.csv                  every plotted value with its exact 95% interval
The own-cost and price-level rates are expectations over random max-of-9 calibration sets; their intervals
treat rate x runs, rounded, as a binomial count.
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from sprint_code.analysis.main_analysis import exact_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "results/scripted_sweep"
OUT = SWEEP / "figures"
TS, PS, T_MAIN = (60, 120, 150, 240), (0.2, 0.4, 0.6, 0.8, 1.0), 150
# dataviz reference palette, light; slots 1-3 validated all-pairs, blue 250 -> 450 validated as an ordinal pair
BLUE, BLUE_LIGHT, ORANGE, AQUA = "#2a78d6", "#86b6ef", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, AXIS, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
# (power.csv prefix, label, colour); the price level is the existing baseline, so it takes the de-emphasis gray
DETECTORS = [("netinf", "Net influence (bid dependence)", BLUE), ("cross", "Cross-cost information", ORANGE),
             ("own", "Own-cost rule", AQUA), ("level", "Price level (baseline)", MUTED)]
DODGE = 0.03                     # horizontal offset between detectors, as a share of the x range: keeps markers apart

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False})


def power() -> dict:
    return {(r["cell"], int(r["T"]), float(r["p"]), r["view"]): r
            for r in csv.DictReader((SWEEP / "power.csv").open()) if r["bins"] == "A"}


def series(ax, xs, rows, column, colour, dodge, data, labels):
    """One detector: line, exact 95% interval bars and ringed markers; appends the plotted values to `data`."""
    ns = [int(r["n"]) for r in rows]
    rates = [float(r[column]) for r in rows]
    ci = [exact_ci(round(v * n), n) for v, n in zip(rates, ns)]
    x = [v + dodge for v in xs]
    ax.vlines(x, [c[0] for c in ci], [c[1] for c in ci], color=colour, lw=1.0, alpha=0.75, zorder=2)
    ax.plot(x, rates, color=colour, lw=1.6, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.plot(x, rates, ls="none", marker="o", ms=5.5, mfc=colour, mec=SURFACE, mew=1.3, zorder=4)
    for xv, r, v, c in zip(xs, rows, rates, ci):
        data.append({**labels, "x": xv, "cell": r["cell"], "T": r["T"], "p": r["p"], "view": r["view"],
                     "measure": column, "n": r["n"], "rate": round(v, 4), "ci_low": round(c[0], 4),
                     "ci_high": round(c[1], 4)})


def style(ax, title, subtitle, xs, xlabel, ylabel=None, ymax=1.0):
    ax.set_title(title, loc="left", pad=15)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=7.8, color=INK2, va="bottom")
    ax.set_xticks(xs)
    ax.set_xlim(xs[0] - 0.09 * (xs[-1] - xs[0]), xs[-1] + 0.09 * (xs[-1] - xs[0]))
    ax.set_ylim(-0.03 * ymax, 1.04 * ymax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def handles(entries):
    return [Line2D([], [], color=c, lw=1.6, marker="o", ms=5.5, mfc=c, mec=SURFACE, mew=1.3) for c in entries]


def header(fig, title, note, legend_y=None):
    fig.text(0.012, 0.985, title, fontsize=11.5, fontweight="bold", color=INK, va="top")
    if legend_y is not None:
        fig.legend(handles(c for *_, c in DETECTORS), [lab for _, lab, _ in DETECTORS], loc="upper left",
                   bbox_to_anchor=(0.012, legend_y), ncol=4, fontsize=8, handlelength=1.8, columnspacing=1.4,
                   labelcolor=INK2)
    fig.text(0.012, 0.012, textwrap.fill(note, 132), fontsize=7.2, color=INK2, va="bottom", linespacing=1.35)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def sample_efficiency(pw, data):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.3))
    panels = [("S3a", 0.6, "correct", "Designated winner", "higher-cost member covers at 11; p = 0.6"),
              ("S3b", 0.6, "correct", "Shared varying target", "both quote a common target (8-11); p = 0.6"),
              ("S4", 1.0, "correct", "Constant pair quote", "both pair members quote 11 every round"),
              ("S0", 1.0, "false_pair", "No coordination", "independent null bidders")]
    span = TS[-1] - TS[0]
    for k, (ax, (cell, p, measure, title, sub)) in enumerate(zip(axes.flat, panels)):
        if cell == "S0":
            ax.axhspan(0.02, 0.08, color=BAND, zorder=0)
            ax.text(TS[-1] + 0.09 * span, 0.205, "shaded: target band for net influence", fontsize=7.2,
                    color=MUTED, ha="right", va="top")
        for j, (pre, lab, colour) in enumerate(DETECTORS):
            series(ax, TS, [pw[(cell, T, p, "public")] for T in TS], f"{pre}_{measure}", colour,
                   (j - 1.5) * DODGE * span, data, {"figure": "sample_efficiency", "panel": title, "detector": lab})
        ax.axvline(T_MAIN, color=AXIS, lw=0.8, zorder=1)
        ylabel = "share of runs returning the true pair" if k == 0 or k == 2 else (
            "share of runs naming any pair" if cell == "S0" else None)
        style(ax, title, sub, TS, "run length T (rounds)" if k >= 2 else "", ylabel, ymax=0.2 if cell == "S0" else 1.0)
    header(fig, "Scripted markets: does each detector find the coordinating pair?",
           "Quote bins 1-4 / 5-8 / 9-12, previous price split at 3|4, burn-in 20 rounds; vertical line: the main-study run length. Net-influence threshold "
           "calibrated at each T on separate no-coordination runs; cross-cost needs both directions at p <= 0.05/3; "
           "own-cost and price-level rules use the max-of-9 calibration (about 1/10 false pairs by construction). "
           "100 runs per point (no coordination: 300). Bars: exact 95% intervals. Scripted bidders fitted to no-talk "
           "Gemini quotes; not evidence about LLM behaviour.", legend_y=0.945)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.835, bottom=0.165, hspace=0.62, wspace=0.27)
    save(fig, "sample_efficiency")


def coordination_strength(pw, data):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    panels = [("S3a", "Designated winner", "higher-cost member covers at 11"),
              ("S3b", "Shared varying target", "both quote a common target (8-11)")]
    span = PS[-1] - PS[0]
    for k, (ax, (cell, title, sub)) in enumerate(zip(axes, panels)):
        for j, (pre, lab, colour) in enumerate(DETECTORS):
            series(ax, PS, [pw[(cell, T_MAIN, p, "public")] for p in PS], f"{pre}_correct", colour,
                   (j - 1.5) * DODGE * span, data, {"figure": "coordination_strength", "panel": title, "detector": lab})
        style(ax, title, sub, PS, "share of rounds the pair follows its scheme, p",
              "share of runs returning the true pair" if k == 0 else None)
    header(fig, "Scripted markets: detection against coordination strength at T = 150",
           "In the other rounds the pair members quote like independent null bidders. Bins, burn-in and thresholds "
           "as in the sample-efficiency figure. 100 runs per point. Bars: exact 95% intervals. Scripted bidders; not "
           "evidence about LLM behaviour.", legend_y=0.885)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.70, bottom=0.25, wspace=0.18)
    save(fig, "coordination_strength")


def shared_signal(pw, data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.7))
    span = TS[-1] - TS[0]
    views = [("public", "public state (previous price)", BLUE),
             ("signal", "signal added to the auditor's state", BLUE_LIGHT)]
    for j, (view, lab, colour) in enumerate(views):
        series(a, TS, [pw[("S2", T, 1.0, view)] for T in TS], "netinf_false_pair", colour, (j - 0.5) * DODGE * span,
               data, {"figure": "shared_signal", "panel": "Net influence", "detector": f"Net influence, {lab}"})
    a.legend(handles(c for *_, c in views), [lab for _, lab, _ in views], loc="upper left", fontsize=7.8,
             labelcolor=INK2, handlelength=1.8)
    for j, (pre, lab, colour) in enumerate(DETECTORS[1:]):
        series(b, TS, [pw[("S2", T, 1.0, "public")] for T in TS], f"{pre}_false_pair", colour, (j - 1) * DODGE * span,
               data, {"figure": "shared_signal", "panel": "Cost- and price-based rules", "detector": lab})
    b.legend(handles(c for *_, c in DETECTORS[1:]), [lab for _, lab, _ in DETECTORS[1:]], loc="center right",
             fontsize=7.8, labelcolor=INK2, handlelength=1.8)
    for k, (ax, title, sub) in enumerate([(a, "Net influence", "two auditor states, same threshold"),
                                          (b, "Cost- and price-based rules",
                                           "same markets; unaffected by the auditor's state")]):
        ax.axvline(T_MAIN, color=AXIS, lw=0.8, zorder=1)
        style(ax, title, sub, TS, "run length T (rounds)", "share of runs naming any pair" if k == 0 else None)
    s2 = {view: pw[("S2", T_MAIN, 1.0, view)] for view in ("public", "signal")}
    count = {view: [round(float(r[c]) * int(r["n"])) for c in ("netinf_names_signal_pair", "netinf_false_pair")]
             for view, r in s2.items()}
    header(fig, "Scripted markets: a shared signal without a channel",
           "Two bidders add 2 ticks whenever a shared signal (uniform 4-11) is at least 9; they exchange nothing. "
           f"At T = {T_MAIN} (vertical line), {count['public'][0]} of the {count['public'][1]} false pairs in the "
           f"public state name the signal pair; with the signal in the state, {count['signal'][0]} of "
           f"{count['signal'][1]}. 200 runs per point. Bars: exact 95% intervals. Scripted bidders; not evidence "
           "about LLM behaviour.")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.25, wspace=0.18)
    save(fig, "shared_signal")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pw, data = power(), []
    sample_efficiency(pw, data)
    coordination_strength(pw, data)
    shared_signal(pw, data)
    with (OUT / "figure_data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    print(f"wrote {len(data)} plotted values and three figures to {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
