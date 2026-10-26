"""Figures for the main study, drawn from the analysis outputs.

    .venv/bin/python -m sprint_code.analysis.main_figures --model gemini-2.5-flash

Reads results/main_analysis/<model>/ (runs.csv, views.csv, directions.csv, contrasts.csv) and the main logs, and writes
results/main_analysis/<model>/figures/:
    detection_matrix.{png,pdf}  which pair each detector named, run by run, for every arm
    statistics.{png,pdf}        the statistics behind the decisions, against their thresholds
    economics.{png,pdf}         same-seed differences from no talk in price, efficiency and profits
    cost_response.{png,pdf}     mean quote by own cost: pair members, third dealer and the competitive quote
No new test is computed here. Figures name the conditions (no talk,
pair talk, shared signal, instructed pair, instructed pair with talk) rather than the arm codes R0, R1, R3, R4 and R5,
and draw only the arms present. The validated palette has three categorical hues, so both instructed conditions are
aqua and the one with talk (R5) is drawn with diamonds.
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from sprint_code.evaluation.economics import BNE_N3  # noqa: E402
from sprint_code.rfq_logs import load_runs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BURN_IN, ALPHA_CROSS = 20, 0.05 / 3
T975 = {5: 2.571, 8: 2.306, 17: 2.110}          # t(0.975, df) for six, nine and eighteen markets
# dataviz reference palette, light: categorical slots validated all-pairs; blue steps are the sequential ramp
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6"]          # share of a row: <=1/4, <=1/2, <=3/4, more
SURFACE, INK, INK2, GRID, AXIS, NA = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#f0efec"
ARM_COLOUR = {"R0_main": GRAY, "R0_heldout": GRAY, "R1": BLUE, "R3": ORANGE, "R4": AQUA, "R5": AQUA}
MARKER = {"R5": "D"}                         # secondary encoding: R5 shares R4's hue
MODEL_NAME = {"gemini-2.5-flash": "Gemini 2.5 Flash", "gpt-oss-20b-8k": "gpt-oss-20b"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False})


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def header(fig, title, note):
    fig.text(0.012, 0.985, title, fontsize=11.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.012, textwrap.fill(note, 136), fontsize=7.2, color=INK2, va="bottom", linespacing=1.35)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)


# --- 1. detection matrix ---

ROWS = [("R1", "AB"), ("R1", "AC"), ("R1", "BC"), ("R4", "AB"), ("R4", "AC"), ("R4", "BC"), ("R5", "AB"), ("R5", "AC"),
        ("R5", "BC"), ("R0_main", ""), ("R0_heldout", ""), ("R3", "AB"), ("R3", "AC"), ("R3", "BC")]
ROW_LABEL = {"R1": "Pair talk · {}", "R4": "Instructed pair · {}", "R5": "Instructed + talk · {}",
             "R0_main": "No talk · calibration", "R0_heldout": "No talk · held out", "R3": "Shared signal · {}"}
TARGET = ("R1", "R4", "R5")                                   # arms whose pair is the right answer
COLS = ["AB", "AC", "BC", "NO_PAIR"]
PANELS = [("netinf", "returned", "Net influence"), ("cross_cost", "returned", "Cross-cost information"),
          ("own_cost", "returned", "Own-cost rule"), ("level", "returned", "Price level"),
          ("cmi", "forced", "CMI, forced choice"), ("mi", "forced", "MI, forced choice")]


def detection_matrix(runs, out, model):
    rows = [(st, pr) for st, pr in ROWS if any(r["set"] == st for r in runs)]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 2.2 + 0.56 * len(rows)))
    for k, (ax, (det, kind, title)) in enumerate(zip(axes.flat, PANELS)):
        ax.set_xlim(0, len(COLS))
        ax.set_ylim(len(rows), 0)
        for i, (st, pr) in enumerate(rows):
            rs = [r for r in runs if r["set"] == st and (not pr or r["pair"] == pr)]
            answers = Counter(r[f"{kind}_{det}"] for r in rs)
            calibration = answers.get("calibration")
            for j, col in enumerate(COLS):
                x, y = j + 0.05, i + 0.08
                if (kind == "forced" and col == "NO_PAIR") or calibration:
                    ax.add_patch(Rectangle((x, y), 0.9, 0.84, facecolor=NA, edgecolor="none"))
                    continue
                c = answers.get(col, 0)
                share = c / len(rs) if rs else 0
                fill = SURFACE if c == 0 else RAMP[min(3, int(np.ceil(share * 4)) - 1)]
                ax.add_patch(Rectangle((x, y), 0.9, 0.84, facecolor=fill, edgecolor=GRID, lw=0.6))
                right = col == pr if st in TARGET else (col == "NO_PAIR" and kind == "returned")
                if right:
                    ax.add_patch(Rectangle((x, y), 0.9, 0.84, fill=False, edgecolor=INK, lw=1.6))
                ax.text(j + 0.5, i + 0.52, str(c) if c else "·", ha="center", va="center", fontsize=8.5 if c else 7.5,
                        fontweight="bold" if c else "normal", color="#ffffff" if share > 0.5 else INK)
            if calibration:
                ax.text(2.0, i + 0.52, "sets the threshold", ha="center", va="center", fontsize=7, color=INK2)
        ax.set_xticks([j + 0.5 for j in range(len(COLS))])
        ax.set_xticklabels(["AB", "AC", "BC", "none"])
        ax.xaxis.tick_top()
        ax.set_yticks([i + 0.5 for i in range(len(rows))])
        ax.set_yticklabels([ROW_LABEL[st].format(pr) for st, pr in rows] if k % 3 == 0 else [])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left", pad=20)
    header(fig, "Which pair did each detector name?",
           f"{MODEL_NAME.get(model, model)}. Columns: the pair named, or none. Outlined: the right answer (pair talk "
           "and instructed pair: the assigned pair; no talk and shared signal: none). Numbers are markets. Own-cost and "
           "price-level thresholds are the maxima over the no-talk calibration markets, which are therefore not "
           "scored. Forced choices always name a pair. Rounds 21-150, public state.")
    fig.subplots_adjust(left=0.175, right=0.99, top=0.9, bottom=0.075, hspace=0.25, wspace=0.08)
    save(fig, out, "detection_matrix")


# --- 2. statistics against their thresholds ---

def strip(ax, groups, rng):
    """groups: [(label, values, colour, hollow[, marker])]; points jittered around each group's x."""
    for x, (label, values, colour, hollow, *marker) in enumerate(groups):
        v = np.asarray(values, float)
        jit = rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(x + jit, v, s=22, facecolors=SURFACE if hollow else colour, edgecolors=colour, linewidths=1.1,
                   marker=marker[0] if marker else "o", zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.3)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def subtitle(ax, title, sub):
    ax.set_title(title, loc="left", pad=16)
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=7.6, color=INK2)


def statistics(runs, views, dirs, calibration, alpha, out, model):
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(7.6, 6.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.62, wspace=0.22)
    sel = lambda st: [r for r in runs if r["set"] == st]  # noqa: E731
    sig = [v for v in views if v["set"] == "R3" and v["view"] == "signal"]
    keep = lambda groups: [g for g in groups if len(g[1])]  # noqa: E731

    a = fig.add_subplot(grid[0, 0])
    floor = 1e-4                                  # log scale: a zero gap (a tie) is drawn at the floor
    groups = keep([("No talk", [float(r["netinf_gap"]) for r in sel("R0_main")], GRAY, False),
                   ("Held out", [float(r["netinf_gap"]) for r in sel("R0_heldout")], GRAY, True),
                   ("Pair\ntalk", [float(r["netinf_gap"]) for r in sel("R1")], BLUE, False),
                   ("Shared\nsignal", [float(r["netinf_gap"]) for r in sel("R3")], ORANGE, False),
                   ("Signal in\nstate", [float(v["gap"]) for v in sig], ORANGE, True),
                   ("Instructed\npair", [float(r["netinf_gap"]) for r in sel("R4")], AQUA, False),
                   ("Instructed\n+ talk", [float(r["netinf_gap"]) for r in sel("R5")], AQUA, False, "D")])
    strip(a, [(g[0], np.maximum(g[1], floor), *g[2:]) for g in groups], rng)
    a.set_yscale("log")
    a.set_ylim(floor * 0.7, 3)
    a.axhline(alpha, color=INK2, lw=0.9, zorder=2)
    a.text(-0.45, alpha / 1.15, f"α = {alpha:.4f}", fontsize=7.2, color=INK2, va="top")
    for x, (_, v, *_) in enumerate(groups):
        a.text(x, 0.95, f"{sum(np.array(v) >= alpha)}/{len(v)}", transform=a.get_xaxis_transform(), ha="center",
               va="top", fontsize=7.2, color=INK2)
    subtitle(a, "Net influence", "top-two CMI gap, bits (log scale)")

    c = fig.add_subplot(grid[0, 1])
    thr = calibration["thresholds"]["own_cost_gap"]
    groups = keep([("No talk", [float(r["own_cost_gap"]) for r in sel("R0_main")], GRAY, False),
                   ("Held out", [float(r["own_cost_gap"]) for r in sel("R0_heldout")], GRAY, True),
                   ("Pair\ntalk", [float(r["own_cost_gap"]) for r in sel("R1")], BLUE, False),
                   ("Shared\nsignal", [float(r["own_cost_gap"]) for r in sel("R3")], ORANGE, False),
                   ("Instructed\npair", [float(r["own_cost_gap"]) for r in sel("R4")], AQUA, False),
                   ("Instructed\n+ talk", [float(r["own_cost_gap"]) for r in sel("R5")], AQUA, False, "D")])
    strip(c, groups, rng)
    c.axhline(thr, color=INK2, lw=0.9, zorder=2)
    c.text(0.55, thr, f"threshold {thr:.3f}", fontsize=7.2, color=INK2, va="bottom")
    subtitle(c, "Own-cost rule", "gap R(3) − R(2), bits")

    b = fig.add_subplot(grid[1, :])
    d = lambda st, role: [x for x in dirs if x["set"] == st and x["role"] == role]  # noqa: E731
    groups = keep([("No talk", d("R0_main", "none") + d("R0_heldout", "none"), GRAY),
                   ("Pair talk:\npair", d("R1", "pair"), BLUE), ("Pair talk:\nother", d("R1", "outsider"), BLUE),
                   ("Shared signal:\npair", d("R3", "signal_pair"), ORANGE),
                   ("Shared signal:\nother", d("R3", "other"), ORANGE),
                   ("Instructed:\npair", d("R4", "pair"), AQUA), ("Instructed:\nother", d("R4", "outsider"), AQUA),
                   ("Instr. + talk:\npair", d("R5", "pair"), AQUA, "D"),
                   ("Instr. + talk:\nother", d("R5", "outsider"), AQUA, "D")])
    for x, (label, ds, colour, *marker) in enumerate(groups):
        mk = marker[0] if marker else "o"
        v = np.array([float(q["X_excess"]) for q in ds])
        hit = np.array([float(q["p_crt"]) <= ALPHA_CROSS for q in ds])
        jit = rng.uniform(-0.2, 0.2, len(v))
        b.scatter(x + jit[~hit], v[~hit], s=16, facecolors=SURFACE, edgecolors=colour, linewidths=0.9, marker=mk,
                  zorder=3)
        b.scatter(x + jit[hit], v[hit], s=22, facecolors=colour, edgecolors=colour, linewidths=1.0, marker=mk, zorder=4)
        b.text(x, 0.95, f"{int(hit.sum())}/{len(v)}", transform=b.get_xaxis_transform(), ha="center", va="top",
               fontsize=7.2, color=INK2)
    b.axhline(0, color=AXIS, lw=0.8, zorder=2)
    b.set_xticks(range(len(groups)))
    b.set_xticklabels([g[0] for g in groups], fontsize=6.9 if len(groups) > 7 else 7.3)
    b.set_ylim(top=b.get_ylim()[1] * 1.25)
    b.grid(axis="y", color=GRID, lw=0.8)
    b.set_axisbelow(True)
    subtitle(b, "Cross-cost information", "X(i ← j) minus its null mean, bits; one point per market and direction; "
             "\"other\": directions involving the third dealer")

    header(fig, "The statistics behind each decision",
           f"{MODEL_NAME.get(model, model)}. One point per market; for cross-cost information, one per market and "
           "direction, filled when p ≤ 0.05/3, with the count of filled points at the top. Net-influence counts are "
           "markets at or above α. Lines: the calibrated thresholds, α calibrated on scripted no-coordination markets and the "
           "own-cost threshold from the nine no-talk calibration markets. Hollow points: held-out no-talk markets, and "
           "shared-signal markets with the signal added to the conditioning state (descriptive).")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.17)
    save(fig, out, "statistics")


# --- 3. economics ---

ECON = [("price_paid", "Price paid", "ticks per unit"), ("efficiency", "Allocative efficiency", "share of rounds"),
        ("profit_pair", "Pair profit", "ticks per round"), ("profit_outsider", "Outsider profit", "ticks per round")]


def economics(runs, contrasts, out, model):
    rng = np.random.default_rng(1)
    r0 = {r["seed"]: r for r in runs if r["set"] == "R0_main"}
    p = {(c["analysis"], c["measure"]): float(c["p_value"]) for c in contrasts}
    arms = [(st, colour, label) for st, colour, label in (("R1", BLUE, "Pair\ntalk"), ("R3", ORANGE, "Shared\nsignal"),
                                                           ("R4", AQUA, "Instructed\npair"),
                                                           ("R5", AQUA, "Instructed\n+ talk"))
            if any(r["set"] == st and r["seed"] in r0 for r in runs)]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6))
    for ax, (m, title, unit) in zip(axes.flat, ECON):
        for x, (st, colour, _) in enumerate(arms):
            d = np.array([float(r[m]) - float(r0[r["seed"]][m]) for r in runs if r["set"] == st and r["seed"] in r0])
            ax.scatter(x + rng.uniform(-0.16, 0.16, len(d)), d, s=20, color=colour, marker=MARKER.get(st, "o"), zorder=3)
            ax.plot([x - 0.26, x + 0.26], [d.mean()] * 2, color=INK, lw=1.6, zorder=4)
            pv = p.get((f"same_seed_{st}_minus_R0", m))
            ax.text(x, 1.0, f"p = {pv:.3f}" if pv is not None else "", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7.2, color=INK2)
        ax.axhline(0, color=AXIS, lw=0.9, zorder=2)
        ax.set_xlim(-0.6, len(arms) - 0.4)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([label if len(arms) > 3 else label.replace("\n", " ") for _, _, label in arms], fontsize=7.6)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", pad=24)
        ax.text(0, 1.09, unit, transform=ax.transAxes, fontsize=7.6, color=INK2)
    header(fig, "What talking, a shared signal and an instruction did to the market",
           f"{MODEL_NAME.get(model, model)}. Each point is one seed: the market minus the no-talk market with the same "
           "seed, which faced identical costs. Bars: means. p: exact two-sided sign-flip tests over the seeds. Pair and "
           "outsider are the seed's talking, signal or instructed pair and the third dealer; in the no-talk market, the "
           "same seats. Rounds 21-150.")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.15, wspace=0.22, hspace=0.62)
    save(fig, out, "economics")


# --- 4. cost response ---

def by_cost(runs, seats_of) -> np.ndarray:
    """(runs, 6): mean quote at each own cost 1..6 over the given seats and rounds 21-150; NaN where absent."""
    out = np.full((len(runs), 6), np.nan)
    for k, run in enumerate(runs):
        q, c = run.quotes[BURN_IN:], run.costs[BURN_IN:]
        seats = seats_of(run)
        for cost in range(1, 7):
            mask = c[:, seats] == cost
            if mask.any():
                out[k, cost - 1] = q[:, seats][mask].mean()
    return out


def cost_response(model, out):
    main = ROOT / "results/sprint_runs/main"
    logs = {"R0": f"no_talk_R0_no_talk_{model}.jsonl", "R1": f"talk_R1_pair_talk_{model}.jsonl",
            "R3": f"no_talk_R3_shared_information_{model}.jsonl", "R4": f"no_talk_R4_instructed_pair_{model}.jsonl",
            "R5": f"talk_R5_instructed_talk_{model}.jsonl"}
    runs = {arm: rs for arm, f in logs.items() if (main / f).exists() and (rs := load_runs(main / f))}
    pair = lambda r: list(r.config.get("talk_seats") or r.config.get("signal_seats")  # noqa: E731
                          or r.config.get("instructed_seats"))
    outsider = lambda r: [3 - sum(pair(r))]  # noqa: E731
    panels = [p for p in (("R0", "No talk", [("all dealers", lambda r: [0, 1, 2], GRAY)]),
                          ("R1", "Pair talk", [("pair", pair, BLUE), ("third dealer", outsider, GRAY)]),
                          ("R3", "Shared signal", [("pair", pair, ORANGE), ("third dealer", outsider, GRAY)]),
                          ("R4", "Instructed pair", [("pair", pair, AQUA), ("third dealer", outsider, GRAY)]),
                          ("R5", "Instructed + talk", [("pair", pair, AQUA, "D"), ("third dealer", outsider, GRAY)]))
              if p[0] in runs]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.6, 3.2), sharey=True)
    costs = np.arange(1, 7)
    for ax, (arm, title, series) in zip(axes, panels):
        ax.step(costs, BNE_N3, where="mid", color=AXIS, lw=1.4, zorder=2)
        for j, (label, seats, colour, *marker) in enumerate(series):
            m = by_cost(runs.get(arm, []), seats)
            n = np.sum(~np.isnan(m), axis=0)
            mean = np.nanmean(m, axis=0)
            half = T975[len(m) - 1] * np.nanstd(m, axis=0, ddof=1) / np.sqrt(n)
            x = costs + (j - (len(series) - 1) / 2) * 0.12
            ax.vlines(x, mean - half, mean + half, color=colour, lw=1.0, alpha=0.8, zorder=3)
            ax.plot(x, mean, color=colour, lw=1.6, marker=marker[0] if marker else "o", ms=4.5, mfc=colour, mec=SURFACE,
                    mew=1.0, zorder=4, label=label)
        ax.set_xticks(costs)
        ax.set_xlabel("own cost (ticks)")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", pad=8)
        ax.legend(loc="upper left", fontsize=7.4, labelcolor=INK2, handlelength=1.6)
    axes[0].set_ylabel("mean quote (ticks)")
    axes[-1].text(5.2, BNE_N3[4] - 1.3, "competitive\nquote", fontsize=7, color=INK2, ha="center")
    header(fig, "Do quotes follow the dealer's own cost?",
           f"{MODEL_NAME.get(model, model)}. Mean quote at each own cost over rounds 21-150; bars are 95% intervals "
           "across markets (t distribution). Step line: the three-bidder equilibrium (competitive) quote.")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.22, wspace=0.08)
    save(fig, out, "cost_response")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-flash")
    a = ap.parse_args()
    src = ROOT / "results/main_analysis" / a.model
    out = src / "figures"
    out.mkdir(parents=True, exist_ok=True)
    runs, views, dirs = read(src / "runs.csv"), read(src / "views.csv"), read(src / "directions.csv")
    contrasts = read(src / "contrasts.csv")
    manifest = json.loads((src / "manifest.json").read_text())
    calibration = json.loads((ROOT / manifest["calibration"]["path"]).read_text())
    detection_matrix(runs, out, a.model)
    statistics(runs, views, dirs, calibration, manifest["settings"]["netinf_alpha_bits"], out, a.model)
    economics(runs, contrasts, out, a.model)
    cost_response(a.model, out)
    print(f"wrote four figures to {out.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
