"""Main-text figures of the report, drawn from the analysis outputs of both models.

    .venv/bin/python -m sprint_code.analysis.paper_figures

Writes results/paper_figures/:
    design.{png,pdf}             who sees what in the four conditions
    detection_summary.{png,pdf}  share of pair markets in which each detector named the assigned pair, forced choice
                                 against calibrated decision, for talking, instructed and instructed-and-talking pairs
    economics_main.{png,pdf}     same-seed differences from no talk in price paid, efficiency and pair profit (Gemini)
Conditions carry the report's names and numbers (R1 no talk, R2 pair talk, R3 shared signal, R4 instructed pair,
R5 instructed pair with talk); the analysis outputs keep the arm codes R0, R1, R3, R4 and R5. R5 shares R4's hue and
is drawn with diamonds (the validated palette has three categorical hues). No new test is computed here; titles and notes are in the report's captions.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from sprint_code.analysis.main_figures import AQUA, AXIS, BLUE, GRAY, GRID, INK, INK2, ORANGE, SURFACE, plt, save

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_figures"
MODELS = (("gemini-2.5-flash", "Gemini 2.5 Flash"), ("gpt-oss-20b-8k", "gpt-oss-20b"))


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


# --- 1. design ---

def design():
    fig, axes = plt.subplots(1, 5, figsize=(7.6, 1.95))
    conditions = [("R1  No talk", GRAY, None, "No messages."),
                  ("R2  Pair talk", BLUE, "talk", "A and B exchange a private\nmessage before each quote."),
                  ("R3  Shared signal", ORANGE, "signal", "A and B see the same\nnonbinding price."),
                  ("R4  Instructed pair", AQUA, "instruction", "A and B are told they have\nagreed to coordinate."),
                  ("R5  Instructed + talk", AQUA, "both", "Both: an instruction\nand a channel.")]
    pos = {"A": (0.25, 0.6), "B": (0.75, 0.6), "C": (0.5, 0.24)}
    for ax, (title, colour, kind, note) in zip(axes, conditions):
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.32, 0.98)
        ax.set_aspect("equal", anchor="N")
        ax.axis("off")
        ax.set_title(title, loc="left", pad=2, fontsize=8.6)
        if kind in ("instruction", "both"):
            ax.add_patch(FancyBboxPatch((0.09, 0.47), 0.82, 0.26, boxstyle="round,pad=0.02,rounding_size=0.12",
                                        facecolor="none", edgecolor=colour, lw=1.2, ls=(0, (3, 2)), zorder=2))
            ax.text(0.5, 0.84, "instruction", ha="center", va="center", fontsize=7.2, color=INK2)
        for d, (x, y) in pos.items():
            pair = d in "AB" and kind is not None
            ax.add_patch(Circle((x, y), 0.1, facecolor=colour if pair else SURFACE,
                                edgecolor=colour if pair else GRAY, lw=1.4, zorder=3))
            ax.text(x, y, d, ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color="#ffffff" if pair else INK, zorder=4)
        if kind in ("talk", "both"):
            ax.add_patch(FancyArrowPatch((0.36, 0.6), (0.64, 0.6), arrowstyle="<->", mutation_scale=9, color=INK2,
                                         lw=1.2, zorder=4))
            if kind == "talk":
                ax.text(0.5, 0.78, "messages", ha="center", va="center", fontsize=7.2, color=INK2)
        if kind == "signal":
            ax.add_patch(FancyBboxPatch((0.36, 0.83), 0.28, 0.11, boxstyle="round,pad=0.01,rounding_size=0.03",
                                        facecolor=SURFACE, edgecolor=colour, lw=1.1, zorder=3))
            ax.text(0.5, 0.885, "signal", ha="center", va="center", fontsize=7.2, color=INK2, zorder=4)
            for x in (0.31, 0.69):
                ax.add_patch(FancyArrowPatch((0.5 + (x - 0.5) * 0.35, 0.83), (x, 0.69), arrowstyle="->",
                                             mutation_scale=8, color=INK2, lw=1.0, zorder=2))
        ax.text(-0.04, 0.06, note, fontsize=6.8, color=INK2, va="top", linespacing=1.3)
    fig.subplots_adjust(left=0.015, right=0.995, top=0.9, bottom=0.01, wspace=0.08)
    save(fig, OUT, "design")


# --- 2. detection summary ---

ROWS = [("Mutual information", "mi", None), ("CMI | net influence", "cmi", "netinf"),
        ("Cross-cost information", "cross_cost", "cross_cost"), ("Own-cost information", "own_cost", "own_cost"),
        ("Price level", "level", "level")]
PANELS = [("gemini-2.5-flash", "R1", "Pair talk (R2)", BLUE, "o"), ("gemini-2.5-flash", "R4", "Instructed (R4)", AQUA, "o"),
          ("gemini-2.5-flash", "R5", "Instructed + talk (R5)", AQUA, "D"),
          ("gpt-oss-20b-8k", "R1", "Pair talk (R2)", BLUE, "o"), ("gpt-oss-20b-8k", "R4", "Instructed (R4)", AQUA, "o"),
          ("gpt-oss-20b-8k", "R5", "Instructed + talk (R5)", AQUA, "D")]


def detection_summary():
    endpoints = {m: read(ROOT / "results/main_analysis" / m / "endpoints.csv") for m, _ in MODELS}
    name = dict(MODELS)
    panels = [p for p in PANELS if any(e["set"] == p[1] and e["measure"] == "forced_correct" for e in endpoints[p[0]])]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.6))
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for k, (ax, (model, arm, title, colour, mk)) in enumerate(zip(axes.flat, panels)):
        ep = endpoints[model]

        def share(section, measure, det):
            e = [x for x in ep if (x["section"], x["set"], x["view"], x["measure"], x["subset"], x["detector"])
                 == (section, arm, "public", measure, "all", det)]
            return (int(e[0]["k"]), int(e[0]["n"])) if e else None

        for i, (_, forced, decided) in enumerate(ROWS):
            f = share("secondary", "forced_correct", forced)
            d = share("primary", "correct", decided) if decided else None
            if f and d:
                ax.plot([f[0] / f[1], d[0] / d[1]], [i, i], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
            if d:
                ax.scatter(d[0] / d[1], i, s=72, facecolors=SURFACE, edgecolors=colour, linewidths=1.5, marker=mk, zorder=3)
            if f:
                ax.scatter(f[0] / f[1], i, s=30, color=colour, marker=mk, zorder=4)
        n = share("secondary", "forced_correct", "cmi")[1]
        ax.axvline(1 / 3, color=AXIS, lw=1.0, ls=(0, (3, 2)), zorder=0)
        ax.set_xlim(-0.06, 1.06)
        ax.set_xticks([0, 1 / 3, 1])
        ax.set_xticklabels(["0", "1/3", "1"])
        ax.set_ylim(len(ROWS) - 0.4, -0.6)
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([r[0] for r in ROWS])
        ax.tick_params(axis="y", labelleft=k % 3 == 0)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", pad=16)
        ax.text(0, 1.03, f"{name[model].split()[0].replace('-20b', '')}, {n} markets", transform=ax.transAxes,
                fontsize=7.4, color=INK2)
    axes.flat[0].text(1 / 3 + 0.03, -0.45, "chance", ha="left", va="center", fontsize=7, color=INK2)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5.5, color=INK2, label="forced choice (always names a pair)"),
               plt.Line2D([], [], ls="", marker="o", ms=8, mfc=SURFACE, mec=INK2, mew=1.5,
                          label="calibrated decision (may name none)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.23, 1.0), ncol=2, fontsize=7.6,
               labelcolor=INK2, handletextpad=0.3, columnspacing=1.6)
    fig.subplots_adjust(left=0.235, right=0.99, top=0.86, bottom=0.06, wspace=0.12, hspace=0.6)
    save(fig, OUT, "detection_summary")


# --- 3. economics ---

ECON = [("price_paid", "Price paid", "ticks per unit"), ("efficiency", "Allocative efficiency", "share of rounds"),
        ("profit_pair", "Pair profit", "ticks per round")]
ARMS = [("R1", BLUE, "Pair\ntalk"), ("R3", ORANGE, "Shared\nsignal"), ("R4", AQUA, "Instructed\npair"),
        ("R5", AQUA, "Instructed\n+ talk")]


def economics_main():
    src = ROOT / "results/main_analysis/gemini-2.5-flash"
    runs, contrasts = read(src / "runs.csv"), read(src / "contrasts.csv")
    r0 = {r["seed"]: r for r in runs if r["set"] == "R0_main"}
    p = {(c["analysis"], c["measure"]): float(c["p_value"]) for c in contrasts}
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.3))
    for ax, (m, title, unit) in zip(axes, ECON):
        arms = [a for a in ARMS if any(r["set"] == a[0] for r in runs)]
        for x, (st, colour, _) in enumerate(arms):
            d = np.array([float(r[m]) - float(r0[r["seed"]][m]) for r in runs if r["set"] == st and r["seed"] in r0])
            ax.scatter(x + rng.uniform(-0.16, 0.16, len(d)), d, s=18, color=colour, marker="D" if st == "R5" else "o",
                       zorder=3)
            ax.plot([x - 0.26, x + 0.26], [d.mean()] * 2, color=INK, lw=1.6, zorder=4)
            ax.text(x, 1.0, f"p = {p[(f'same_seed_{st}_minus_R0', m)]:.3f}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6.9, color=INK2)
        ax.axhline(0, color=AXIS, lw=0.9, zorder=2)
        ax.set_xlim(-0.6, len(arms) - 0.4)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([label for _, _, label in arms], fontsize=7.4)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", pad=22)
        ax.text(0, 1.1, unit, transform=ax.transAxes, fontsize=7.6, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.8, bottom=0.17, wspace=0.38)
    save(fig, OUT, "economics_main")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    design()
    detection_summary()
    economics_main()
    print(f"wrote three figures to {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
