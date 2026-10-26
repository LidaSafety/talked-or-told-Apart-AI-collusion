# Talked or Told? Formation, Detection and Disruption of Collusion among LLM Dealers

Code and data for the report of the same name (Apart Research AI Collusion Research Sprint, October 2026).

Three LLM dealers quote in a repeated request-for-quote (RFQ) procurement auction in which each dealer's cost is private and redrawn every round. One randomly assigned pair talks privately, is told it has secretly agreed to coordinate, does both, or shares a signal. Information-theoretic statistics (mutual and conditional mutual information, net influence, cross-cost information) and price screens are scored on whether they name the pair. The models are Gemini 2.5 Flash and gpt-oss-20b.

## Conditions

| Report | Code and logs | Condition |
|---|---|---|
| R1 | R0 | No talk |
| R2 | R1 | Pair talk: one pair exchanges a private message before each round |
| R3 | R3 | Shared signal: two dealers see the same nonbinding price signal (Gemini only) |
| R4 | R4 | Instructed pair: the pair is told it has secretly agreed to coordinate, without a channel |
| R5 | R5 | Instructed pair with talk |

## Repository

| Path | Contents |
|---|---|
| `sprint_code/rfq_runner/` | Auction runner, conditions and prompts |
| `sprint_code/detectors/`, `sprint_code/cost_response/`, `sprint_code/evaluation/` | Information measures, pair localisation and economic outcomes |
| `sprint_code/scripted_controls/` | Scripted markets with known coordination |
| `sprint_code/analysis/` | Analysis, calibration and figures |
| `external/dealer_market/` | Simulator, prompts and model clients from the authors' earlier dealer-market study, and three of its auction logs |
| `results/sprint_runs/` | Every market log: one JSON record per model call |
| `results/main_analysis/<model>/` | Per-run statistics, detector decisions, contrasts and figures |
| `results/scripted_sweep/` | Scripted markets that set the bins, the run length and the net-influence threshold |
| `results/calibration/`, `results/coding/` | Calibrated thresholds; message coding of the Gemini pair-talk markets |
| `results/paper_figures/` | Figures of the report |
| `tests/` | Test suite |

The logs keep every prompt, message, quote, cost and outcome. The models' reasoning text is not released: the `reasoning` field of each record is empty.

## Setup

Python 3.12:

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

## Reproducing the analysis

```
.venv/bin/python -m sprint_code.analysis.main_analysis --model gemini-2.5-flash
.venv/bin/python -m sprint_code.analysis.main_analysis --model gpt-oss-20b-8k
.venv/bin/python -m sprint_code.analysis.interaction --model gemini-2.5-flash
.venv/bin/python -m sprint_code.analysis.interaction --model gpt-oss-20b-8k
.venv/bin/python -m sprint_code.analysis.fallback_check --model gpt-oss-20b-8k
.venv/bin/python -m sprint_code.analysis.main_figures --model gemini-2.5-flash
.venv/bin/python -m sprint_code.analysis.paper_figures
.venv/bin/python -m sprint_code.analysis.sweep_figures
```

The analysis reads the calibrated thresholds in `results/calibration/` and checks that they are committed and match the no-talk logs.

## Running markets

Gemini needs `GEMINI_API_KEY` in a `.env` file at the repository root. gpt-oss runs through Ollama's OpenAI-compatible API, as a model named `gpt-oss-20b-8k` (gpt-oss:20b with `num_ctx 8192`). For example:

```
.venv/bin/python -m sprint_code.rfq_runner.run --arm R1_pair_talk --model gemini-2.5-flash --runs 9 --first-seed 10 --T 150 --out-dir results/sprint_runs/main --confirm
```

The runner prints a cost estimate before any call and runs the test suite first.

## Provenance

The simulator, prompts and model clients in `external/dealer_market/`, and the quoting-market and two-dealer auction results reported in the paper, come from the authors' earlier dealer-market study, completed before the sprint. The pair-localization approach and the instructed-pair sentence come from an earlier pricing study by the authors. The three-dealer conditions, the detectors adapted to sealed quotes, the cross-cost and own-cost statistics, the scripted markets, all sprint markets and the analysis were produced during the sprint, 23–25 October 2026.

## Licence

MIT (see `LICENSE`).
