# predictive-maintenance-time-series

EDA and modelling groundwork for machine failure prediction on real-world time series.

Current dataset: **SCANIA Component X** - multivariate operational readouts from ~33,000
trucks, with time-to-event labels for one anonymised engine component. Released CC BY 4.0 by
Scania CV AB via the Swedish National Data Service
([DOI 10.5878/jvb5-d390](https://doi.org/10.5878/jvb5-d390)).

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt    # Linux/macOS: .venv/bin/pip
python scripts/download_data.py                  # ~1.7 GB into data/raw/
```

## Notebooks

| Notebook | Contents |
|---|---|
| `notebooks/01_eda_scania_component_x.ipynb` | Full EDA: coverage audit, schema anatomy, missingness structure, time-axis regularity, survival/censoring, counter semantics, histogram normalisation, signal search, spec association, label windows, cost-matrix baselines |

Notebooks are paired with a jupytext `.py` percent script, which is the version-controlled
source of truth. After editing the notebook:

```bash
.venv/Scripts/jupytext --to py:percent notebooks/01_eda_scania_component_x.ipynb
```

To regenerate and execute from the script:

```bash
.venv/Scripts/jupytext --to notebook --output notebooks/01_eda_scania_component_x.ipynb notebooks/01_eda_scania_component_x.py
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_scania_component_x.ipynb
```

## Model and web console

The headline model is AutoGluon over LightGBM, trained on macmini-2 in the container (see
`docker/`). Training cut points and the validation rows are pooled into one table;
`StratifiedGroupKFold` assigns folds by `vehicle_id` so a vehicle's near-duplicate cut points never
straddle a fold; AutoGluon bags over those folds and its `predict_proba_oof()` is what the decision
rule is tuned on. Test is scored once at the end.

```bash
docker/run-seeds.sh --only GBM --time-limit 900      # on macmini-2: three seeds, mean and spread
.venv/Scripts/python scripts/build_web.py            # inlines web/data.json into web/index.html
.venv/Scripts/python -m http.server 8777 --directory web
```

Features are taken at a cut point: counter wear rates over 3/10/20-readout windows plus
acceleration against the vehicle's lifetime rate, row-normalised histogram shapes with per-family
entropy, drift against the first readout, reporting-gap statistics, and the eight spec categories.

The decision rule is the Bayes action under the challenge loss: predict
`argmin_j sum_i P(class=i) * Cost[i][j]`, not the most likely class.

| Policy | Test cost |
|---|---|
| **AutoGluon LightGBM, pooled, OOF-tuned rule (3 seeds)** | **35,469 ± 849** |
| HistGradientBoosting, single split, validation-tuned rule | 37,645 |
| Best trivial baseline (always class 4, check every truck) | 49,671 |
| Most likely class (argmax) | 56,100 |
| Always class 0 (check nothing) | 56,100 |

At the mean operating point the model catches 109 of the 142 at-risk trucks (recall 0.77) while
calling in roughly 2,100 of 5,045. Checking every truck catches all 142 but costs 40% more.

Seed-to-seed spread is the thing to watch: with a single held-out split and the rule tuned on its
136 at-risk vehicles, test cost varied by std 1,668 across seeds, wider than most of the
differences this repo's earlier experiments were chasing. Pooling validation into training and
tuning on out-of-fold predictions halved that to 849. `scripts/measure_noise.py` reproduces the
measurement.

Argmax collapses onto the do-nothing policy: at a 2.7% positive rate the likeliest class is
essentially always 0, so a model scored that way never raises an alarm. The web console lets
you rescale the cost matrix and switch rules to watch that happen.

Recall at the tuned point is 0.72 on test, at a precision around 0.05. That precision is
correct, not broken: a missed class-4 truck costs 500 and a wasted workshop check costs 10, so
one real catch pays for 50 false alarms. Optimising for precision here destroys value.

Ranking is the real constraint. `scripts/ranking_report.py` prints how many of the 142 at-risk
test trucks sit in the top-k by risk score:

| Features | top 50 | top 100 | top 500 |
|---|---|---|---|
| single 5-readout window | 3 | 12 | 43 |
| multi-window + acceleration + drift | 8 | 15 | 47 |

Better, still weak in absolute terms, and the cost rule compensates by alarming broadly.

## Package

`src/scania/` holds what the notebooks share:

- `schema.py` - column families, bin counts, the IDA 2024 cost matrix, class-window meanings
- `io.py` - split loading with float32 dtypes and a parquet cache (first load of the 1 GB
  train CSV is slow; later loads come from `data/cache/`)
- `cost.py` - total cost, cost per vehicle, constant-prediction baselines

## What the EDA found

1. 4,911 of 23,550 labelled train vehicles have **no operational readouts at all**, and their
   repair rate is half the covered fleet's (5.5% vs 10.8%). Not missing at random.
2. Sampling is **irregular**: gaps from 0.2 to 387 time steps, 5 to 303 readouts per vehicle.
   Fixed-stride windowing models "readouts ago", not elapsed operating time.
3. **89% of outcomes are censored.** Survival framing beats binary classification.
4. The eight scalar features are **cumulative counters** that essentially never reset, so the
   signal lives in the rate, not the level.
5. Missingness arrives **per histogram family**, all-or-nothing per row - impute at family
   level and keep a `reported` indicator.
6. The last readout sits a variable distance (median 2.2, max 324.6 time steps) before the end
   of follow-up, so a TTE target derived from it is noisy.
7. The cost matrix is so asymmetric that **predicting class 4 for every vehicle (49,566) is
   cheaper than predicting class 0 for every vehicle (57,400)**, despite class 0 being 97% of
   the fleet.

## Data licence

The dataset is CC BY 4.0, Scania CV AB. `data/` is gitignored - run the download script.
