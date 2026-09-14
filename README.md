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

```bash
.venv/Scripts/python scripts/train_model.py   # fits the classifier, writes web/data.json
.venv/Scripts/python scripts/build_web.py     # inlines the payload into web/index.html
.venv/Scripts/python -m http.server 8777 --directory web
```

`web/app.html` is the version-controlled page; `web/index.html` is the generated single-file
build with the payload inlined, and is what you open.

The model is a `HistGradientBoostingClassifier` over vehicle-level features taken at a cut
point: counter wear rates (lifetime and 5-readout window), row-normalised histogram shapes
plus per-family entropy and volume, reporting-gap statistics, and the eight spec categories.
Training rows are five cut points sampled uniformly per training vehicle, which reproduces
how validation and test truncate each series at a random readout and keeps the class prior
comparable.

The decision rule is the Bayes action under the challenge loss: predict
`argmin_j sum_i P(class=i) * Cost[i][j]`, not the most likely class.

| Policy | Validation cost | Test cost |
|---|---|---|
| Model, minimum expected cost | **39,846** | **42,649** |
| Best trivial baseline (always class 4) | 49,566 | 49,671 |
| Most likely class (argmax) | 57,400 | 56,114 |
| Always class 0 | 57,400 | 56,100 |

Argmax collapses onto the do-nothing policy: at a 2.7% positive rate the likeliest class is
essentially always 0, so a model scored that way never raises an alarm. The web console lets
you rescale the cost matrix and switch rules to watch that happen.

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
