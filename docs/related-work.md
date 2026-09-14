# Published results on SCANIA Component X

All numbers are total challenge cost on the 5,045-vehicle test split, lower is better.

| Work | Model | Test cost |
|---|---|---|
| Empirical Study (arXiv 2606.12486), best CatBoost | CatBoost | 36,724 |
| Empirical Study, headline | XGBoost | 37,733 |
| Zhong & Wang, IDA 2024 LNCS 14642 pp. 268-276 | Bi-LSTM | 39,123 |
| Parton et al., IDA 2024 LNCS 14642 pp. 251-259 | GNN on signature-augmented graphs | 47,612 |
| Carpentier, De Temmerman & Verbeke, IDA 2024 | XGBoost | 49,671 |

The Carpentier figure equals this repo's computed `always_4` baseline to the unit, which is the
evidence that our cost function and test split match the published ones.

## What the leading approach does differently

From the empirical study, which reports the lowest published cost:

1. **One training row per vehicle, taken at the last observation.** They collapse each series to its
   final readout and label it from the time-to-event table. This repo samples five random cut points
   per vehicle instead. Their construction matches the inference distribution exactly (evaluation
   also scores the last readout of a truncated series) and naturally weights the positive classes
   toward the imminent ones, which is the prior mismatch measured in `scripts/error_analysis.py`.
2. **Counter noise correction.** They separate two corruption types in the cumulative counters:
   glitches (transient deviations, imputed for a smooth transition) and step noise (persistent
   drops, corrected by adding each step's magnitude to all later observations). This repo leaves
   both untouched; the negative counter diffs found during EDA are the same artefact.
3. **Linear interpolation of missing values** per series, rather than leaving them for the model.

They report no ablation, so which of the three carries the gain is unmeasured.

## Ideas from the related APS dataset

The UCI APS Failure at Scania Trucks set (IDA 2016) is a different problem - one snapshot per truck,
binary target, no time axis - so its costs are not comparable. The transferable idea is SMOTE-style
synthetic minority sampling, which unlike resampling existing rows adds new points in feature space.
