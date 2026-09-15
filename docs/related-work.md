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

## What the sources actually say, and what they do not

Table VII of the empirical study (arXiv 2606.12486) is the origin of the three IDA 2024 figures above,
and it matches the table here row for row. Two things came out of reading it that change how the
comparison should be stated.

**Their headline is 37,733, not 36,724.** CatBoost reaches 36,724 on test in their Table III, but they
select XGBoost because it won on validation, and 37,733 is the number they carry into the
state-of-the-art table. Reporting 36,724 as the figure to beat picks their best row with knowledge of
the test set, which is the protocol error this repo avoids. Both rows are listed above; 37,733 is the
honest target.

**They also used AutoGluon.** Their Table II describes ten preconfigured AutoGluon configurations per
model family. The gap between their result and this repo's is therefore a difference of protocol and
feature construction, not of toolkit.

**No published work states its decision rule.** The empirical study does not describe how it converts
predicted class probabilities into a predicted class, and the two deep-learning papers are paywalled
(Springer LNCS 14642), so the same question is open for them. This matters because the rule is the
largest single lever measured in this repo: the same trained model costs 56,100 scored by most-likely
class and 35,469 under the Bayes minimum-expected-cost rule, a swing of 20,631. The entire spread
between first and last place in the table above is 13,947. Any claim that a given architecture
"lost" on this benchmark is unsafe until its decision rule is known.

**The GNN entry may not be solving the same task.** The empirical study describes Parton et al. as
applying Graph Isomorphism Networks for binary classification. If that characterisation is accurate,
the model cannot express the graded actions the 5x5 cost matrix prices, which would explain 47,612
without saying anything about graph networks as such. This is the citing paper's summary, not the
GNN paper's own words, and the primary source is paywalled.
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
