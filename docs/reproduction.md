# Reproduction notes

## Protocols

| Setting | Strict (`hangzhou.json`, `guangzhou.json`) | Original (`legacy_*.json`) |
| --- | --- | --- |
| Station vocabulary and OD support | Training users' history; all nonself pairs | All retained records; observed OD pairs |
| Training targets | History window only | Full retained sequences |
| User condition | One shared token | User-index embedding |
| Split seed | Fixed at 42, separate from model seed | Model seed also determines split |
| Destination loss | Conditional on observed origin | Marginal destination probability |
| Primary aggregation | Pooled events | Equal mean over batch means for accuracy/NLL |
| Time point estimate | Conditional mixture median | Mean of 20 samples, each capped at 24 hours |

These settings represent distinct experiments and can produce different numbers. The strict configurations correspond to the validation-frozen AMTPP capacity settings; the original configurations retain the primary-table AMTPP setup. Macro F1 averages nonpadding classes with nonzero union support. Time MAE and RMSE use hours; time NLL is a negative log density with the hours-scale Jacobian.

## Configurations

| Config | History cutoff | Future begins | Minimum history | Rank | Mixtures | Batch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `hangzhou.json` | 2019-01-20 23:59:59 | 2019-01-21 | 40 | 3 | 16 | 128 |
| `guangzhou.json` | 2017-07-20 23:59:59 | 2017-07-21 | 55 | 1 | 16 | 128 |
| `legacy_hangzhou.json` | 2019-01-20 23:59:59 | 2019-01-21 | 40 | 3 | 16 | 256 |
| `legacy_guangzhou.json` | 2017-07-20 23:59:59 | 2017-07-21 | 55 | 3 | 16 | 256 |

The strict configurations use Adam, learning rate 0.001, weight decay 0.00001, at most 100 epochs, and patience 8. Original configurations use Adam, learning rate 0.001, no weight decay, at most 200 epochs, and patience 10. Embeddings have width 64, causal attention has 4 heads, and hidden width is 100. The example configuration reduces dimensions and epochs for a smoke run.

Select checkpoints by validation NLL. Run final evaluation using exactly the same configuration, data, seed, and topology. Newly trained checkpoints record the input pickle's SHA-256. Evaluation rejects a changed data file or protocol; strict evaluation also checks corpus counts and topology content/settings. Identical files can be moved to another directory. Historical checkpoints without a data hash emit a warning and explicitly report that data identity was not verified. For the strict experiments, model seeds 42–46 use split seed 42; supply `--seed` and use a separate output folder for each run. CUDA hardware and library differences can change numerical results.

To reproduce the original setup:

```bash
amtpp-run --config configs/legacy_hangzhou.json --data data/hangzhou.pkl --output outputs/legacy_hangzhou --device cuda:0
amtpp-run --config configs/legacy_hangzhou.json --data data/hangzhou.pkl --output outputs/legacy_hangzhou --device cuda:0 --evaluate
```

The legacy evaluator retains the original AMTPP metric functions and sampling procedure. The cleaned legacy trainer omits per-epoch test monitoring; model selection still uses validation loss. Training from scratch requires the original prepared records. Replaying the manuscript's fitted predictions additionally requires the corresponding fitted checkpoints; neither is distributed here.

The released code has been checked through analytic distribution tests, model/protocol invariants, and training/evaluation on artificial records. This establishes executable workflows and consistency of the checked calculations. It does not establish independent numerical reproduction of every published primary-table value; those values were retained from the earlier manuscript. Do not interpret synthetic smoke results or passing CI as evidence of reproducing Table 1.

## Implementation and environment

The model, strict corpus builder, strict trainer, and original AMTPP metric functions were extracted from the authors' revision workspace. Package imports were reorganized; baseline implementations were removed. Model calculations and strict-protocol training calculations were preserved. Tests use artificial records and verify temporal/user boundaries, causality, normalization, finite gradients, analytic time densities and quantiles, both destination losses, stepwise/sequence prediction agreement, graph propagation, and checkpoint input identity. Nonfinite clipped gradients now stop strict training instead of updating parameters.

Undefined statistics for empty cohorts are serialized as JSON `null`. Because legacy accuracy and NLL average batch means, keep the evaluation batch size fixed; newly trained legacy checkpoints enforce it, together with the mixture count and OD rank.

Locally checked with Python 3.13, PyTorch 2.6.0, NumPy 2.3.3, pandas 2.3.3, and CPU training/evaluation. See `requirements-tested.txt` for the tested Python dependency versions. The requirements describe the checked environment; they are not a guarantee of identical GPU results.
