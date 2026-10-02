# Model calculations

The implementation is in `src/amtpp/models/amtpp.py`. The default model uses causal attention, an asymmetric log-Laplace (ALL) mixture, and a low-rank OD head.

## Causal alignment

The sequence begins with a condition token. Prediction for event `i` uses the encoded state immediately before event `i`: its own time, origin, destination, and calendar labels are not inputs to that prediction. `predict_next` uses the final state of the observed prefix. Daily and weekly embeddings describe observed context trips. The default `raw` time-to-spatial pathway uses predicted time parameters.

## Time density

For a component with log-time location `beta`, positive scale parameter `lambda`, and positive asymmetry `gamma`, define `a = lambda / gamma` and `b = lambda * gamma`. The log-time density is

```
C = a * b / (a + b)
p(y) = C * exp(a * (y - beta))   if y < beta
       C * exp(-b * (y - beta))  otherwise
```

Mixture weights sum to one. With `y = log(tau)` and `tau` in hours, `log p(tau) = log p(y) - log(tau)`. The time NLL includes this Jacobian. The first trip's placeholder interval is excluded from the time loss. Strict point predictions use the mixture median; legacy point predictions use capped Monte Carlo samples as documented in [reproduction notes](reproduction.md).

## Origin and destination

The OD matrix is indexed **destination × origin**. Each valid origin column is normalized over allowed destinations. The marginal destination probability is `p(d) = sum_o q(d | o) p(o)`. PAD origin and PAD destination receive zero probability. The low-rank logits are `D1 @ D2.T`.

The legacy destination objective uses `-log p(d_true)`. The strict objective uses `-log q(d_true | o_true)`. These are different losses; their configurations and reported NLLs must be identified explicitly.

## Network adapter

In `graph` mode, the separate origin and destination embedding tables each receive one residual propagation step:

```
E_graph = E + softplus(strength) * D^(-1/2) @ A @ D^(-1/2) @ E
```

`A` contains aligned undirected adjacent station pairs. The graph adapter neither adds a distance bias to OD logits nor removes destinations by hop distance. Other topology modes and time distribution variants are retained research options; they are not the default AMTPP configuration.

`oracle` time features use true target labels and are an explicit diagnostic, not a deployable next-event predictor. Shuffled time features and permuted/surrogate topology are controls. They must not be presented as ordinary causal forecasts or physical network inputs.
