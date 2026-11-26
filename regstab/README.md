# regstab

Regionally certified dynamic recurrent networks (DRN), trained without constraints through a
direct parametrization. This is the lean rewrite of the training code in `../python`, which stays
as-is because the results of the submitted paper come from it.

## Model and certificate

```
x_{k+1} = A x_k + B u_k + B2 w_k,    y_k = C x_k,    w_k = dzn(C2 x_k + D21 u_k)
```

`(A, B, B2, C2, D21) = Γ(θ̂)` satisfies the conditions of Theorem 1 (with X = I) for every value
of the free parameters θ̂, so training is unconstrained: no barrier, feasibility repair, rollback
or SDP. The certificate is read off the parameters:

- the unit ball is forward invariant for inputs `‖u_k‖² ≤ s²(1 − α²‖x_k‖²)` (admissible input map U),
- outputs stay in `Y = {Cx : ‖x‖ ≤ 1} = E((CCᵀ)⁻¹)`.

`α = sigmoid(α̂)` and `s = exp(ŝ)` are trainable (or fixed via `trainable: false`).

| `model.certificate` | Γ | paper |
|---|---|---|
| `regional` | H = z/√(1+‖z‖²) | RegStable (ours) |
| `global` | H = 0 (standard sector condition) | GlobStable |
| `none` | free matrices | NotStable |

`model.type: lstm` is the LSTM baseline.

## Objective (Pareto scalarization)

```
min_θ̂  L_conv + w_div · L_div + ν₁ pen(size_U) + ν₂ pen(size_Y)
```

- `L_conv`: MSE on converging trajectories after `data.washout` (x0 unknown, model starts at 0).
- `L_div`: loss on diverging trajectories (`<split>_div/`, x0 = 0, no washout). `data.diverging_loss: relative` (default) is Σ‖ŷ−y‖²/Σ‖y‖² per trajectory, averaged, so every trajectory counts equally however large it grows; `mse` is the plain MSE, which on Duffing swamps the converging fit.
- `regularization.input.measure`: `s` | `sigma` (= s√(1−α²)) | `mean_radius` (admissible radius averaged over B₁).
- `regularization.output.measure`: `trace` (‖C‖_F) | `min_eig` (√λ_min(CCᵀ)) | `det` (det(CCᵀ)^{1/2p}).
- `regularization.form`: `inverse_square` (1/size², for `s` this is ŝ = 1/s²) | `neg_log`.

All sizes are lengths in normalized signal units, so they compare directly with the data
(`s_over_umax`, `ybar_over_ymax` are logged). With `certificate: global`, the state can be
rescaled without changing the input-output map, so s and C can grow without bound; keep ν = 0 there.

## Initialization (echo state network)

`init.restarts` random reservoirs θ̂ (entries ~ N(0, scale²), scale drawn log-uniformly from
`init.scale`), each rolled out on the training inputs; C is fitted by ridge regression on the
states after the washout. The reservoir with the lowest validation error is kept, then all
parameters are trained with Adam.

## Usage

```bash
pip install -e .                       # into the project venv
python scripts/train.py configs/duffing.yaml
python scripts/train.py configs/duffing.yaml --set model.nx=4 model.nz=40 regularization.input.weight=1e-3
pytest
```

Each run writes `model.pt`, `config.yaml` and `results.json` (test metrics, ESN restarts, per-epoch
history) to `output_dir/<run name>/` and logs the same to MLflow. Unknown config keys are errors.

## Layout

| file | content |
|---|---|
| `parametrization.py` | Cayley transform, Γ, numerical check of Theorem 1 |
| `model.py` | DRN (regional / global / none), LSTM |
| `measures.py` | sizes of U and Y, penalties |
| `esn.py` | ESN initialization with ridge readout |
| `data.py` | CSV folders, normalization, windows, padding, masked MSE |
| `trainer.py` | training loop, evaluation, coverage metrics, run pipeline |
| `config.py` | dataclass config with strict keys |
