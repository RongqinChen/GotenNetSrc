# GotenNet Gap Analysis: ExpressiveGeoNet vs. Original Paper vs. Official Implementation

## Summary

ExpressiveGeoNet experiment configurations show **systematic deviations** from the original paper's hyperparameters across all four benchmark datasets (QM9, rMD17, MD22, Molecule3D). These deviations, combined with one architectural addition not present in the paper, explain the inferior experimental performance.

---

## 1. Architectural Differences

### 1.1 SteerableInit Layer (CRITICAL)

| Component | Original Paper | Official gotennet | ExpressiveGeoNet |
|-----------|:---:|:---:|:---:|
| SteerableInit | **NOT present** | **NOT present** | **PRESENT** |

`ExpressiveGeoNet/src/model/representation/encoder.py:80-205` adds a `SteerableInit` class that initializes high-degree steerable features via attention-based message passing. The paper's GotenNet initializes X as zeros directly (`X = torch.zeros(...)`). The official implementation matches the paper: no SteerableInit.

**Impact**: The SteerableInit adds an extra learnable layer with its own query/key/value/attention mechanism before the main GATA blocks. This changes the model's parameter count, gradient flow, and training dynamics. The paper's Eq.4 and description do not mention this component.

### 1.2 Default Parameter Differences in Encoder

| Parameter | Paper | Official gotennet | ExpressiveGeoNet |
|-----------|:---:|:---:|:---:|
| `lmax` (default) | 2 | **1** | 2 |
| `sep_dir` (default) | True | **False** | True |
| `sep_tensor` (default) | True | **False** | True |
| `scale_edge` (default) | — | **True** | **False** |
| `attn_dropout` (default) | 0.0/0.1* | **0.0** | **0.1** |
| `rej` (vector rejection, default) | — | **True** | **False** |
| `n_rbf` (default) | 32/64* | **32** | **64** |
| `residual_scale` | 1.0 (implicit) | 1.0 (implicit) | 1.0 (explicit) |
| `edge_residual_scale` | 1.0 (implicit) | 1.0 (implicit) | 1.0 (explicit) |

*Paper uses 0.1 for QM9, 0.0 for MD tasks; 64 for QM9, 32 for MD tasks.*

### 1.3 EQFF and GATA Residual Scaling

ExpressiveGeoNet adds `residual_scale` parameters to both EQFF and GATA:
```python
# ExpressiveGeoNet EQFF:
h = h + self.residual_scale * m1       # Official: h = h + m1
X = X + self.residual_scale * dX_intra # Official: X = X + dX_intra

# ExpressiveGeoNet GATA:
h = h + self.residual_scale * d_h      # Official: h = h + d_h
X = X + self.residual_scale * d_X      # Official: X = X + d_X
```

When `residual_scale=1.0`, these are equivalent. But ExpressiveGeoNet's `qm9_large` reduces them to 0.5/0.25, which is not a paper-recommended setting.

---

## 2. Training Hyperparameter Deviations

### 2.1 QM9 (GotenNetB)

| Parameter | Paper | ExpressiveGeoNet `qm9.yaml` | Delta |
|-----------|:---:|:---:|:---:|
| `weight_decay` | **0.01** | 0.001 | **10x too small** |
| `gradient_clip_val` | **10.0** | 2.0 | **5x too aggressive** |
| `attn_dropout` | **0.1** | 0.05 | **2x too small** |
| `lr_monitor` | val_loss | MeanAbsoluteError_${label_str} | **different signal** |
| `edge_updates` specified | True | **missing (defaults to True)** | implicit only |
| `evec_dim` | 256 | **not set** | defaults to hidden_dim |
| `emlp_dim` | 256 | **not set** | defaults to hidden_dim |

### 2.2 rMD17

| Parameter | Paper | ExpressiveGeoNet `rmd17.yaml` | Delta |
|-----------|:---:|:---:|:---:|
| `n_interactions` | **12** | 10 | **2 fewer layers** |
| `weight_decay` | **0.01** | 0.001 | **10x too small** |
| `lr_warmup_steps` | **1000** | 1500 | **50% longer** |
| `lr_patience` | **30** | 20 | **33% shorter** |
| `lr_decay` | **0.8** | 0.7 | **more aggressive decay** |
| `gradient_clip_val` | **10.0** | 5.0 | **2x too aggressive** |
| `batch_size` | **4** | 24 | **6x larger** |
| `evec_dim` | **768** | 512 | **33% smaller** |
| `lr_monitor` | val_loss | MAE_force | **different signal** |
| `use_ema` | **True** | **not enabled** | missing |
| `ema_rates` | [0.05, 1.00] | **not set** | missing |

### 2.3 MD22 (GotenNetB)

The `md22.yaml` and `md22_paper.yaml` are relatively close to the paper, but:

| Parameter | Paper | ExpressiveGeoNet `md22.yaml` | Issue |
|-----------|:---:|:---:|:---|
| `attn_dropout` | **0.0** | not set (defaults to **0.1**) | **dropout on MD task** |
| `use_ema` | **True** | **not enabled** | missing EMA loss tracking |
| `emlp_dim` | 768 | **not set** | defaults to hidden_dim |
| `lr` | [4e-5, 1e-4] | 1e-4 (upper bound only) | no per-system tuning |

### 2.4 Molecule3D

| Parameter | Paper | ExpressiveGeoNet `molecule3d.yaml` | Delta |
|-----------|:---:|:---:|:---:|
| `n_interactions` | **12** | 6 | **half the layers** |
| `batch_size` | **256** | 64 | **4x smaller** |
| `gradient_clip_val` | **0.0** (none) | 10.0 | **added clipping** |
| `weight_decay` | 0.0† | 0.01 | **added weight decay** |
| `edge_updates` specified | True | **missing** | implicit default |

†Paper Table 6 does not report weight_decay for Molecule3D; official repo uses 0.0.

---

## 3. Gaps Between ExpressiveGeoNet and Official gotennet Implementation

### 3.1 Key Implementation Differences

| Aspect | Official gotennet | ExpressiveGeoNet |
|--------|:---:|:---:|
| SteerableInit layer | No | **Yes** |
| `lmax` default | 1 | **2** |
| `sep_dir` default | False | **True** |
| `sep_tensor` default | False | **True** |
| `scale_edge` default | True | **False** |
| `attn_dropout` default | 0.0 | **0.1** |
| GATA `rej` (vector rejection) default | True | **False** |
| EQFF residual scale | No (hardcoded 1.0) | **Yes** (configurable) |
| GATA residual scale | No (hardcoded 1.0) | **Yes** (configurable) |
| `edge_residual_scale` | No | **Yes** |
| `n_rbf` default | 32 | **64** |

### 3.2 Official Repo Configs vs Paper

The official gotennet experiment configs also deviate from the paper's Table 6:

| Dataset | Official Config | Paper |
|---------|:---|:---|
| MD22 | n_interactions=**16**, n_atom_basis=**64**, edge_updates="linw_mlpa" | n_interactions=**6**, n_atom_basis=**256**, edge_updates="linw" |
| rMD17 | n_interactions=**16**, edge_updates="linw_mlpa" | n_interactions=**12**, edge_updates="linw" |
| Molecule3D | n_interactions=**4**, n_atom_basis=**256**, batch_size=**32** | n_interactions=**12**, n_atom_basis=**384**, batch_size=**256** |

**Conclusion**: The official gotennet repo configs appear to be development/experimental settings rather than exact paper reproduction configs. This may indicate the official repo was used for hyperparameter exploration beyond the paper's published ranges.

---

## 4. Factors Causing Inferior Performance

### Primary Factors (High Impact)

1. **QM9 `weight_decay` (0.001 vs 0.01)**: 10x lower regularization allows the model to overfit the 110k training set. QM9 molecules are small (≤9 heavy atoms), making regularization critical.

2. **rMD17 `batch_size` (24 vs 4)**: A 6x larger batch size reduces gradient noise and can lead to sharper minima with worse generalization. MD tasks with small molecules benefit from lower batch sizes.

3. **Molecule3D `n_interactions` (6 vs 12)**: Half the model depth means significantly reduced expressiveness for ~3.9M molecules.

4. **rMD17 `n_interactions` (10 vs 12)**: Two fewer interaction blocks reduce the model's capacity to capture complex force fields.

### Secondary Factors (Medium Impact)

5. **`gradient_clip_val` too aggressive**: QM9 uses 2.0 instead of 10.0, rMD17 uses 5.0 instead of 10.0. Overly aggressive clipping can prevent the model from escaping poor local minima.

6. **Missing EMA for MD tasks**: The official implementation uses `use_ema: True` with `ema_rates: [0.05, 1.00]` for loss tracking in MD tasks. Without EMA, the loss signal can be noisy, especially for energy predictions which have 0.05 weight.

7. **`lr_monitor` using wrong metric**: Some configs monitor `MeanAbsoluteError_${label_str}` instead of `val_loss`. With ReduceLROnPlateau, monitoring different signals changes when (or if) the learning rate is reduced.

8. **rMD17 `evec_dim` (512 vs 768)**: Smaller edge vector dimension reduces the capacity of the Hierarchical Tensor Refinement to capture geometric relationships, critical for force prediction.

### Tertiary Factors (Lower Impact)

9. **SteerableInit layer**: Adds parameters and changes initialization dynamics. While not necessarily harmful, it introduces behavior not validated in the paper.

10. **`attn_dropout` mismatches**: QM9 uses 0.05 instead of 0.1 (less regularization); MD22 inherits 0.1 from model default instead of 0.0 (unnecessary dropout).

11. **rMD17 `lr_warmup_steps` (1500 vs 1000)**: 50% longer warmup delays when the model reaches full learning rate.

12. **Molecule3D `gradient_clip_val` (10.0 vs 0.0)**: Adding gradient clipping may slow convergence on this large-scale task.

---

## 5. Fix Summary: `$dataset_original.yaml` Files

The following YAML files have been created with paper-correct hyperparameters:

| File | Dataset | Variant | Key Fixes |
|------|---------|---------|-----------|
| `qm9_original.yaml` | QM9 | GotenNetB (6×256) | weight_decay→0.01, grad_clip→10.0, attn_dropout→0.1, evec_dim/emlp_dim→256 |
| `qm9_small_original.yaml` | QM9 | GotenNetS (4×256) | Extends qm9_original with n_interactions=4 |
| `qm9_large_original.yaml` | QM9 | GotenNetL (12×256) | Extends qm9_original with n_interactions=12, lr→6e-5 |
| `rmd17_original.yaml` | rMD17 | Base (12×192) | n_interactions→12, weight_decay→0.01, batch_size→4, evec_dim→768, warmup→1000, patience→30, ema enabled |
| `md22_original.yaml` | MD22 | GotenNetB (6×256) | ema→True, attn_dropout→0.0, emlp_dim→768 |
| `md22_compact_original.yaml` | MD22 | GotenNetS (4×128) | Extends md22_original |
| `md22_wide_original.yaml` | MD22 | GotenNetL (8×384) | Extends md22_original, cutoff→4.0 |
| `molecule3d_original.yaml` | Molecule3D | Base (12×384) | n_interactions→12, batch_size→256, grad_clip→0.0, weight_decay→0.0 |

### Usage

```bash
# QM9 (GotenNetB)
python train.py experiment=qm9_original.yaml label=mu

# rMD17
python train.py experiment=rmd17_original.yaml label=aspirin

# MD22
python train.py experiment=md22_original.yaml label=buckycatcher

# Molecule3D
python train.py experiment=molecule3d_original.yaml label=homo
```

### Remaining Reproduction Gaps

1. **rMD17 5-fold averaging**: The paper reports mean ± std over 5 predefined splits (0-4). Set `datamodule.hparams.splits: 0` through `4` in 5 separate runs and average.

2. **QM9/MD22 per-target LR tuning**: The paper reports LR ranges ([6e-5, 1e-4] for QM9, [4e-5, 1e-4] for MD22). The provided `_original.yaml` files use conservative defaults. For exact reproduction, sweep each label's LR.

3. **Data split randomness**: QM9 and MD22 use random splits. The paper's exact split may differ, affecting exact numeric reproduction.

4. **SteerableInit removal**: For exact paper architecture match, remove the `SteerableInit` from `encoder.py:GotenNet.forward()` and initialize X as zeros (matching the official implementation).
