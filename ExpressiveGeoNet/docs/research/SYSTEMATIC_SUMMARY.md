# ExpGeoNet — Systematic Codebase Summary

## 1. Project Overview

**ExpGeoNet** is a reproduction/reconstruction of **GotenNet (Geometric Tensor Network)**, a novel E(3)-equivariant graph neural network architecture for molecular property prediction. The project is organized as a "script-first" layout — you can enter the project root and run training directly without installing a package.

- **License:** MIT (Copyright 2025 Sarp Aykent)
- **Paper reference:** `docs/research/13629_GotenNet_Rethinking_Effi.pdf`
- **Paper interpretation:** `docs/research/paper_interpretation.md`

---

## 2. Project Layout

```
ExpGeoNet/
├── train.py                        # Hydra training entrypoint
├── test.py                         # Hydra evaluation entrypoint
├── requirements.txt                # Dependencies
├── configs/                        # Hydra configuration system
│   ├── train.yaml                  #   Root training config
│   ├── test.yaml                   #   Root evaluation config
│   ├── model/gotennet.yaml         #   Model hyperparameters
│   ├── dataset/                    #   Dataset configs (qm9, md22, rmd17, molecule3d)
│   ├── experiment/                 #   Experiment presets
│   ├── callbacks/default.yaml      #   Lightning callbacks
│   ├── trainer/default.yaml        #   Trainer config
│   ├── logger/                     #   Logger configs (wandb, csv, tensorboard)
│   └── hydra/default.yaml          #   Hydra job orchestration
├── src/                            # Source code
│   ├── common/                     #   Shared utilities
│   ├── data/                       #   Dataset and datamodule code
│   ├── model/                      #   Core model components
│   ├── runtime/                    #   Training/evaluation orchestration
├── scripts/train_matrix.sh         # Batch training commands
└── docs/                           # Research materials
```

---

## 3. Architecture Overview

### 3.1 High-Level Pipeline

```
Input (atomic numbers, positions)
  ↓
  ┌─ GotenNetWrapper ─────────────────────────────────────┐
  │  1. Distance module: radius_graph → edge_index, r_ij  │
  │  2. GotenNet.forward():                               │
  │     a. NodeInit (Eq.1-2): initial scalar node features│
  │     b. EdgeInit (Eq.3): initial edge features         │
  │     c. SteerableInit (Eq.4): high-degree steerable X  │
  │     d. N× [GATA → EQFF] interaction blocks:           │
  │        - GATA: message passing + attention + edge upd │
  │        - EQFF: equivariant feed-forward mixing         │
  └───────────────────────────────────────────────────────┘
  ↓
  ┌─ Task Handler (QM9 | MD22 | rMD17 | Molecule3D) ─────┐
  │  - Output heads: Atomwise, AtomwiseV3, Dipole, etc.  │
  │  - Losses: L1Loss, MSELoss, EMA-smoothed losses       │
  │  - Metrics: MAE, MSE                                  │
  └───────────────────────────────────────────────────────┘
  ↓
Output: scalar property (energy, HOMO, etc.) ± forces
```

### 3.2 Core Architectural Components (GotenNet Paper)

The model implements four principal components as described in the paper:

| Component | Module | Function |
|-----------|--------|----------|
| **Unified Structural Embedding** | `NodeInit` + `EdgeInit` | Initializes node & edge representations from atomic numbers and distances using radial basis functions |
| **Steerable Initialization** | `SteerableInit` | Initializes high-degree steerable (tensor) features via attention-based message passing |
| **Geometry-Aware Tensor Attention (GATA)** | `GATA` | Degree-wise attention message passing with spatial filtering; updates both scalar (`h`) and steerable (`X`) node features, plus edge features |
| **Hierarchical Tensor Refinement (HTR)** | `GATA.edge_update()` | Updates edge features using inner products of steerable node features across degrees |
| **Equivariant Feed-Forward (EQFF)** | `EQFF` | Intra-atomic mixing of scalar and steerable features while preserving equivariance |

### 3.3 Key Design Principles

- **Spherical scalarization paradigm:** Replaces expensive Clebsch-Gordan (CG) tensor products with inner-product-based geometric tensor operations
- **E(3) equivariance:** Strict rotation, translation, and reflection equivariance guaranteed by construction
- **Degree-specific processing:** Spherical harmonic components for each angular momentum (l=1..lmax) are processed separately with `sep_dir`, `sep_tensor`, `sep_htr` flags

---

## 4. Source Code Details

### 4.1 `src/model/` — Core Model

| File | Contents |
|------|----------|
| `system.py` | **`GotenModel`** — PyTorch Lightning `LightningModule` that orchestrates representation encoder + task handler + optimizers |
| `representation/encoder.py` | **`GotenNet`** (main encoder), **`GotenNetWrapper`** (adds Distance layer), **`SteerableInit`**, **`GATA`**, **`EQFF`** |
| `representation/checkpoint.py` | `load_representation_from_checkpoint()` — loads representation from Lightning checkpoints |
| `layers/graph.py` | **`Distance`** (radius graph), **`NodeInit`** (Eq.1-2), **`EdgeInit`** (Eq.3), **`TensorLayerNorm`** (degree-wise normalization) |
| `layers/basis.py` | **`BesselBasis`**, **`GaussianRBF`**, **`ExpNormalSmearing`** — radial basis function expansions |
| `layers/cutoff.py` | **`CosineCutoff`**, **`PolynomialCutoff`**, **`safe_norm`** |
| `layers/common.py` | **`Dense`** (linear + norm + activation), **`MLP`**, **`SchnetMLP`**, **`ScaleShift`**, **`GetItem`**, weight init helpers |
| `layers/activations.py` | **`ShiftedSoftplus`**, **`Swish`**, `str2act()` |
| `heads/atomwise.py` | **`Atomwise`** (scalar property), **`AtomwiseV3`** (energy + force derivatives) |
| `heads/molecular.py` | **`Dipole`** (equivariant dipole moment), **`ElectronicSpatialExtentV2`** (R²) |
| `heads/blocks.py` | **`GatedEquivariantBlock`** — gated equivariant block for tensor properties |
| `tasks/base.py` | **`Task`** base class |
| `tasks/qm9.py` | **`QM9Task`** — 12 quantum chemical properties |
| `tasks/md.py` | **`MDTask`** — energy + force prediction for MD trajectories |
| `tasks/molecule3d.py` | **`Molecule3DTask`** — HOMO, LUMO, gap, SCF energy, dipole |

### 4.2 `src/data/` — Dataset & Data Loading

| File | Contents |
|------|----------|
| `module.py` | **`DataModule`** — PyTorch Lightning `LightningDataModule` unifying all datasets. Handles loading, splitting (random & pre-defined), standardization, and DataLoader creation |
| `splits.py` | `train_val_test_split()`, `make_splits()` — train/val/test splitting with support for fractional sizes, saving/loading `.npz` splits |
| `transforms.py` | `normalize_positions()` — center positions by center of mass or centroid |
| `datasets/qm9.py` | **`QM9`** — wraps PyTorch Geometric's QM9 dataset with property filtering (12 targets: mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv) |
| `datasets/md22.py` | **`MD22`** — 7 molecules from Hugging Face parquet shards with energy + forces |
| `datasets/rmd17.py` | **`rMD17`** — 10 small molecules from figshare `.npz` archive with energy + forces, 5 predefined split configurations |
| `datasets/molecule3d.py` | **`Molecule3D`** — ~3.9M molecules from Hugging Face parquet shards with 7 DFT properties, random & scaffold splits |
| `datasets/utils.py` | Shared download utilities: `stream_download()`, `parse_csv_selection()`, `is_manifest_complete()` |

### 4.3 `src/runtime/` — Orchestration

| File | Contents |
|------|----------|
| `bootstrap.py` | `bootstrap_entrypoint()` — configures PyTorch load compatibility, loads `.env`, enables TF32 |
| `runner.py` | `train()` and `test()` — orchestrates datamodule/model instantiation, trainer setup, and execution |
| `factories.py` | `build_trainer_config()`, `build_callbacks()`, `build_loggers()`, `populate_run_metadata()`, `resolve_label()` |
| `ui.py` | `task_wrapper()` decorator, `print_config()`, `extras()`, `get_metric_value()` |
| `checkpoints.py` | `download_checkpoint()` / `download_file()` — downloads pre-trained models from Hugging Face |

### 4.4 `src/common/` — Shared Utilities

| File | Contents |
|------|----------|
| `logging.py` | `get_logger()` (rank-zero-only), `log_hyperparameters()` |
| `project.py` | `find_config_directory()`, `get_function_name()`, `humanbytes()` |

---

## 5. Training & Evaluation Pipeline

### 5.1 Entry Points

- **`train.py`**: Hydra entrypoint — `python train.py experiment=qm9.yaml label=mu`
- **`test.py`**: Hydra evaluation entrypoint — `python test.py checkpoint=QM9_small_homo`

### 5.2 Training Flow (in `runner.train()`)

1. **Bootstrap** (TF32, env)
2. **Clone config** for mutable runtime adjustments
3. **Configure matmul precision**
4. **Seed RNGs**
5. **Instantiate DataModule** from `cfg.datamodule._target_`
6. **Populate run metadata** (resolve label → label_index)
7. **Get dataset metadata** (mean, std, atomref)
8. **Instantiate GotenModel** from `cfg.model._target_` with `dataset_meta`
9. **Build callbacks & loggers** from config
10. **Build trainer config** (with MPS fallback)
11. **Instantiate Trainer**
12. **Log hyperparameters**
13. **Train** (`trainer.fit()`) — optionally test at end

### 5.3 GotenModel Lifecycle

- **`__init__`**: Configures representation encoder, task handler, metrics, output heads, loss functions, EMA tracking
- **`configure_optimizers`**: AdamW optimizer + CosineAnnealingLR or ReduceLROnPlateau scheduler
- **`optimizer_step`**: Linear LR warmup support
- **`forward`**: Full pipeline with gradient tracking enabled → `_forward_pipeline(batch)` → `_enable_grads_on_positions` → `_encode` → `_apply_output_heads`
- **Loss**: Weighted combination of individual losses, optional EMA smoothing per loss component

---

## 6. Configuration System

Uses **Hydra** with a hierarchical config structure:

| Config Group | Files |
|-------------|-------|
| Root | `train.yaml`, `test.yaml` |
| Model | `model/gotennet.yaml` |
| Dataset | `dataset/{qm9,md22,rmd17,molecule3d}.yaml` |
| Experiment | `experiment/{qm9,qm9_small,qm9_large,md22,md22_wide,md22_compact,rmd17,molecule3d}.yaml` |
| Callbacks | `callbacks/default.yaml` |
| Logger | `logger/{default,wandb,csv,tensorboard,mlflow}.yaml` |
| Trainer | `trainer/default.yaml` |
| Paths | `paths/default.yaml` |
| Hydra | `hydra/default.yaml` |

---

## 7. Supported Datasets

| Dataset | Size | Properties | Labels (examples) |
|---------|------|-----------|-------------------|
| **QM9** | ~133k molecules | 12 scalar properties | mu, alpha, homo, lumo, gap, r2, U0, Cv |
| **rMD17** | 10 molecules × ~150k conformers | energy + forces | aspirin, benzene, ethanol, uracil |
| **MD22** | 7 molecules (up to 370 atoms) | energy + forces | buckycatcher, stachyose, dha |
| **Molecule3D** | ~3.9M molecules | 7 DFT properties | homo, lumo, gap, scf_energy, dipole_xyz |

---

## 8. Key Dependencies

- **PyTorch Geometric** (`torch_geometric`) — graph operations, `Data` objects
- **PyTorch Lightning** (`pytorch_lightning`) — training framework
- **e3nn** — spherical harmonics, SO(3) operations
- **Hydra** — configuration management
- **RDKit** (2025.3.6) — molecule parsing (Molecule3D SDF)
- **pyarrow** — parquet file reading (MD22, Molecule3D)
- **torch_scatter** — scatter operations
- **torch_cluster** — `radius_graph` for neighbor finding
- **Weights & Biases** — experiment logging (optional)
- **ASE** — atomic masses (for electronic spatial extent)

---

## 9. GotenNet Configurations

| Variant | Interactions | Hidden Dim | Limitation |
|---------|:----------:|:----------:|:----------:|
| QM9 Small | 4 | 256 | 6.1M params |
| QM9 Base (default) | 6 | 256 | 8.0M params |
| QM9 Large | 12 | 256 | 12.9M params |
| MD22 Compact | 4 | 128 | — |
| MD22 Base | 6 | 256 | — |
| MD22 Wide | 8 | 384 | — |
| rMD17 | 12 | 192 | — |
| Molecule3D | 12 | 384 | — |

---

## 10. Reproducibility Analysis — Paper Experiments

The paper reports experiments across four benchmarks. Below is an assessment of how well the current code can reproduce each one.

### 10.1 QM9 (GotenNetS, GotenNetB, GotenNetL)

| Aspect | Paper Requirement | Code Status |
|--------|-------------------|-------------|
| **Variants** | S (4 layers), B (6), L (12) | ✅ `qm9_small.yaml`, `qm9.yaml`, `qm9_large.yaml` |
| **All 12 targets** | mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv | ✅ `label=<property>` |
| **Training budget** | 1000 epochs | ✅ `max_epochs: 1000` |
| **Loss function** | MSE | ✅ `task_loss: "MSELoss"` |
| **LR range** | [6e-5, 1e-4] (per-target) | ⚠️ Config uses upper bound 1e-4 only; per-target LR not scripted |
| **Weight decay** | 0.01 | ✅ |
| **Warmup steps** | 10,000 | ✅ |
| **Gradient clip** | 10.0 | ✅ |
| **Batch size** | 32 | ✅ |
| **Edge updates** | `true` (boolean) | ✅ Default config uses `edge_updates: True` |
| **Data split** | Random 110k/10k/remainder | ✅ Reproducible via seed |
| **All targets automated** | — | ✅ `scripts/train_matrix.sh` sweeps all 12 |

**Verdict:** ✅ Fully reproducible. Per-target LR fine-tuning is left to the user.

### 10.2 rMD17 (Energy + Forces)

| Aspect | Paper Requirement | Code Status |
|--------|-------------------|-------------|
| **Model depth** | 12 interactions | ✅ `n_interactions: 12` |
| **Hidden dim** | 192 | ✅ `n_atom_basis: 192` |
| **Training budget** | 3000 epochs | ✅ `max_epochs: 3000` |
| **Loss function** | MSE | ✅ `task_loss: "MSELoss"` |
| **Loss weights** | 0.05 (energy), 0.95 (forces) | ✅ `loss_weights: [0.05, 0.95]` |
| **Learning rate** | 2e-4 | ✅ |
| **Warmup** | 1000 steps | ✅ |
| **Edge updates** | `"linw"` | ✅ |
| **All 10 molecules** | — | ✅ `label=<molecule>` in `train_matrix.sh` |
| **5-fold cross-splits** | Report average over splits 0-4 | ⚠️ Config uses `splits: 0` only; multi-split averaging not scripted |
| **Batch size** | 4 | ✅ |

**Verdict:** ⚠️ Partial. All 10 molecules are covered, but paper results average over 5 predefined splits (0-4). The current config only uses split 0 by default. A user would need to run 5× per molecule and average to match the paper exactly.

### 10.3 MD22 (Energy + Forces)

| Aspect | Paper Requirement | Code Status |
|--------|-------------------|-------------|
| **Variants** | Compact (4×128), Base (6×256), Wide (8×384) | ✅ `md22_compact.yaml`, `md22.yaml`, `md22_wide.yaml` |
| **Training budget** | 3000 epochs | ✅ |
| **Loss function** | MSE | ✅ |
| **Loss weights** | 0.05 (energy), 0.95 (forces) | ✅ |
| **LR range** | [4e-5, 1e-4] (per-system) | ⚠️ Config uses upper bound 1e-4 |
| **Warmup** | 1000 steps | ✅ |
| **Edge updates** | `"linw"` | ✅ |
| **All 7 systems** | — | ✅ `label=at_at, ...` in `train_matrix.sh` |
| **Batch size** | 4 | ✅ |
| **Cutoff** | 5.0 (Base), 4.0 (Wide) | ✅ |

**Verdict:** ⚠️ Partial. Same per-system LR tuning issue as QM9; otherwise fully covered.

### 10.4 Molecule3D

| Aspect | Paper Requirement | Code Status |
|--------|-------------------|-------------|
| **Model depth** | 12 interactions | ✅ |
| **Hidden dim** | 384 | ✅ |
| **Training budget** | 300 epochs | ✅ `max_epochs: 300` |
| **Loss function** | L1 | ✅ `task_loss: "L1Loss"` |
| **Learning rate** | 1e-4 | ✅ |
| **Warmup** | 5000 steps | ✅ |
| **All 7 properties** | homo, lumo, gap, scf_energy, dipole_xyz | ✅ `scripts/train_matrix.sh` covers all |
| **Batch size** | 256 | ✅ |
| **Split** | Random & Scaffold | ✅ Both available via `split_config` |

**Verdict:** ✅ Fully reproducible.

### 10.5 Summary of Reproducibility

| Benchmark | Completeness | Notes |
|-----------|:-----------:|-------|
| **QM9** | ✅ Complete | All 12 targets × 3 variants scripted |
| **rMD17** | ⚠️ Partial | 5-fold averaging missing; single-split per molecule |
| **MD22** | ⚠️ Partial | Per-system LR not tuned per paper range |
| **Molecule3D** | ✅ Complete | All 7 properties × 2 splits |

**Overall Assessment:** The codebase faithfully implements all architectural components, training procedures, and evaluation protocols described in the GotenNet paper. The main gaps in exact reproduction are:
1. **rMD17**: The paper reports averages over 5 predefined data splits (indices 0-4), but the config defaults to split 0 only.
2. **Learning rate ranges**: QM9 and MD22 use per-target/per-system LR ranges reported in the paper, but the provided configs use the upper bound of each range for simplicity.
3. **Random splits vs. paper splits**: QM9 uses randomized splits (reproducible via seed), which may differ from the exact split used in the paper.

These are **configuration gaps**, not implementation gaps. The architectural implementation is complete and correct.

---

## 11. Notable Design Decisions


1. **EMA loss smoothing**: Individual loss components can be EMA-smoothed (configurable `ema_rate` per loss), used primarily for MD tasks where energy and force losses have different noise characteristics.
2. **Degree-specific configurations**: The `sep_dir`, `sep_tensor`, `sep_htr` flags control whether spherical harmonic components are processed separately or combined, offering a tunable expressiveness-efficiency trade-off.
3. **Edge updates**: Configurable through the `edge_updates` string parameter (supports `"gated"`, `"mlp"`, `"linw"`, `"rej"` etc.).
4. **TF32 acceleration**: Enabled by default for CUDA.
5. **MPS fallback**: Apple Silicon users are automatically switched to CPU because `torch_cluster.radius_graph` doesn't support MPS.
6. **Checkpoint download**: Pre-trained models hosted on Hugging Face, downloaded by name convention `{task}_{size}_{label}`.
