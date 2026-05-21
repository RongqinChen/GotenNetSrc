## Dataset Summary: `src/data/datasets/`

All 4 dataset classes plus the shared utilities module.

### Shared Infrastructure (`utils.py`)

| Function                                     | Purpose                                                      | Used By                 |
| -------------------------------------------- | ------------------------------------------------------------ | ----------------------- |
| `parse_csv_selection(value, available, ...)` | Parses `label="dha,buckycatcher"` or `"all"` into validated, deduplicated list | MD22, rMD17             |
| `stream_download(url, path, ...)`            | Single-stream HTTP download with `tqdm` progress bar + empty-file detection | MD22, rMD17, Molecule3D |

---

### 1. QM9 (`qm9.py` — 178 lines) — Single-property regression

| Aspect             | Detail                                                       |
| ------------------ | ------------------------------------------------------------ |
| **Samples**        | ~134k small organic molecules (≤9 heavy atoms)               |
| **Source**         | PyTorch Geometric built-in `QM9` — no external download      |
| **Target**         | `y` — one of 12 scalar properties (mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv) |
| **Splits**         | Random 110k/10k/10k, via `_random_split_dataset()` in `module.py` |
| **Special**        | `get_atomref()` — per-element atomic reference values; `mean()/std()/min()` with `@lru_cache` |
| **Key difference** | Only dataset that has **atomic reference values** (atomrefs) and scalar-only targets |
| **Usage**          | `QM9(root=..., label="U0")` — selects property column, `batch.y` becomes `(N, 1)` |

---

### 2. MD22 (`md22.py` — 180 lines) — Energy + forces (large molecules)

| Aspect      | Detail                                                       |
| ----------- | ------------------------------------------------------------ |
| **Samples** | ~50k–300k conformations per molecule                         |
| **Source**  | Hugging Face parquet shards (`colabfit/MD22_*`)              |
| **Target**  | `y` (energy) + `dy` (forces) — both per conformation         |
| **Splits**  | Random, via `_random_split_dataset()`                        |
| **Special** | Each molecule has its own Hugging Face repo; multi-molecule datasets produce a warning about different reference energies |
| **Usage**   | `MD22(root=..., label="dha")` or `label="dha,buckycatcher"` or `label="all"` |

---

### 3. rMD17 (`rmd17.py` — 147 lines) — Energy + forces (small molecules)

| Aspect      | Detail                                                       |
| ----------- | ------------------------------------------------------------ |
| **Samples** | ~50k–500k conformations per molecule                         |
| **Source**  | figshare tar.bz2 archive → `.npz` files                      |
| **Target**  | `y` (energy) + `dy` (forces)                                 |
| **Splits**  | **Pre-defined** — 5 fold splits (0–4) loaded from CSV files in the archive |
| **Special** | `get_split(idx)` returns train/test index lists; splits are nested (training indices must be further split into train/val) |
| **Usage**   | `rMD17(root=..., label="aspirin")`                           |

---

### 4. Molecule3D (`molecule3d.py` — 281 lines) — Property prediction (~3.9M molecules)

| Aspect         | Detail                                                       |
| -------------- | ------------------------------------------------------------ |
| **Samples**    | ~3.9 million molecules — largest dataset by far              |
| **Source**     | Hugging Face parquet shards (`maomlab/Molecule3D`)           |
| **Target**     | `y` — one of 7 DFT properties (homo, lumo, gap, scf_energy, dipole_x/y/z) |
| **Splits**     | **Pre-defined** — `"Molecule3D_random_split"` or `"Molecule3D_scaffold_split"`; split boundaries recorded during processing |
| **Special**    | SDF strings parsed via RDKit; `get_split()` returns train/val/test index lists; `get_atomref()` returns `None` (not available) |
| **Processing** | Largest bottleneck — ~3.9M SDF → RDKit parsings              |
| **Usage**      | `Molecule3D(root=..., label="homo", split_config="Molecule3D_random_split")` |

---

### Comparison Table

| Feature                | QM9                          | MD22                | rMD17                                    | Molecule3D                                        |
| ---------------------- | ---------------------------- | ------------------- | ---------------------------------------- | ------------------------------------------------- |
| **Size**               | ~134k                        | ~50k–300k/mol       | ~50k–500k/mol                            | ~3.9M                                             |
| **Target type**        | Scalar only                  | Energy + Forces     | Energy + Forces                          | Scalar only                                       |
| **Data format**        | PyG built-in                 | Parquet (HF)        | NPZ (figshare)                           | Parquet (HF) → RDKit                              |
| **Molecule selection** | Property column              | Molecule name(s)    | Molecule name(s)                         | Property name                                     |
| **Splits**             | Random                       | Random              | Pre-defined CSV (5 folds)                | Pre-defined (random/scaffold)                     |
| **Atomrefs**           | ✅ Yes                        | ❌ No                | ❌ No                                     | ❌ No                                              |
| **`mean/std/min`**     | ✅ (cached)                   | ❌ removed           | ❌ removed                                | ❌ removed                                         |
| **Download mechanism** | None needed                  | `requests` + HF API | `requests` + tar.bz2                     | `requests` + HF API                               |
| **Notable complexity** | Property filtering transform | Per-molecule repos  | Multi-threaded ranged download (removed) | SDF header normalization, split boundary tracking |
