# ExpGeoNet

`ExpGeoNet` is a script-first reconstruction of the original GotenNet project. It is organized so you can enter the project root and run training directly without installing a package:

```bash
source ~/miniforge3/bin/activate gnn
cd /Users/chenrq/Study/GotenNetSrc/ExpGeoNet
python train.py
```

## Layout

```text
ExpGeoNet/
├── train.py
├── test.py
├── README.md
├── requirements.txt
├── configs/
├── src/
│   ├── common/
│   ├── runtime/
│   ├── data/
│   └── model/
├── docs/
│   └── research/
├── tools/
└── artifacts/
```

## Entry Scripts

- `train.py`: Hydra training entrypoint.
- `test.py`: Hydra evaluation entrypoint.
- `tools/train_matrix.sh`: batch training commands for common experiment sweeps.

Both entry scripts automatically add `src/` to `sys.path`, bootstrap runtime settings, and load configs from `configs/`.

## Configuration

- Root training config: `configs/train.yaml`
- Root evaluation config: `configs/test.yaml`
- Dataset configs: `configs/dataset/`
- Experiment presets: `configs/experiment/`
- Model config: `configs/model/gotennet.yaml`

Examples:

```bash
python train.py experiment=qm9.yaml label=U0
python train.py experiment=molecule3d.yaml label=gap
python test.py checkpoint=QM9_small_homo
```

## Code Organization

- `src/runtime/`: bootstrap, checkpoint download helpers, trainer/logger factories, and train/test runners
- `src/data/`: shared Lightning datamodule, dataset wrappers, split helpers, and transforms
- `src/model/`: Lightning system, representation network, task logic, layer primitives, and prediction heads
- `src/common/`: generic logging and project helpers

## Research References

Reference material is kept outside the source tree:

- `docs/research/13629_GotenNet_Rethinking_Effi.pdf`
- `docs/research/paper_interpretation.md`

## Runtime Artifacts

Hydra runs, logger outputs, config dumps, and temporary logs are written under `artifacts/` so source directories stay uncluttered.
