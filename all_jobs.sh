source ~/miniforge3/bin/activate gnn270
export CUDA_VISIBLE_DEVICES=0

python train.py experiment=qm9.yaml label=mu    logger=csv.yaml
python train.py experiment=qm9.yaml label=alpha logger=csv.yaml
python train.py experiment=qm9.yaml label=homo  logger=csv.yaml
python train.py experiment=qm9.yaml label=lumo  logger=csv.yaml
python train.py experiment=qm9.yaml label=gap   logger=csv.yaml
python train.py experiment=qm9.yaml label=r2    logger=csv.yaml

python train.py experiment=qm9.yaml label=Cv    logger=csv.yaml
python train.py experiment=qm9.yaml label=G     logger=csv.yaml
python train.py experiment=qm9.yaml label=H     logger=csv.yaml
python train.py experiment=qm9.yaml label=U     logger=csv.yaml
python train.py experiment=qm9.yaml label=U0    logger=csv.yaml
python train.py experiment=qm9.yaml label=zpve  logger=csv.yaml
