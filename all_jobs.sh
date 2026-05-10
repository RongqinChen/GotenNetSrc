export CUDA_VISIBLE_DEVICES=0

# python train.py experiment=qm9.yaml label=mu
# python train.py experiment=qm9.yaml label=alpha
# python train.py experiment=qm9.yaml label=homo
# python train.py experiment=qm9.yaml label=lumo
# python train.py experiment=qm9.yaml label=gap
# python train.py experiment=qm9.yaml label=r2

# python train.py experiment=qm9.yaml label=Cv
# python train.py experiment=qm9.yaml label=G
python train.py experiment=qm9.yaml label=H
# python train.py experiment=qm9.yaml label=U
# python train.py experiment=qm9.yaml label=U0
# python train.py experiment=qm9.yaml label=zpve
