#!/usr/bin/env bash

source ~/miniforge3/bin/activate gnn270
export CUDA_VISIBLE_DEVICES=0

Running
python train.py experiment=qm9_small logger=wandb label=mu        3090x2
python train.py experiment=qm9_small logger=wandb label=alpha     3090x2
python train.py experiment=qm9_small logger=wandb label=homo      3090x2
python train.py experiment=qm9_small logger=wandb label=lumo      3090x2
python train.py experiment=qm9_small logger=wandb label=gap       3090x2
python train.py experiment=qm9_small logger=wandb label=r2        3090x2
python train.py experiment=qm9_small logger=wandb label=Cv        3090x2
python train.py experiment=qm9_small logger=wandb label=G         3090x2
python train.py experiment=qm9_small logger=wandb label=H         3090x2
python train.py experiment=qm9_small logger=wandb label=U         3090x2
python train.py experiment=qm9_small logger=wandb label=U0        3090x2
python train.py experiment=qm9_small logger=wandb label=zpve      3090x2

Running
python train.py experiment=qm9 logger=wandb label=mu              4090x2-2
python train.py experiment=qm9 logger=wandb label=alpha           4090x2-2
python train.py experiment=qm9 logger=wandb label=homo            4090x2-2
python train.py experiment=qm9 logger=wandb label=lumo            4090x2-2
python train.py experiment=qm9 logger=wandb label=gap             4090x2-2
python train.py experiment=qm9 logger=wandb label=r2              4090x2-2
python train.py experiment=qm9 logger=wandb label=Cv              4090x2-2
python train.py experiment=qm9 logger=wandb label=G               4090x2-2
python train.py experiment=qm9 logger=wandb label=H               4090x2-2
python train.py experiment=qm9 logger=wandb label=U               4090x2-2
python train.py experiment=qm9 logger=wandb label=U0              4090x2-2
python train.py experiment=qm9 logger=wandb label=zpve            4090x2-2

Running
python train.py experiment=qm9_large logger=wandb label=mu        4090x2
python train.py experiment=qm9_large logger=wandb label=alpha     4090x2
python train.py experiment=qm9_large logger=wandb label=homo      4090x2
python train.py experiment=qm9_large logger=wandb label=lumo      4090x2
python train.py experiment=qm9_large logger=wandb label=gap       4090x2
python train.py experiment=qm9_large logger=wandb label=r2        4090x2
python train.py experiment=qm9_large logger=wandb label=Cv        4090x2
python train.py experiment=qm9_large logger=wandb label=G         4090x2
python train.py experiment=qm9_large logger=wandb label=H         4090x2
python train.py experiment=qm9_large logger=wandb label=U         4090x2
python train.py experiment=qm9_large logger=wandb label=U0        4090x2
python train.py experiment=qm9_large logger=wandb label=zpve      4090x2


Running
python train.py experiment=rmd17 logger=wandb label=aspirin           3090x1
python train.py experiment=rmd17 logger=wandb label=azobenzene        3090x1 
python train.py experiment=rmd17 logger=wandb label=benzene           3090x1
python train.py experiment=rmd17 logger=wandb label=ethanol           3090x1 
python train.py experiment=rmd17 logger=wandb label=malonaldehyde     3090x1 
python train.py experiment=rmd17 logger=wandb label=naphthalene       3090x1  
python train.py experiment=rmd17 logger=wandb label=paracetamol       3090x1 
python train.py experiment=rmd17 logger=wandb label=salicylic         3090x1
python train.py experiment=rmd17 logger=wandb label=toluene           3090x1
python train.py experiment=rmd17 logger=wandb label=uracil            3090x1


Running
python train.py experiment=md22 logger=wandb label=at_at                      4090x1
python train.py experiment=md22 logger=wandb label=at_at_cg_cg                4090x1
python train.py experiment=md22 logger=wandb label=ac_ala3_nhme               4090x1
python train.py experiment=md22 logger=wandb label=dha                        4090x1 
python train.py experiment=md22 logger=wandb label=buckycatcher               4090x1
python train.py experiment=md22 logger=wandb label=double_walled_nanotube     4090x1
python train.py experiment=md22 logger=wandb label=stachyose                  4090x1


python train.py experiment=md22_wide logger=wandb label=at_at                         l20x2
python train.py experiment=md22_wide logger=wandb label=at_at_cg_cg                   l20x2
python train.py experiment=md22_wide logger=wandb label=ac_ala3_nhme                  l20x2
python train.py experiment=md22_wide logger=wandb label=dha                           l20x2
python train.py experiment=md22_wide logger=wandb label=buckycatcher                  l20x2
python train.py experiment=md22_wide logger=wandb label=double_walled_nanotube        l20x2
python train.py experiment=md22_wide logger=wandb label=stachyose                     l20x2


python train.py experiment=md22_compact logger=wandb label=at_at                      l20x2
python train.py experiment=md22_compact logger=wandb label=at_at_cg_cg                l20x2
python train.py experiment=md22_compact logger=wandb label=ac_ala3_nhme               l20x2
python train.py experiment=md22_compact logger=wandb label=dha                        l20x2
python train.py experiment=md22_compact logger=wandb label=buckycatcher               l20x2
python train.py experiment=md22_compact logger=wandb label=double_walled_nanotube     l20x2
python train.py experiment=md22_compact logger=wandb label=stachyose                  l20x2


python train.py experiment=molecule3d logger=wandb label=homo                         l20x2
python train.py experiment=molecule3d logger=wandb label=lumo                         l20x2
python train.py experiment=molecule3d logger=wandb label=gap                          l20x2
python train.py experiment=molecule3d logger=wandb label=scf_energy                   l20x2
python train.py experiment=molecule3d logger=wandb label=dipole_x                     l20x2
python train.py experiment=molecule3d logger=wandb label=dipole_y                     l20x2
python train.py experiment=molecule3d logger=wandb label=dipole_z                     l20x2
