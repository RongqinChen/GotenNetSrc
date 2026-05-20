#!/usr/bin/env bash

source ~/miniforge3/bin/activate gnn270
export CUDA_VISIBLE_DEVICES=0

Running
python train.py experiment=qm9_small logger=tensorboard label=mu        3090x2
python train.py experiment=qm9_small logger=tensorboard label=alpha     3090x2
python train.py experiment=qm9_small logger=tensorboard label=homo      3090x2
python train.py experiment=qm9_small logger=tensorboard label=lumo      3090x2
python train.py experiment=qm9_small logger=tensorboard label=gap       3090x2
python train.py experiment=qm9_small logger=tensorboard label=r2        3090x2
python train.py experiment=qm9_small logger=tensorboard label=Cv        3090x2
python train.py experiment=qm9_small logger=tensorboard label=G         3090x2
python train.py experiment=qm9_small logger=tensorboard label=H         3090x2
python train.py experiment=qm9_small logger=tensorboard label=U         3090x2
python train.py experiment=qm9_small logger=tensorboard label=U0        3090x2
python train.py experiment=qm9_small logger=tensorboard label=zpve      3090x2

Running
python train.py experiment=qm9 logger=tensorboard label=mu              4090x2-2
python train.py experiment=qm9 logger=tensorboard label=alpha           4090x2-2
python train.py experiment=qm9 logger=tensorboard label=homo            4090x2-2
python train.py experiment=qm9 logger=tensorboard label=lumo            4090x2-2
python train.py experiment=qm9 logger=tensorboard label=gap             4090x2-2
python train.py experiment=qm9 logger=tensorboard label=r2              4090x2-2
python train.py experiment=qm9 logger=tensorboard label=Cv              4090x2-2
python train.py experiment=qm9 logger=tensorboard label=G               4090x2-2
python train.py experiment=qm9 logger=tensorboard label=H               4090x2-2
python train.py experiment=qm9 logger=tensorboard label=U               4090x2-2
python train.py experiment=qm9 logger=tensorboard label=U0              4090x2-2
python train.py experiment=qm9 logger=tensorboard label=zpve            4090x2-2

Running
python train.py experiment=qm9_large logger=tensorboard label=mu        4090x2
python train.py experiment=qm9_large logger=tensorboard label=alpha     4090x2
python train.py experiment=qm9_large logger=tensorboard label=homo      4090x2
python train.py experiment=qm9_large logger=tensorboard label=lumo      4090x2
python train.py experiment=qm9_large logger=tensorboard label=gap       4090x2
python train.py experiment=qm9_large logger=tensorboard label=r2        4090x2
python train.py experiment=qm9_large logger=tensorboard label=Cv        4090x2
python train.py experiment=qm9_large logger=tensorboard label=G         4090x2
python train.py experiment=qm9_large logger=tensorboard label=H         4090x2
python train.py experiment=qm9_large logger=tensorboard label=U         4090x2
python train.py experiment=qm9_large logger=tensorboard label=U0        4090x2
python train.py experiment=qm9_large logger=tensorboard label=zpve      4090x2


Running
python train.py experiment=rmd17 logger=tensorboard label=aspirin           3090x1
python train.py experiment=rmd17 logger=tensorboard label=azobenzene        3090x1 
python train.py experiment=rmd17 logger=tensorboard label=benzene           3090x1
python train.py experiment=rmd17 logger=tensorboard label=ethanol           3090x1 
python train.py experiment=rmd17 logger=tensorboard label=malonaldehyde     3090x1 
python train.py experiment=rmd17 logger=tensorboard label=naphthalene       3090x1  
python train.py experiment=rmd17 logger=tensorboard label=paracetamol       3090x1 
python train.py experiment=rmd17 logger=tensorboard label=salicylic         3090x1
python train.py experiment=rmd17 logger=tensorboard label=toluene           3090x1
python train.py experiment=rmd17 logger=tensorboard label=uracil            3090x1


Running
python train.py experiment=md22 logger=tensorboard label=at_at                      4090x1
python train.py experiment=md22 logger=tensorboard label=at_at_cg_cg                4090x1
python train.py experiment=md22 logger=tensorboard label=ac_ala3_nhme               4090x1
python train.py experiment=md22 logger=tensorboard label=dha                        4090x1 
python train.py experiment=md22 logger=tensorboard label=buckycatcher               4090x1
python train.py experiment=md22 logger=tensorboard label=double_walled_nanotube     4090x1
python train.py experiment=md22 logger=tensorboard label=stachyose                  4090x1


python train.py experiment=md22_wide logger=tensorboard label=at_at                         l20x2
python train.py experiment=md22_wide logger=tensorboard label=at_at_cg_cg                   l20x2
python train.py experiment=md22_wide logger=tensorboard label=ac_ala3_nhme                  l20x2
python train.py experiment=md22_wide logger=tensorboard label=dha                           l20x2
python train.py experiment=md22_wide logger=tensorboard label=buckycatcher                  l20x2
python train.py experiment=md22_wide logger=tensorboard label=double_walled_nanotube        l20x2
python train.py experiment=md22_wide logger=tensorboard label=stachyose                     l20x2


python train.py experiment=md22_compact logger=tensorboard label=at_at                      l20x2
python train.py experiment=md22_compact logger=tensorboard label=at_at_cg_cg                l20x2
python train.py experiment=md22_compact logger=tensorboard label=ac_ala3_nhme               l20x2
python train.py experiment=md22_compact logger=tensorboard label=dha                        l20x2
python train.py experiment=md22_compact logger=tensorboard label=buckycatcher               l20x2
python train.py experiment=md22_compact logger=tensorboard label=double_walled_nanotube     l20x2
python train.py experiment=md22_compact logger=tensorboard label=stachyose                  l20x2


python train.py experiment=molecule3d logger=tensorboard label=homo                         l20x2
python train.py experiment=molecule3d logger=tensorboard label=lumo                         l20x2
python train.py experiment=molecule3d logger=tensorboard label=gap                          l20x2
python train.py experiment=molecule3d logger=tensorboard label=scf_energy                   l20x2
python train.py experiment=molecule3d logger=tensorboard label=dipole_x                     l20x2
python train.py experiment=molecule3d logger=tensorboard label=dipole_y                     l20x2
python train.py experiment=molecule3d logger=tensorboard label=dipole_z                     l20x2
