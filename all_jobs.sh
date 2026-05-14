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


python train.py experiment=rmd17.yaml label=aspirin         logger=csv.yaml
python train.py experiment=rmd17.yaml label=azobenzene      logger=csv.yaml
python train.py experiment=rmd17.yaml label=benzene         logger=csv.yaml

python train.py experiment=rmd17.yaml label=ethanol         logger=csv.yaml
python train.py experiment=rmd17.yaml label=malonaldehyde   logger=csv.yaml
python train.py experiment=rmd17.yaml label=naphthalene     logger=csv.yaml

python train.py experiment=rmd17.yaml label=paracetamol     logger=csv.yaml
python train.py experiment=rmd17.yaml label=salicylic       logger=csv.yaml

python train.py experiment=rmd17.yaml label=toluene         logger=csv.yaml
python train.py experiment=rmd17.yaml label=uracil          logger=csv.yaml



python train.py experiment=md22.yaml label=at_at                    logger=csv.yaml
python train.py experiment=md22.yaml label=at_at_cg_cg              logger=csv.yaml
python train.py experiment=md22.yaml label=ac_ala3_nhme             logger=csv.yaml
python train.py experiment=md22.yaml label=dha                      logger=csv.yaml
python train.py experiment=md22.yaml label=buckycatcher             logger=csv.yaml
python train.py experiment=md22.yaml label=double_walled_nanotube   logger=csv.yaml
python train.py experiment=md22.yaml label=stachyose                logger=csv.yaml
