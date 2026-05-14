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



python train.py experiment=md22.yaml logger=csv.yaml label=at_at                    datamodule.hparams.dataset_arg=at_at 
python train.py experiment=md22.yaml logger=csv.yaml label=at_at_cg_cg              datamodule.hparams.dataset_arg=at_at_cg_cg 
python train.py experiment=md22.yaml logger=csv.yaml label=ac_ala3_nhme             datamodule.hparams.dataset_arg=ac_ala3_nhme 
python train.py experiment=md22.yaml logger=csv.yaml label=dha                      datamodule.hparams.dataset_arg=dha 
python train.py experiment=md22.yaml logger=csv.yaml label=buckycatcher             datamodule.hparams.dataset_arg=buckycatcher 
python train.py experiment=md22.yaml logger=csv.yaml label=double_walled_nanotube   datamodule.hparams.dataset_arg=double_walled_nanotube 
python train.py experiment=md22.yaml logger=csv.yaml label=stachyose                datamodule.hparams.dataset_arg=stachyose 
