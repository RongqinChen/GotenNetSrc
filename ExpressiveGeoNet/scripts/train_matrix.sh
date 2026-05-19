#!/usr/bin/env bash

source ~/miniforge3/bin/activate gnn
export CUDA_VISIBLE_DEVICES=0

python train.py experiment=qm9_small logger=tensorboard label=mu    
python train.py experiment=qm9_small logger=tensorboard label=alpha 
python train.py experiment=qm9_small logger=tensorboard label=homo  
python train.py experiment=qm9_small logger=tensorboard label=lumo  
python train.py experiment=qm9_small logger=tensorboard label=gap   
python train.py experiment=qm9_small logger=tensorboard label=r2    
python train.py experiment=qm9_small logger=tensorboard label=Cv    
python train.py experiment=qm9_small logger=tensorboard label=G     
python train.py experiment=qm9_small logger=tensorboard label=H     
python train.py experiment=qm9_small logger=tensorboard label=U     
python train.py experiment=qm9_small logger=tensorboard label=U0    
python train.py experiment=qm9_small logger=tensorboard label=zpve  


python train.py experiment=rmd17 logger=tensorboard label=aspirin         
python train.py experiment=rmd17 logger=tensorboard label=azobenzene      
python train.py experiment=rmd17 logger=tensorboard label=benzene         
python train.py experiment=rmd17 logger=tensorboard label=ethanol         
python train.py experiment=rmd17 logger=tensorboard label=malonaldehyde   
python train.py experiment=rmd17 logger=tensorboard label=naphthalene     
python train.py experiment=rmd17 logger=tensorboard label=paracetamol     
python train.py experiment=rmd17 logger=tensorboard label=salicylic       
python train.py experiment=rmd17 logger=tensorboard label=toluene         
python train.py experiment=rmd17 logger=tensorboard label=uracil          


python train.py experiment=md22 logger=tensorboard label=at_at    
python train.py experiment=md22 logger=tensorboard label=at_at_cg_cg    
python train.py experiment=md22 logger=tensorboard label=ac_ala3_nhme   
python train.py experiment=md22 logger=tensorboard label=dha             
python train.py experiment=md22 logger=tensorboard label=buckycatcher    
python train.py experiment=md22 logger=tensorboard label=double_walled_nanotube  
python train.py experiment=md22 logger=tensorboard label=stachyose      

python train.py experiment=molecule3d logger=tensorboard label=homo        
python train.py experiment=molecule3d logger=tensorboard label=lumo        
python train.py experiment=molecule3d logger=tensorboard label=gap         
python train.py experiment=molecule3d logger=tensorboard label=scf_energy  
python train.py experiment=molecule3d logger=tensorboard label=dipole_x    
python train.py experiment=molecule3d logger=tensorboard label=dipole_y    
python train.py experiment=molecule3d logger=tensorboard label=dipole_z    
