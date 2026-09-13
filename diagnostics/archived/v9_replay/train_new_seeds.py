"""Two preregistered seeds, unchanged production trainers, fail before training on parity errors."""
from pathlib import Path
import copy, json, os, sys, time
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
from training.config import load_config
from training.data import build_datasets
from analysis.soft_spectral.training import save_paired_initial_state, load_paired_model, materialize_dataset
from analysis.paper1_geometry.data import build_paper1_split, make_loaders
from analysis.paper1_geometry.training import IndexedTrainer, GateInterventionTrainer, TemporalANNTrainer, build_matched_ann, train_with_validation
from methods.gate_intervention import GateMode
from replay_v9 import METHODS, SEEDS, write_json

def main():
 torch.set_num_threads(4)
 # No training unless every original method/seed has completed numerical replay.
 for method,info in METHODS.items():
  for seed in SEEDS:
   p=OUT/'logs'/f'{info[1]}_{seed}'/'complete.json'
   if not p.exists():raise RuntimeError(f'Prerequisite parity incomplete: {p}')
 if list((OUT/'logs').rglob('BLOCKED.json')):raise RuntimeError('Blocked diagnostic branch; no new training')
 config=load_config(ROOT/'configs/nmnist/paper1_experiment03.yaml')
 raw_train,raw_test=build_datasets(config)
 # Materialize in RAM, no extra dataset disk cache.
 train=materialize_dataset(raw_train);test=materialize_dataset(raw_test)
 split=build_paper1_split(len(train),split_seed=20260830,probe_seed=20260831,validation_fraction=.1,probe_size=1024)
 device=torch.device('cuda')
 for seed in (20260908,20260909):
  initial=OUT/'03_five_seed'/'initial_states'/f'seed_{seed}.pt'
  if not initial.exists():save_paired_initial_state(config=config,seed=seed,path=initial)
  for display,info in METHODS.items():
   method=info[1];mode=info[3];destination=OUT/'03_five_seed'/'checkpoints'/method/f'seed_{seed}'
   if (destination/'variant_summary.json').exists():continue
   if destination.exists() and any(destination.iterdir()):raise RuntimeError(f'Partial training exists; manual resume audit required: {destination}')
   c=copy.deepcopy(config);c['experiment']['seed']=seed;c['experiment']['name']=f'paper_v9_{method}_{seed}'
   c['method']['name']='bptt' if display=='matched SNN-BPTT' else 'sdfa'
   c.setdefault('paper1',{})['gate_mode']=mode;c['paper1']['model_kind']='temporal_ann' if display=='temporal ANN-DFA' else 'snn'
   c['training']['output_dir']=str(destination)
   loaders=make_loaders(train,test,split,c,seed=seed)
   if display=='temporal ANN-DFA':
    state=torch.load(initial,map_location='cpu',weights_only=False);model,bank=build_matched_ann(c,state,device)
    trainer=TemporalANNTrainer(model,bank,c,device)
   else:
    model,bank,_=load_paired_model(config=c,initial_state_path=initial,device=device)
    trainer=GateInterventionTrainer(model,bank,c,device,gate_mode=mode,seed=seed) if display in ('MeanGate','ShuffledGate') else IndexedTrainer(model,bank,c,device)
   print(json.dumps({'event':'training_start','method':display,'seed':seed,'time':time.time()}),flush=True)
   # Log each completed epoch without changing the production update.
   old_eval=trainer.evaluate
   count=[0]
   def logged_eval(loader):
    r=old_eval(loader)
    if loader is loaders['validation']:
     count[0]+=1;print(json.dumps({'event':'validation','method':display,'seed':seed,'epoch':count[0],**r}),flush=True)
    return r
   trainer.evaluate=logged_eval
   train_with_validation(trainer=trainer,loaders=loaders,config=c,output_dir=destination,method=method,seed=seed,epochs=100)
   print(json.dumps({'event':'training_complete','method':display,'seed':seed}),flush=True)
   del trainer,model,bank;torch.cuda.empty_cache()
 write_json(OUT/'03_five_seed'/'new_training_complete.json',{'new_seeds':[20260908,20260909],'methods':list(METHODS),'status':'complete'})

if __name__=='__main__':main()
