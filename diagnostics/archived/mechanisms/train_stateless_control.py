"""Train one preregistered stateless-surrogate DFA smoke with canonical settings."""
from pathlib import Path
import sys,json,hashlib,copy,time,torch,argparse
HERE=Path(__file__).resolve().parent;BRANCH=HERE.parent;V9=BRANCH.parent;ROOT=V9.parents[1]
sys.path[:0]=[str(HERE),str(ROOT)]
from stateless_surrogate import make_stateless
from training.config import load_config
from training.data import build_datasets
from analysis.soft_spectral.training import load_paired_model,materialize_dataset
from analysis.paper1_geometry.data import build_paper1_split,make_loaders
from analysis.paper1_geometry.training import IndexedTrainer,train_with_validation

def model_hash(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=20260830,choices=[20260830,20260831,20260901]);args=ap.parse_args();seed=args.seed
 torch.set_num_threads(4);output=HERE/'checkpoints'/f'seed_{seed}'
 if (output/'variant_summary.json').exists():print('smoke already complete');return
 if output.exists() and any(output.iterdir()):raise RuntimeError(f'partial stateless smoke requires audit: {output}')
 config=load_config(ROOT/'configs/nmnist/paper1_experiment03.yaml');config=copy.deepcopy(config)
 config['experiment']['seed']=seed;config['experiment']['name']=f'mechanism_stateless_surrogate_seed_{seed}'
 config['method']['name']='sdfa';config['training']['output_dir']=str(output)
 config['mechanism_exploration']={'variant':'stateless_surrogate','state_equation':'u_t=I_t','canonical_LIF_files_modified':False,'same_surrogate_spike_function':True}
 initial=ROOT/'results/experiment03_soft_spectral/experiment03_20260831_server/initial_states'/f'seed_{seed}.pt'
 raw_train,raw_test=build_datasets(config);train=materialize_dataset(raw_train);test=materialize_dataset(raw_test)
 split=build_paper1_split(len(train),split_seed=20260830,probe_seed=20260831,validation_fraction=.1,probe_size=1024)
 loaders=make_loaders(train,test,split,config,seed=seed)
 model,bank,state=load_paired_model(config=config,initial_state_path=initial,device=torch.device('cuda'))
 before=model_hash(model);replaced=make_stateless(model);after=model_hash(model);assert before==after
 facts={'seed':seed,'status':'launched','initial_state':str(initial.relative_to(ROOT)),'initial_model_checksum':state['model_checksum'],'initial_feedback_checksum':state['feedback_checksum'],'state_dict_hash_unchanged_by_cell_replacement':before,'replaced_cells':replaced,'T':30,'hidden_widths':[800,800,800],'epochs':100,'optimizer':'Adam','learning_rate':.001,'weight_decay':0.,'gradient_clip':1.,'lr_step':60,'lr_gamma':.1,'checkpoint_selection':'validation_accuracy_max_tie_lower_loss','training_split_seed':20260830,'probe_seed':20260831,'no_relu':True,'production_source_modified':False}
 HERE.mkdir(parents=True,exist_ok=True);(HERE/f'launch_facts_seed_{seed}.json').write_text(json.dumps(facts,indent=2),encoding='utf8')
 trainer=IndexedTrainer(model,bank,config,torch.device('cuda'))
 old_eval=trainer.evaluate;counter=[0]
 def logged(loader):
  r=old_eval(loader)
  if loader is loaders['validation']:
   counter[0]+=1;print(json.dumps({'event':'validation','seed':seed,'epoch':counter[0],**r}),flush=True)
  return r
 trainer.evaluate=logged
 train_with_validation(trainer=trainer,loaders=loaders,config=config,output_dir=output,method='stateless_surrogate_dfa',seed=seed,epochs=100)
 summary=json.loads((output/'variant_summary.json').read_text())
 print(json.dumps({'event':'training_complete','seed':seed,'accuracy':summary['best_validation_test']['accuracy']}),flush=True)

if __name__=='__main__':main()
