"""Within-model offline gate manipulations on canonical five-seed DFA checkpoints."""
from pathlib import Path
import sys,csv,json,math,hashlib,time
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent;BRANCH=HERE.parents[0];V9=BRANCH.parent;ROOT=V9.parents[1]
sys.path.insert(0,str(V9));sys.path.insert(0,str(ROOT))
from replay_v9 import spectral,gate_stats,sha,_prepare,_capture_snn_dfa_layer
from replay_v9 import FORMAL,LEGACY
from training.data import build_datasets
from analysis.paper1_geometry.data import build_paper1_split,make_loaders
from models import build_model
from methods import FeedbackBank

SEEDS=(20260830,20260831,20260901,20260908,20260909)
LAYERS=('H1','H2','H3')

def write_csv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not rows:raise RuntimeError(f'no rows: {path}')
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def write_or_validate(path,rows):
 if path.exists():
  if len(pd.read_csv(path))!=len(rows):raise RuntimeError(f'partial aggregate conflict: {path}')
  return
 write_csv(path,rows)

def checkpoint(seed):
 if seed in SEEDS[:3]:return LEGACY/'checkpoints/full_dense'/f'seed_{seed}'/'epoch_100.pt'
 return V9/'03_five_seed/checkpoints/dfa_trained'/f'seed_{seed}'/'epoch_100.pt'

def metric_row(base,g,d):
 m,_,H=spectral(d)
 return {**base,**gate_stats(g),'r50':m['r50'],'r80':m['r80'],'r90':m['r90'],'r95':m['r95'],'r99':m['r99'],'stable_rank':m['stable_rank'],'spectral_entropy':H,'entropy_effective_rank':m['entropy_rank'],'normalized_entropy_effective_rank':m['entropy_rank']/d.shape[-1]}

def exact_smallest_mask(g,fraction):
 flat=g.reshape(-1);k=int(round(float(fraction)*flat.numel()))
 order=torch.argsort(flat.abs(),stable=True);mask=torch.ones(flat.numel(),dtype=torch.bool);mask[order[:k]]=False
 return mask.reshape_as(g),k

def random_nested_masks(shape,fractions,seed):
 size=int(np.prod(shape));rng=np.random.default_rng(seed);order=rng.permutation(size);masks={}
 for f in fractions:
  k=int(round(float(f)*size));mask=np.ones(size,dtype=np.bool_);mask[order[:k]]=False;masks[f]=torch.from_numpy(mask.reshape(shape))
 return masks

def rescale(g,reference):
 return g*(torch.linalg.vector_norm(reference)/(torch.linalg.vector_norm(g)+1e-12))

def marginal_shuffle(g,seed):
 # Preserve each neuron coordinate and its complete sample*time value multiset.
 x=g.permute(1,0,2).reshape(-1,g.shape[-1]).numpy().copy();original=x.copy();rng=np.random.default_rng(seed)
 for neuron in range(x.shape[1]):x[:,neuron]=original[rng.permutation(x.shape[0]),neuron]
 assert np.array_equal(np.sort(x,axis=0),np.sort(original,axis=0))
 return torch.from_numpy(x).reshape(g.shape[1],g.shape[0],g.shape[2]).permute(1,0,2).contiguous()

def main():
 torch.set_num_threads(4);work=HERE/'work';work.mkdir(parents=True,exist_ok=True)
 ann_source=V9/'04_gate_statistics/per_seed.csv'
 if not ann_source.exists():ann_source=V9/'tables/5_seed_review/gate_per_seed.csv'
 ann=pd.read_csv(ann_source)
 ann=ann[(ann.method=='temporal ANN-DFA')&(ann.probe_split=='basis_eval')]
 for seed in SEEDS:
  cp=checkpoint(seed);state=torch.load(cp,map_location='cpu',weights_only=False);config=state['config']
  model=build_model(config).cuda();model.load_state_dict(state['model_state']);model.eval()
  bank=FeedbackBank(model,config).cuda();bank.load_state_dict(state['feedback_state']);bank.eval()
  train,test=build_datasets(config);split=build_paper1_split(len(train),split_seed=20260830,probe_seed=20260831,validation_fraction=.1,probe_size=1024)
  with (FORMAL/'diagnostic_probe.csv').open() as f:probe=list(csv.DictReader(f))
  assert split.probe_indices.tolist()==[int(r['dataset_index']) for r in probe]
  batches=[[] for _ in LAYERS];gates=[[] for _ in LAYERS];qs=[[] for _ in LAYERS]
  for batch in make_loaders(train,test,split,config,seed=seed)['basis_eval']:
   x,y,idx=_prepare(batch,torch.device('cuda'))
   with torch.no_grad():
    output=model.forward_with_cache(x,detach_temporal=True);error=output[0].mean(0)-torch.nn.functional.one_hot(y,10).to(output[0].dtype)
   for i,inp in enumerate(output[1]):
    cap=_capture_snn_dfa_layer(model,bank,i,inp,error,gate_mode='actual',seed=seed,epoch=100,sample_indices=idx)
    gates[i].append(cap['gate'].cpu());qs[i].append(cap['q'].cpu())
  for i,layer in enumerate(LAYERS):
   destination=work/f'seed_{seed}_{layer}.csv'
   if destination.exists():continue
   g=torch.cat(gates[i],1);q=torch.cat(qs[i],0);assert tuple(g.shape)==(30,512,800) and tuple(q.shape)==(512,800)
   common={'dataset':'N-MNIST','source_model':'SNN-DFA','seed':seed,'layer':layer,'checkpoint':'completed_epoch_100','checkpoint_file':str(cp.relative_to(ROOT)),'checkpoint_sha256':sha(cp),'probe_split':'basis_eval','N':512,'T':30,'D':800,'coordinate_system':'same_trained_model_and_neuron_coordinates','offline_only':True}
   rows=[]
   # 1A: convex direction interpolation followed by one full-tensor Frobenius scale.
   mean=g.mean(0,keepdim=True).expand_as(g)
   for alpha in (0.,.25,.5,.75,1.):
    mixed=(1-alpha)*g+alpha*mean;applied=rescale(mixed,g);d=applied*q.unsqueeze(0)
    rows.append(metric_row({**common,'family':'temporal_homogenization','condition':f'alpha_{alpha:g}','alpha':alpha,'scale_mode':'global_frobenius_norm_matched','mask_seed':'','target_zero_fraction':'','achieved_zero_fraction':float((applied==0).double().mean()),'spectrum_reused_by_scale_invariance':False},applied,d))
   # 1B: sort only once; all target masks are deterministic and exact by entry count.
   ann_target=float(ann[(ann.seed==seed)&(ann.layer==layer)].exact_zero_fraction.iloc[0]);targets=[('0',0.),('0.25',.25),('0.50',.5),('0.75',.75),('ann_matched',ann_target)]
   flat_order=torch.argsort(g.reshape(-1).abs(),stable=True);size=g.numel()
   for label,target in targets:
    k=int(round(target*size));mask=torch.ones(size,dtype=torch.bool);mask[flat_order[:k]]=False;masked=g*mask.reshape_as(g);achieved=float((masked==0).double().mean())
    unscaled=metric_row({**common,'family':'magnitude_sparsity','condition':label,'alpha':'','scale_mode':'unscaled','mask_seed':'','target_zero_fraction':target,'achieved_zero_fraction':achieved,'ann_target_source':str(ann_source.relative_to(V9)),'spectrum_reused_by_scale_invariance':False},masked,masked*q.unsqueeze(0));rows.append(unscaled)
    matched=rescale(masked,g)
    duplicate={**unscaled,'scale_mode':'global_frobenius_norm_matched','l2_norm_all_T_N_D':float(torch.linalg.vector_norm(matched)),'rms_g':float(matched.double().square().mean().sqrt()),'mean_abs_g':float(matched.double().abs().mean()),'mean_per_sample_timestep_l2':float(matched.double().norm(dim=-1).mean()),'spectrum_reused_by_scale_invariance':True}
    rows.append(duplicate)
   # 1C: ten deterministic random masks; nested target sets within one mask seed.
   fractions=[x[1] for x in targets]
   for replicate in range(10):
    mask_seed=202609100000+seed*100+i*10+replicate;masks=random_nested_masks(g.shape,fractions,mask_seed)
    for label,target in targets:
     masked=g*masks[target];matched=rescale(masked,g);d=matched*q.unsqueeze(0)
     rows.append(metric_row({**common,'family':'random_sparsity','condition':label,'alpha':'','scale_mode':'global_frobenius_norm_matched','mask_seed':mask_seed,'mask_replicate':replicate,'target_zero_fraction':target,'achieved_zero_fraction':float((matched==0).double().mean()),'ann_target_source':str(ann_source.relative_to(V9)),'spectrum_reused_by_scale_invariance':False},matched,d))
   # 1D: independent observation permutation inside every fixed neuron coordinate.
   shuffle_seed=202609190000+seed*10+i;shuffled=marginal_shuffle(g,shuffle_seed)
   rows.append(metric_row({**common,'family':'per_neuron_marginal_shuffle','condition':'sample_time_independent_per_neuron','alpha':'','scale_mode':'value_multiset_exact','mask_seed':shuffle_seed,'target_zero_fraction':float((g==0).double().mean()),'achieved_zero_fraction':float((shuffled==0).double().mean()),'marginal_preservation_check':'exact sorted values per neuron','spectrum_reused_by_scale_invariance':False},shuffled,shuffled*q.unsqueeze(0)))
   write_csv(destination,rows);print(json.dumps({'seed':seed,'layer':layer,'rows':len(rows),'status':'complete'}),flush=True)
  del model,bank,gates,qs;torch.cuda.empty_cache()
 files=sorted(work.glob('seed_*_H*.csv'));assert len(files)==15
 all_rows=pd.concat([pd.read_csv(p) for p in files],ignore_index=True);write_or_validate(HERE/'offline_all_conditions.csv',all_rows.to_dict('records'))
 temporal=all_rows[all_rows.family=='temporal_homogenization'];write_or_validate(HERE/'temporal_homogenization_per_seed.csv',temporal.to_dict('records'))
 temporal_summary=temporal.groupby(['family','layer','alpha','scale_mode'],as_index=False).agg(seed_count=('seed','nunique'),temporal_gate_cosine_mean=('temporal_gate_cosine','mean'),temporal_gate_cosine_std=('temporal_gate_cosine','std'),gate_temporal_variance_mean=('gate_temporal_variance','mean'),gate_temporal_variance_std=('gate_temporal_variance','std'),norm_cv_mean=('norm_cv_over_time','mean'),norm_cv_std=('norm_cv_over_time','std'),r95_mean=('r95','mean'),r95_std=('r95','std'),stable_rank_mean=('stable_rank','mean'),stable_rank_std=('stable_rank','std'),entropy_effective_rank_mean=('entropy_effective_rank','mean'),entropy_effective_rank_std=('entropy_effective_rank','std'),normalized_entropy_effective_rank_mean=('normalized_entropy_effective_rank','mean'),normalized_entropy_effective_rank_std=('normalized_entropy_effective_rank','std'))
 write_or_validate(HERE/'temporal_homogenization_summary.csv',temporal_summary.to_dict('records'))
 for family,name in [('magnitude_sparsity','magnitude_sparsification'),('random_sparsity','random_mask_control'),('per_neuron_marginal_shuffle','marginal_distribution_shuffle')]:
  f=all_rows[all_rows.family==family];write_or_validate(HERE/f'{name}_per_seed.csv',f.to_dict('records'))
  if family=='random_sparsity':f=f.groupby(['family','seed','layer','condition','scale_mode'],as_index=False).mean(numeric_only=True)
  keys=['family','layer','condition','scale_mode'];metrics=['target_zero_fraction','achieved_zero_fraction','temporal_gate_cosine','gate_temporal_variance','norm_cv_over_time','r95','stable_rank','entropy_effective_rank','normalized_entropy_effective_rank']
  summary=f.groupby(keys,as_index=False)[metrics].agg(['mean','std']);summary.columns=['_'.join(x).strip('_') for x in summary.columns];write_or_validate(HERE/f'{name}_summary.csv',summary.to_dict('records'))
 with (HERE/'offline_complete.json').open('x',encoding='utf8') as f:json.dump({'status':'complete','seeds':list(SEEDS),'layers':list(LAYERS),'work_files':len(files),'rows':len(all_rows),'historical_v9_modified':False},f,indent=2)

if __name__=='__main__':main()
