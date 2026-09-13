"""Scientific CSV intermediates; keep 3-seed snapshot separate from eventual 5-seed freeze."""
from pathlib import Path
import argparse,json,csv
import pandas as pd
import numpy as np
OUT=Path(__file__).resolve().parent
OLD=(20260830,20260831,20260901)
METHODS=('dfa_trained','bptt_trained','temporal_ann_dfa','mean_gate','shuffled_gate')

def save(frame,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',newline='',encoding='utf8') as f:frame.to_csv(f,index=False)

def summary(df,keys,metrics):
 result=[]
 for ids,g in df.groupby(keys,sort=True,dropna=False):
  if not isinstance(ids,tuple):ids=(ids,)
  for metric in metrics:
   result.append({**dict(zip(keys,ids)),'metric':metric,'seed_count':g.seed.nunique(),'mean':g[metric].mean(),'std':g[metric].std(ddof=1),'std_definition':'between_seed_sample_std_ddof1','source_file':';'.join(sorted(g.input_csv.unique())),'git_commit':'UNAVAILABLE'})
 return pd.DataFrame(result)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed-count',type=int,choices=(3,5),required=True);a=ap.parse_args()
 seeds=OLD if a.seed_count==3 else (*OLD,20260908,20260909)
 target=OUT/'tables'/f'{a.seed_count}_seed_review'
 spectra=[];gates=[];samples=[]
 for method in METHODS:
  for seed in seeds:
   done=OUT/'logs'/f'{method}_{seed}'/'complete.json'
   assert done.exists(),f'Incomplete replay: {done}'
   for folder,destination in [('01_spectral_metrics',spectra),('04_gate_statistics',gates)]:
    p=OUT/folder/f'{method}_{seed}.csv';df=pd.read_csv(p);df['input_csv']=str(p.relative_to(OUT));destination.append(df)
   if method in METHODS[:3]:
    p=OUT/'02_sample_robustness'/f'{method}_{seed}.csv';df=pd.read_csv(p);df['input_csv']=str(p.relative_to(OUT));samples.append(df)
 s=pd.concat(spectra,ignore_index=True);g=pd.concat(gates,ignore_index=True);r=pd.concat(samples,ignore_index=True)
 assert len(s)==len(seeds)*5*12 and len(g)==len(seeds)*5*6
 assert not s.duplicated(['method','seed','layer','probe_split','temporal_mode']).any()
 assert set(s.seed)==set(seeds)
 keys=['dataset','method','checkpoint_rule','layer','probe_split']
 sm=summary(s,keys+['temporal_mode'],['r50','r80','r90','r95','r99','stable_rank','spectral_entropy','entropy_effective_rank','stable_rank_over_D','entropy_rank_over_D'])
 gm=summary(g,keys,['mean_abs_g','rms_g','l2_norm_all_T_N_D','exact_zero_fraction','near_zero_fraction','temporal_gate_cosine','adjacent_timestep_cosine','norm_cv_over_time','gate_temporal_variance'])
 save(s,target/'spectral_metrics_per_seed.csv');save(sm,target/'spectral_metrics_summary.csv')
 save(g,target/'gate_per_seed.csv');save(gm,target/'gate_summary.csv');save(r,target/'sample_metrics.csv')
 # Repeats are nested within seed, not treated as additional independent seeds.
 rm=r.groupby(keys+['seed','N','T','observations','input_csv'],as_index=False)[['r95','stable_rank','entropy_effective_rank']].mean()
 rs=summary(rm,keys+['N','T','observations'],['r95','stable_rank','entropy_effective_rank'])
 save(rs,target/'sample_summary.csv')
 ss=r.groupby(keys+['seed','N'],as_index=False)[['r95','stable_rank','entropy_effective_rank']].std(ddof=1)
 save(ss,target/'within_seed_subsample_std.csv')
 assert (ss[ss.N==512][['r95','stable_rank','entropy_effective_rank']].fillna(0).abs()<1e-8).all().all()
 # Within each trained MeanGate checkpoint the repeated-timestep spectrum should match its own collapse.
 mg=s[(s.method=='MeanGate')&(s.probe_split=='basis_eval')]
 pair=mg.pivot(index=['seed','layer'],columns='temporal_mode',values='r95')
 assert (pair['timestep']==pair['time-collapsed']).all()
 save(pair.reset_index(),target/'meangate_repetition_check.csv')
 lines=[f'# {a.seed_count}-seed real-checkpoint replay review','',
 'This is a checkpoint-replay snapshot, not the final v9 freeze. Accuracy, trajectory, DVS and final figures are separate pending branches.',
 '',f'Spectral rows: {len(s)}. Gate rows: {len(g)}. Subsample rows: {len(r)}.',
 'Fits and evaluation sets remain separate. Mean±SD is across seeds (ddof=1).',
 'N=512 has only one distinct full probe, repeated 20 times; these are not independent resamples.',
 'Subsample summary first averages 20 repeats within each seed, then summarizes seeds.',
 'MeanGate timestep r95 equals its own time-collapsed r95 in every evaluated layer/seed.',
 '', '## Evaluation timestep metrics','']
 for _,row in sm[(sm.probe_split=='basis_eval')&(sm.temporal_mode=='timestep')&(sm.metric.isin(['r95','stable_rank','entropy_effective_rank']))].iterrows():
  lines.append(f"- {row['method']} {row['layer']} {row['metric']}: {row['mean']:.6g} ± {row['std']:.6g}")
 lines+=['','## Limits','',
 'Between-method spectral differences do not identify leak/reset as a unique cause.',
 'Exact-zero/near-zero gate statistics are descriptive, not causal mediation tests.',
 'Source Git commit is unavailable in this unpacked repository. Checkpoint and source hashes are retained elsewhere; no commit is invented.']
 with (target/'report.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
 print(json.dumps({'path':str(target),'spectral_rows':len(s),'gate_rows':len(g),'subsample_rows':len(r)}))

if __name__=='__main__':main()
