"""Factual Stage-1 effect decomposition from completed offline manipulations."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent;V9=HERE.parents[1]

def stats(values):
 x=np.asarray(values,dtype=float);return {'mean':float(x.mean()),'std':float(x.std(ddof=1)),'min':float(x.min()),'max':float(x.max()),'n':len(x)}

def markdown(frame):
 cols=list(frame.columns);out=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
 for _,r in frame.iterrows():out.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in r)+' |')
 return '\n'.join(out)

def ratios(frame,condition,family,metric):
 actual=frame[(frame.family=='temporal_homogenization')&(frame.alpha==0)].set_index(['seed','layer'])[metric]
 target=frame[(frame.family==family)&(frame.condition==condition)]
 if family=='magnitude_sparsity':target=target[target.scale_mode=='global_frobenius_norm_matched']
 if family=='random_sparsity':target=target.groupby(['seed','layer'],as_index=False)[metric].mean()
 return target.assign(ratio=[v/actual.loc[(s,l)] for v,s,l in zip(target[metric],target.seed,target.layer)])

def main():
 all_rows=pd.read_csv(HERE/'offline_all_conditions.csv');t=all_rows[all_rows.family=='temporal_homogenization']
 metrics=['r95','stable_rank','entropy_effective_rank'];verdict={'temporal_homogenization':{},'ann_matched_sparsity':{},'marginal_shuffle':{}}
 lines=['# Offline gate-manipulation analysis','', 'All operations use q and gates from the same trained SNN-DFA checkpoint and neuron coordinate system. They are diagnostic signal manipulations, not learning-performance interventions.','']
 for metric in metrics:
  monotonic=0;endpoint=[]
  for _,g in t.groupby(['seed','layer']):
   g=g.sort_values('alpha');monotonic+=int(np.all(np.diff(g[metric].to_numpy())<=1e-10));endpoint.append(g.iloc[-1][metric]/g.iloc[0][metric])
  verdict['temporal_homogenization'][metric]={'nonincreasing_trajectories':monotonic,'total_trajectories':15,'alpha1_over_alpha0':stats(endpoint)}
 lines+=['## Temporal homogenization','']
 for metric,v in verdict['temporal_homogenization'].items():lines.append(f"- {metric}: non-increasing in {v['nonincreasing_trajectories']}/15 seed-layer trajectories; alpha=1 / alpha=0 mean {v['alpha1_over_alpha0']['mean']:.4f}, range {v['alpha1_over_alpha0']['min']:.4f}-{v['alpha1_over_alpha0']['max']:.4f}.")
 lines+=['','## ANN-matched sparsity','']
 for family,label in [('magnitude_sparsity','magnitude-based'),('random_sparsity','random mask')]:
  verdict['ann_matched_sparsity'][family]={}
  for metric in metrics:
   r=ratios(all_rows,'ann_matched',family,metric);verdict['ann_matched_sparsity'][family][metric]=stats(r.ratio)
   v=verdict['ann_matched_sparsity'][family][metric];lines.append(f"- {label} {metric}: manipulated / actual mean {v['mean']:.4f}, range {v['min']:.4f}-{v['max']:.4f} across 15 seed-layer units.")
 lines+=['','Magnitude masks remove the smallest-|g| entries. Random masks retain a uniformly chosen set and are averaged over 10 deterministic masks within each seed-layer before between-seed summaries. Global norm matching cannot change the normalized spectrum; unscaled and norm-matched spectral metrics are therefore exactly identical by construction.','', '## Per-neuron marginal shuffle','']
 for metric in metrics:
  r=ratios(all_rows,'sample_time_independent_per_neuron','per_neuron_marginal_shuffle',metric);verdict['marginal_shuffle'][metric]=stats(r.ratio);v=verdict['marginal_shuffle'][metric];lines.append(f"- {metric}: shuffled / actual mean {v['mean']:.4f}, range {v['min']:.4f}-{v['max']:.4f} across 15 units.")
 # Layer-specific endpoint effects and cross-depth ratios.
 endpoints=[]
 for family,condition,label in [('temporal_homogenization','alpha_1','alpha1'),('magnitude_sparsity','ann_matched','magnitude_ann'),('random_sparsity','ann_matched','random_ann'),('per_neuron_marginal_shuffle','sample_time_independent_per_neuron','marginal_shuffle')]:
  z=all_rows[(all_rows.family==family)&(all_rows.condition==condition)]
  if family=='magnitude_sparsity':z=z[z.scale_mode=='global_frobenius_norm_matched']
  if family=='random_sparsity':z=z.groupby(['seed','layer'],as_index=False).mean(numeric_only=True)
  for metric in metrics:
   wide=z.pivot(index='seed',columns='layer',values=metric);values=wide.H3/wide.H1
   endpoints.append({'condition':label,'metric':metric,'H3_over_H1_mean':values.mean(),'H3_over_H1_std':values.std(ddof=1),'seed_count':len(values)})
 endpoint_df=pd.DataFrame(endpoints);endpoint_df.to_csv(HERE/'depth_contraction_summary.csv',index=False)
 lines+=['','## Depth-wise ratios after manipulation','',markdown(endpoint_df),'']
 # Compare target ANN gate fractions and actual ANN geometry without mixing tensors.
 ann_gate=pd.read_csv(V9/'tables/5_seed_review/gate_per_seed.csv');ann_gate=ann_gate[(ann_gate.method=='temporal ANN-DFA')&(ann_gate.probe_split=='basis_eval')]
 ann_spec=pd.read_csv(V9/'tables/5_seed_review/spectral_metrics_per_seed.csv');ann_spec=ann_spec[(ann_spec.method=='temporal ANN-DFA')&(ann_spec.probe_split=='basis_eval')&(ann_spec.temporal_mode=='timestep')]
 lines+=['## Reference-only ANN measurements','', 'ANN gates are used only to define per-seed/layer target zero fractions. ANN q and gates are never combined with SNN tensors.','']
 for layer in ('H1','H2','H3'):
  lines.append(f"- {layer}: ANN exact-zero fraction {ann_gate[ann_gate.layer==layer].exact_zero_fraction.mean():.4f}; independently trained ANN r95 {ann_spec[ann_spec.layer==layer].r95.mean():.2f}.")
 (HERE/'offline_verdict.json').write_text(json.dumps(verdict,indent=2),encoding='utf8');(HERE/'offline_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 print(json.dumps(verdict))

if __name__=='__main__':main()
