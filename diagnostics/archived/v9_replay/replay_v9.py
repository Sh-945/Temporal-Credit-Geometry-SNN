"""Real-checkpoint replay, exclusive output creation, no training or basis comparison."""
from pathlib import Path
import argparse, csv, hashlib, json, math, sys, time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models import build_model
from methods import FeedbackBank
from models.temporal_ann_control import TemporalANN
from analysis.paper1_geometry.diagnostics import _prepare, _capture_snn_dfa_layer, _capture_snn_gate, _capture_ann_layer
from analysis.update_relevance.core import capture_bptt_updates
from analysis.feedback_subspace.metrics import OnlineCovariance, spectrum_statistics
from analysis.feedback_expansion.core import temporal_coherence
from analysis.paper1_geometry.data import build_paper1_split, make_loaders
from training.data import build_datasets
from methods.gate_intervention import GateMode

OUT=Path(__file__).resolve().parent
FORMAL=ROOT/'results/paper1_experiment03_20260901_server'
LEGACY=ROOT/'results/experiment03_soft_spectral/experiment03_20260831_server'
SEEDS=(20260830,20260831,20260901)
METHODS={
 'SNN-DFA':('stageA_bptt','dfa_trained','dfa_actual','actual','full_dense'),
 'matched SNN-BPTT':('stageA_bptt','bptt_trained','bptt_actual','actual','full_bptt'),
 'temporal ANN-DFA':('stageB_ann','temporal_ann_dfa','ann_dfa_actual','actual',None),
 'MeanGate':('stageC_gate','mean_gate','dfa_temporal_mean_normmatched','temporal_mean_normmatched',None),
 'ShuffledGate':('stageC_gate','shuffled_gate','dfa_timestep_shuffled','timestep_shuffled',None),
}
def write_json(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f: json.dump(x,f,indent=2,allow_nan=False)
def write_csv(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not rows: raise ValueError(f'No rows for {p}')
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with p.open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def spectral(x):
 acc=OnlineCovariance(x.shape[-1])
 # Match the four canonical minibatch covariance chunks exactly.
 for start in range(0,x.shape[1] if x.ndim==3 else x.shape[0],128):
  acc.update(x[:,start:start+128].cuda() if x.ndim==3 else x[start:start+128].cuda())
 m,s,p,c=spectrum_statistics(acc,algebraic_max_dim=x.shape[-1],device=torch.device('cuda'))
 H=float(-np.sum(p[p>0]*np.log(p[p>0]))) if p.sum()>0 else None
 return m,s,H
def gate_stats(g):
 # Explicit population statistics. Dimensions are T,N,D.
 x=g.double();norm=x.norm(dim=-1);mean=norm.mean(dim=0)
 norm_cv=norm.std(dim=0,correction=0)/(mean+1e-12)
 units=x/(norm.unsqueeze(-1)+1e-12)
 adjacent=(units[1:]*units[:-1]).sum(-1)
 _,cos=temporal_coherence(g)
 return {'mean_abs_g':float(x.abs().mean()),'rms_g':float(x.square().mean().sqrt()),
 'l2_norm_all_T_N_D':float(x.norm()),'mean_per_sample_timestep_l2':float(norm.mean()),
 'exact_zero_fraction':float((x==0).double().mean()),
 'near_zero_fraction':float((x.abs()<1e-6).double().mean()),'near_zero_threshold':1e-6,
 'temporal_gate_cosine':float(cos.mean()),'adjacent_timestep_cosine':float(adjacent.mean()),
 'norm_cv_over_time':float(norm_cv.mean()),'gate_temporal_variance':float(x.var(dim=0,correction=0).mean()),
 'zero_norm_fraction':float((norm==0).double().mean()),
 'cosine_definition':'canonical epsilon=1e-12; zero-vector effects retained',
 'variance_definition':'population, time axis then sample/feature mean'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--method',choices=list(METHODS),required=True);ap.add_argument('--seed',type=int,required=True)
 ap.add_argument('--robustness-repeats',type=int,default=20);ap.add_argument('--checkpoint');a=ap.parse_args()
 torch.set_num_threads(4)
 stage,method,signal,mode,legacy=METHODS[a.method]
 cp=(LEGACY/'checkpoints'/legacy if legacy else FORMAL/'checkpoints'/stage/method)/f'seed_{a.seed}'/'epoch_100.pt'
 if a.checkpoint:
  assert a.seed not in SEEDS,'canonical checkpoints cannot be replaced'
  cp=Path(a.checkpoint)
 job=OUT/'logs'/f'{method}_{a.seed}';job.mkdir(parents=True,exist_ok=True)
 if (job/'complete.json').exists():print('already complete',job,flush=True);return
 start=time.time();state=torch.load(cp,map_location='cpu',weights_only=False);config=state['config']
 model=(TemporalANN(config['model']['input_shape'],config['model']['hidden_features'],config['model']['num_classes'])
        if a.method=='temporal ANN-DFA' else build_model(config)).cuda()
 bank=FeedbackBank(model,config).cuda();model.load_state_dict(state['model_state']);bank.load_state_dict(state['feedback_state']);model.eval();bank.eval()
 assert state['epoch']==99,('checkpoint epoch conflict',state['epoch'])
 train,test=build_datasets(config)
 split=build_paper1_split(len(train),split_seed=20260830,probe_seed=20260831,validation_fraction=.1,probe_size=1024)
 with (FORMAL/'diagnostic_probe.csv').open() as f: archived=list(csv.DictReader(f))
 assert split.probe_indices.tolist()==[int(r['dataset_index']) for r in archived],'probe mismatch'
 loaders=make_loaders(train,test,split,config,seed=a.seed)
 with (FORMAL/stage/'geometry.csv').open() as f: expected=list(csv.DictReader(f))
 metrics=[];gates=[];checks=[];robust=[];shapes=[];feedback=[]
 for i,b in enumerate(bank.layers):
  B=b.B.detach().double().cpu()
  feedback.append({'layer':f'H{i+1}','shape':list(B.shape),'measured_rank':int(torch.linalg.matrix_rank(B)), 'sha256_tensor':hashlib.sha256(B.numpy().tobytes()).hexdigest()})
 for split_name in ('basis_fit','basis_eval'):
  values=[[] for _ in range(3)];gs=[[] for _ in range(3)];qs=[[] for _ in range(3)]
  for batch in loaders[split_name]:
   x,y,idx=_prepare(batch,torch.device('cuda'))
   assert x.shape[0]==30
   with torch.no_grad():
    outputs=model.forward_with_cache(x) if a.method=='temporal ANN-DFA' else model.forward_with_cache(x,detach_temporal=True)
    e=outputs[0].mean(0)-torch.nn.functional.one_hot(y,10).to(outputs[0].dtype)
   bp=capture_bptt_updates(model,x,y) if a.method=='matched SNN-BPTT' else None
   for i,inp in enumerate(outputs[1]):
    if bp is not None:
     d=bp['delta'][i];g=_capture_snn_gate(model,i,inp);q=None
    elif a.method=='temporal ANN-DFA':q,g,d,parity=_capture_ann_layer(model,bank,i,inp,e)
    else:
     c=_capture_snn_dfa_layer(model,bank,i,inp,e,gate_mode=mode,seed=a.seed,epoch=100,sample_indices=idx)
     q,g,d=c['q'],c['applied_gate'],c['delta']
    values[i].append(d.detach().cpu());gs[i].append(g.detach().cpu())
    if q is not None:qs[i].append(q.detach().cpu())
  for i in range(3):
   d=torch.cat(values[i],dim=1);g=torch.cat(gs[i],dim=1);layer=f'hidden_{i+1}'
   assert tuple(d.shape)==(30,512,800),('credit shape conflict',tuple(d.shape))
   base={'dataset':'N-MNIST','method':a.method,'seed':a.seed,'layer':f'H{i+1}',
     'checkpoint':str(cp.relative_to(ROOT)),'checkpoint_rule':'completed_epoch_100','width':800,
     'probe_split':split_name,'source_file':str(cp.relative_to(ROOT)),'checkpoint_sha256':sha(cp),
     'git_commit':'UNAVAILABLE','provenance_status':'real_checkpoint_replay'}
   for temporal,v in [('timestep',d),('time-collapsed',d.mean(0))]:
    m,s,H=spectral(v)
    ref=[r for r in expected if r['method']==method and int(r['seed'])==a.seed and int(r['epoch'])==100 and r['layer']==layer and r['signal_type']==signal and r['probe_split']==split_name and r['residualization']=='raw' and r['temporal_mode']==('aggregated' if temporal=='time-collapsed' else temporal)]
    assert len(ref)==(1 if a.seed in SEEDS else 0),('ambiguous historical source',method,signal,len(ref))
    parity_ok=not ref or all(int(ref[0][k])==m[k] for k in ('r50','r80','r90','r95','r99'))
    continuous_ok=not ref or all(np.isclose(float(ref[0][k]),m[k],rtol=2e-4,atol=1e-5) for k in ('stable_rank','entropy_rank'))
    checks.append({**base,'temporal_mode':temporal,'historical_r95':ref[0]['r95'] if ref else None,'replay_r95':m['r95'],'historical_comparison_available':bool(ref),'rank_thresholds_exact':parity_ok if ref else None,'continuous_metrics_rtol_2e_4':continuous_ok if ref else None})
    if not(parity_ok and continuous_ok):
     write_json(job/'BLOCKED.json',{'reason':'historical canonical result cannot be reproduced','checks':checks,'replay_metrics':m,'historical_row':ref[0]})
     raise RuntimeError('Historical mismatch. Corresponding branch stopped; see BLOCKED.json')
    row={**base,'temporal_mode':temporal,**{k:m[k] for k in ('r50','r80','r90','r95','r99','stable_rank')},
     'spectral_entropy':H,'entropy_effective_rank':m['entropy_rank'],'stable_rank_over_D':m['stable_rank']/800,'entropy_rank_over_D':m['entropy_rank']/800}
    metrics.append(row)
    p=OUT/'01_spectral_metrics'/'singular_values'/f'{method}_{a.seed}_H{i+1}_{split_name}_{temporal}.npy';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.save(f,s)
    shapes.append({**base,'temporal_mode':temporal,'shape':f'{m["observations"]}x800'})
   gates.append({**base,**gate_stats(g)})
   print(json.dumps({'method':a.method,'seed':a.seed,'split':split_name,'layer':layer,'shape':list(d.shape),'r95':metrics[-2]['r95'],'parity':'passed'}),flush=True)
   if split_name=='basis_eval' and a.method in ('SNN-DFA','matched SNN-BPTT','temporal ANN-DFA'):
    for N in (128,256,512):
     for rep in range(a.robustness_repeats):
      rng_seed=2026090800+rep
      indices=np.random.default_rng(rng_seed).choice(512,N,replace=False);indices.sort()
      m,_,H=spectral(d[:,indices])
      robust.append({**base,'N':N,'T':30,'observations':N*30,'repeat':rep,'subsample_seed':rng_seed,
       'selected_positions':json.dumps(indices.tolist()),'r95':m['r95'],'stable_rank':m['stable_rank'],'entropy_effective_rank':m['entropy_rank'],
       'full_sample_repetition':N==512})
   del d,g
 write_csv(OUT/'01_spectral_metrics'/f'{method}_{a.seed}.csv',metrics)
 write_csv(OUT/'04_gate_statistics'/f'{method}_{a.seed}.csv',gates)
 if robust:write_csv(OUT/'02_sample_robustness'/f'{method}_{a.seed}.csv',robust)
 write_csv(job/'parity.csv',checks);write_csv(job/'shapes.csv',shapes)
 write_json(job/'complete.json',{'status':'complete','wall_seconds':time.time()-start,'feedback':feedback,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'seed':a.seed,'method':a.method})

if __name__=='__main__':main()
