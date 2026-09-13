"""Replay real DFA scheduled and validation-selected checkpoints, no training."""
from replay_v9 import *
from replay_v9 import _prepare, _capture_snn_dfa_layer
import pandas as pd

def main():
 torch.set_num_threads(4)
 rows=[];aliases=[]
 for seed in (*SEEDS,20260908,20260909):
  base=(LEGACY/'checkpoints/full_dense' if seed in SEEDS else OUT/'03_five_seed/checkpoints/dfa_trained')/f'seed_{seed}'
  candidates=[('scheduled',base/('init.pt' if e==0 else f'epoch_{e:03d}.pt')) for e in (0,10,25,50,75,100)]
  candidates.append(('validation_selected_best',base/'best_validation.pt'))
  seen={}
  for role,cp in candidates:
   state=torch.load(cp,map_location='cpu',weights_only=False);epoch=int(state['epoch'])+1
   digest=hashlib.sha256()
   for name,v in sorted(state['model_state'].items()):digest.update(name.encode());digest.update(v.numpy().tobytes())
   identity=digest.hexdigest()
   aliases.append({'seed':seed,'epoch':epoch,'role':role,'checkpoint':str(cp.relative_to(ROOT)),'model_sha256':identity,'duplicate':identity in seen})
   if identity in seen:
    assert seen[identity]==epoch,'identical model at different completed epochs: audit required'
    continue
   seen[identity]=epoch
   config=state['config'];model=build_model(config).cuda();model.load_state_dict(state['model_state']);model.eval()
   bank=FeedbackBank(model,config).cuda();bank.load_state_dict(state['feedback_state']);bank.eval()
   train,test=build_datasets(config)
   split=build_paper1_split(len(train),split_seed=20260830,probe_seed=20260831,validation_fraction=.1,probe_size=1024)
   with (FORMAL/'diagnostic_probe.csv').open() as f:archived=list(csv.DictReader(f))
   assert split.probe_indices.tolist()==[int(r['dataset_index']) for r in archived]
   loader=make_loaders(train,test,split,config,seed=seed)['basis_eval']
   ds=[[] for _ in range(3)];gs=[[] for _ in range(3)];qs=[[] for _ in range(3)]
   for batch in loader:
    x,y,idx=_prepare(batch,torch.device('cuda'))
    with torch.no_grad():
     outputs=model.forward_with_cache(x,detach_temporal=True)
     e=outputs[0].mean(0)-torch.nn.functional.one_hot(y,10).to(outputs[0].dtype)
    for i,inp in enumerate(outputs[1]):
     c=_capture_snn_dfa_layer(model,bank,i,inp,e,gate_mode='actual',seed=seed,epoch=epoch,sample_indices=idx)
     ds[i].append(c['delta'].detach().cpu());gs[i].append(c['applied_gate'].detach().cpu());qs[i].append(c['q'].detach().cpu())
   for i in range(3):
    d=torch.cat(ds[i],1);g=torch.cat(gs[i],1);q=torch.cat(qs[i],0)
    assert tuple(d.shape)==(30,512,800)
    base_row={'dataset':'N-MNIST','method':'SNN-DFA','seed':seed,'layer':f'H{i+1}','epoch':epoch,'checkpoint_role':role,'checkpoint':str(cp.relative_to(ROOT)),'checkpoint_sha256':sha(cp),'model_sha256':identity,'probe_split':'basis_eval','width':800}
    gate=gate_stats(g)
    for label,v in [('pre-gate',q),('time-collapsed',d.mean(0)),('timestep',d)]:
     m,_,H=spectral(v)
     rows.append({**base_row,'signal':label,**{k:m[k] for k in ('r50','r80','r90','r95','r99','stable_rank')},'r95_over_D':m['r95']/800,'entropy_effective_rank':m['entropy_rank'],**gate})
   print(json.dumps({'seed':seed,'epoch':epoch,'role':role,'status':'replayed'}),flush=True)
   del model,bank,ds,gs,qs;torch.cuda.empty_cache()
 write_csv(OUT/'03_five_seed/trajectory_per_checkpoint.csv',rows)
 write_csv(OUT/'03_five_seed/checkpoint_observation_manifest.csv',aliases)
 correlations=[]
 for signal in ('timestep','time-collapsed'):
  r=[x for x in rows if x['signal']==signal]
  groups=[('pooled','all',r)]+[('layer',l,[x for x in r if x['layer']==l]) for l in ('H1','H2','H3')]+[('seed',str(s),[x for x in r if x['seed']==s]) for s in (*SEEDS,20260908,20260909)]
  for kind,name,g in groups:
   # Spearman is Pearson correlation of average ranks; ties get their mean rank.
   xr=pd.Series([x['temporal_gate_cosine'] for x in g]).rank(method='average').to_numpy()
   yr=pd.Series([x['r95_over_D'] for x in g]).rank(method='average').to_numpy()
   rho=float(np.corrcoef(xr,yr)[0,1])
   correlations.append({'group_type':kind,'group':name,'signal':signal,'rho':rho,'n':len(g),'seed_count':len(set(x['seed'] for x in g)),'x':'temporal_gate_cosine','y':'r95_over_D','probe_split':'basis_eval','checkpoint_rule':'scheduled_0_10_25_50_75_100_plus_validation_best_deduplicated_by_model_identity','p_value':'not_reported'})
 write_csv(OUT/'03_five_seed/correlation_breakdown.csv',correlations)

if __name__=='__main__':main()
