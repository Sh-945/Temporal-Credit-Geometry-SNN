"""DVS boundary replay. Spatially projected diagnostic and raw gate stats are separate."""
from replay_v9 import *
from replay_v9 import _prepare,_capture_snn_dfa_layer
from analysis.paper1_geometry.metrics import feature_signal

def main():
 torch.set_num_threads(4);rows=[];checks=[]
 stage=FORMAL/'stageD_cross_dataset'
 with (stage/'geometry.csv').open() as f:historical=list(csv.DictReader(f))
 for seed in SEEDS:
  cp=FORMAL/'checkpoints/stageD_cross_dataset/dvs_dfa'/f'seed_{seed}'/'epoch_100.pt'
  state=torch.load(cp,map_location='cpu',weights_only=False);assert state['epoch']==99
  c=state['config'];model=build_model(c).cuda();model.load_state_dict(state['model_state']);model.eval()
  bank=FeedbackBank(model,c).cuda();bank.load_state_dict(state['feedback_state']);bank.eval()
  train,test=build_datasets(c);protocol=c['experiment03']
  split=build_paper1_split(len(train),split_seed=protocol['split_seed'],probe_seed=protocol['probe_seed'],validation_fraction=protocol['validation_fraction'],probe_size=protocol['probe_size'])
  with (stage/'diagnostic_probe.csv').open() as f:probe=list(csv.DictReader(f))
  assert split.probe_indices.tolist()==[int(r['dataset_index']) for r in probe]
  loader=make_loaders(train,test,split,c,seed=seed)['basis_eval']
  acc=[OnlineCovariance(int(s['dimension'])) for s in model.hidden_specs]
  qacc=[OnlineCovariance(int(s['dimension'])) for s in model.hidden_specs]
  projected=[[] for _ in acc];raw=[[] for _ in acc];shapes={}
  for batch in loader:
   x,y,idx=_prepare(batch,torch.device('cuda'))
   with torch.no_grad():
    outputs=model.forward_with_cache(x,detach_temporal=True)
    e=outputs[0].mean(0)-torch.nn.functional.one_hot(y,model.num_classes).to(outputs[0].dtype)
   for i,inp in enumerate(outputs[1]):
    captured=_capture_snn_dfa_layer(model,bank,i,inp,e,gate_mode='actual',seed=seed,epoch=100,sample_indices=idx)
    g=captured['gate'];d=feature_signal(captured['delta']);q=captured['q']
    acc[i].update(d);qacc[i].update(q.unsqueeze(0).expand(x.shape[0],-1,-1))
    projected[i].append(feature_signal(g).cpu())
    stats=gate_stats(g.flatten(2).cpu());stats['batch_samples']=len(y);raw[i].append(stats)
    shapes[i]=list(g.shape);del captured,g,d,q
  for i,a in enumerate(acc):
   width=int(model.hidden_specs[i]['dimension']);m,_,_,_=spectrum_statistics(a,algebraic_max_dim=width,device=torch.device('cuda'))
   qm,_,_,_=spectrum_statistics(qacc[i],algebraic_max_dim=width,device=torch.device('cuda'))
   ref=[r for r in historical if r['method']=='dvs_dfa' and int(r['seed'])==seed and int(r['epoch'])==100 and r['layer']==f'hidden_{i+1}' and r['signal_type']=='dfa_actual' and r['probe_split']=='basis_eval' and r['residualization']=='raw' and r['temporal_mode']=='timestep']
   assert len(ref)==1
   good=all(m[k]==int(ref[0][k]) for k in ('r50','r80','r90','r95','r99'))
   checks.append({'seed':seed,'layer':f'H{i+1}','archived_r95':ref[0]['r95'],'replay_r95':m['r95'],'rank_thresholds_exact':good})
   if not good:
    write_json(OUT/'logs/dvs_BLOCKED.json',{'reason':'DVS historical geometry parity failed','checks':checks});raise RuntimeError('DVS parity failed')
   common={'dataset':'DVS-Gesture','method':'SNN-DFA','seed':seed,'layer':f'H{i+1}','checkpoint':str(cp.relative_to(ROOT)),'checkpoint_sha256':sha(cp),'checkpoint_rule':'completed_epoch_100','probe_split':'basis_eval','N':len(split.basis_eval_indices),'T':30,'width':width,'credit_r95':m['r95'],'q_r95':qm['r95'],'GE':m['r95']/qm['r95'],'source_raw_gate_batch_shape':str(shapes[i]),'git_commit':'UNAVAILABLE'}
   rows.append({**common,'gate_space':'canonical_spatial_mean_channels',**gate_stats(torch.cat(projected[i],1))})
   # All batch sizes are equal in the canonical DVS evaluation split (448/16).
   n=sum(r['batch_samples'] for r in raw[i]);merged={}
   for k,v in raw[i][0].items():
    if isinstance(v,(float,int)) and k!='batch_samples':merged[k]=sum(r[k]*r['batch_samples'] for r in raw[i])/n
    elif isinstance(v,str):merged[k]=v
   merged['l2_norm_all_T_N_D']=math.sqrt(sum(r['l2_norm_all_T_N_D']**2 for r in raw[i]))
   merged['rms_g']=math.sqrt(sum(r['rms_g']**2*r['batch_samples'] for r in raw[i])/n)
   rows.append({**common,'gate_space':'raw_neuron_and_spatial_features',**merged})
  print(json.dumps({'dataset':'DVS-Gesture','seed':seed,'parity':'passed'}),flush=True)
  del model,bank,projected;torch.cuda.empty_cache()
 write_csv(OUT/'04_gate_statistics/dvs_per_seed.csv',rows);write_csv(OUT/'logs/dvs_parity.csv',checks)

if __name__=='__main__':main()
