"""Collect archived and new validation-selected/final accuracy with explicit rules."""
from replay_v9 import *
import pandas as pd

def main():
 rows=[]
 for display,info in METHODS.items():
  source=FORMAL/info[0]/'test_results.csv';old=pd.read_csv(source)
  for seed in (*SEEDS,20260908,20260909):
   base={'dataset':'N-MNIST','method':display,'seed':seed,'git_commit':'UNAVAILABLE'}
   if seed in SEEDS:
    group=old[(old.method==info[1])&(old.seed==seed)]
    for role,rule in [('final','completed_epoch_100'),('best_validation','validation_accuracy_max_tie_lower_loss')]:
     g=group[group.checkpoint==role];assert len(g)==1
     r=g.iloc[0]
     rows.append({**base,'checkpoint_rule':rule,'test_accuracy':float(r.test_accuracy),'test_loss':float(r.test_loss),'best_validation_epoch':int(r.best_validation_epoch),'source_file':str(source.relative_to(ROOT))})
   else:
    source_new=OUT/'03_five_seed/checkpoints'/info[1]/f'seed_{seed}'/'variant_summary.json';r=json.loads(source_new.read_text())
    for key,rule in [('final_test','completed_epoch_100'),('best_validation_test','validation_accuracy_max_tie_lower_loss')]:
     rows.append({**base,'checkpoint_rule':rule,'test_accuracy':r[key]['accuracy'],'test_loss':r[key]['loss'],'best_validation_epoch':r['best_validation_epoch'],'source_file':str(source_new.relative_to(ROOT))})
 assert len(rows)==50
 write_csv(OUT/'03_five_seed/all_runs.csv',rows)
 result=[]
 df=pd.DataFrame(rows)
 for (method,rule),g in df.groupby(['method','checkpoint_rule']):
  assert g.seed.nunique()==5
  result.append({'dataset':'N-MNIST','method':method,'checkpoint_rule':rule,'seed_count':5,'metric':'test_accuracy','mean':g.test_accuracy.mean(),'std':g.test_accuracy.std(ddof=1),'units':'fraction','source_file':';'.join(sorted(g.source_file.unique())),'git_commit':'UNAVAILABLE'})
 write_csv(OUT/'03_five_seed/accuracy_summary.csv',result)

if __name__=='__main__':main()
