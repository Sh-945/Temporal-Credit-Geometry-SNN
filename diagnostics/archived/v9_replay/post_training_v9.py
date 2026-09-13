"""Wait for isolated training completion, then run diagnostic jobs sequentially."""
from pathlib import Path
import os,time,subprocess,sys,json
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
from replay_v9 import METHODS,write_json

def main():
 pid=int(sys.argv[1]);done=OUT/'03_five_seed/new_training_complete.json'
 while not done.exists():
  try:os.kill(pid,0)
  except ProcessLookupError:
   write_json(OUT/'logs/post_training_BLOCKED.json',{'reason':'training process exited without full completion marker'});raise RuntimeError('Incomplete training')
  time.sleep(30)
 for seed in (20260908,20260909):
  for display,info in METHODS.items():
   cp=OUT/'03_five_seed/checkpoints'/info[1]/f'seed_{seed}'/'epoch_100.pt'
   subprocess.run([sys.executable,'-u',str(OUT/'replay_v9.py'),'--method',display,'--seed',str(seed),'--checkpoint',str(cp)],check=True)
 for script in ('trajectory_v9.py','dvs_gate_v9.py','collect_accuracy_v9.py'):
  subprocess.run([sys.executable,'-u',str(OUT/script)],check=True)
 subprocess.run([sys.executable,str(OUT/'summarize_replay.py'),'--seed-count','5'],check=True)
 write_json(OUT/'logs/post_training_complete.json',{'status':'real_replay_complete_not_final_freeze','pending':['accuracy consolidation','figures','optional state smoke','final fact report and canonical freeze']})

if __name__=='__main__':main()
