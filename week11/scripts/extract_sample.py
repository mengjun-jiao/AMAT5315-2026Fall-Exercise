#!/usr/bin/env python3
import json, subprocess
from pathlib import Path
root=Path(__file__).resolve().parent.parent
run=json.loads((root/'runs/sample_f15/run.json').read_text())['evidence']
sample=next(s for s in run['samples'] if s['energy']==run['e_min'])
result=subprocess.run([str(root/'.install/bin/pred'),'extract',str(root/'runs/pred/factor_15.bundle.json'),
                       '--config',','.join(map(str,sample['bits'])),'--json'],
                      check=True,text=True,capture_output=True)
(root/'runs/sample_f15/extract.json').write_text(result.stdout)
(root/'runs/sample_f15/extract.stderr.txt').write_text(result.stderr)
print(result.stdout)
