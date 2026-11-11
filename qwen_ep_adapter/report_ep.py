import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);a=p.parse_args()
rows=[]
for method in ('fifo','priority'):
 f=a.folder/f'{method}.json'
 if f.exists():rows.append(json.loads(f.read_text()))
if len(rows)!=2:raise SystemExit('Missing valid runs: report not generated')
fields=['model','method','mean_ms','p95_ms','throughput','world_size','top_k','validation_status']
with (a.folder/'metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
b,our=rows
lines=['# Qwen 4 GPU EP Prototype Results','',
 '**This is NOT the DeepSpeed vs DSCA-MoE comparison.** Both methods use custom torch.distributed expert parallel; priority is only an expert-order prototype.',
 '','| Mode | Throughput (samples/s) | Mean (ms) | P95 (ms) |','|---|---:|---:|---:|']
for r in rows:lines.append(f"| {r['method']} | {r['throughput']:.2f} | {r['mean_ms']:.2f} | {r['p95_ms']:.2f} |")
lines+=['',f"Priority vs FIFO throughput change: {(our['throughput']/b['throughput']-1)*100:+.2f}%",'','The model-wide semantic equivalence has not yet been checked. The included tiny-layer distributed test is necessary but insufficient for publication-grade validation.']
(a.folder/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
