#!/usr/bin/env python3
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("metrics",nargs="?",default="data/results/evaluation/metrics.json"); a=p.parse_args()
path=Path(a.metrics)
if not path.exists(): raise SystemExit(f"No measured metrics at {path}")
m=json.loads(path.read_text()); Path("data/results/REPORT.md").write_text("# Evaluation report\n\n```json\n"+json.dumps(m,indent=2)+"\n```\n",encoding="utf8"); print("Wrote data/results/REPORT.md")

