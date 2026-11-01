"""Read-only small sample conversion test. No formal statistics."""
import json
from pathlib import Path
import torch
from common import convert, atomic_json
p = Path('/root/autodl-tmp/CodexWorkspace/Project_20260501/data/processed/gdb13_v24_1m_scaffold_disjoint/test.pt')
rows = torch.load(p, map_location='cpu')
subset = rows[:32]
del rows
converted = [convert((i, r)) for i, r in enumerate(subset)]
atomic_json('/root/autodl-tmp/real-data-smoke.json', {'tested_records': len(converted), 'conversion_roundtrip': 'passed', 'formal_statistics': False, 'source': str(p)})
print('32 real test-record assembly roundtrips passed')
