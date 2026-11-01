"""Stage-3 lightweight artifact checksum verification; no chemistry recompute."""
import argparse
import json
from pathlib import Path
from common import sha256
p = argparse.ArgumentParser()
p.add_argument('bundle', type=Path)
a = p.parse_args()
state = json.loads((a.bundle / 'state.json').read_text())
if state['status'] != 'COMPLETE':
    raise SystemExit(f'Formal stage 2 not complete: {state}')
for name, digest in json.loads((a.bundle / 'artifacts.json').read_text()).items():
    path = (a.bundle / name).resolve()
    if not path.is_relative_to(a.bundle.resolve()) or sha256(path) != digest:
        raise SystemExit(f'Artifact validation failed: {name}')
print('Bundle checksums verified. No molecular metrics recomputed.')
