"""Small deterministic manifests; full artifacts stay under the remote run root."""
import gzip
import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.pending')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')


def freeze(path, value):
    path = Path(path)
    if path.exists() and read_json(path) != value:
        raise ValueError('Frozen contract differs: ' + str(path))
    if not path.exists():
        atomic_json(path, value)
    return value


def verify_generation_chain(run):
    run=Path(run)
    generation=read_json(run/'generation.json');scoring=read_json(run/'scoring.json')
    raw=file_sha(run/'samples.jsonl.gz');scored=file_sha(run/'scored.jsonl.gz')
    if generation['samples_sha256']!=raw or scoring['input_samples_sha256']!=raw or scoring['scored_sha256']!=scored:
        raise ValueError('Generation/scoring evidence chain differs')
    return {'input_samples_sha256':raw,'scored_sha256':scored,'checkpoint_sha256':generation['checkpoint_sha256']}
