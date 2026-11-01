import hashlib, json, os, subprocess, platform, sys
from pathlib import Path

def write_run_manifest(path, *, command=None, data_paths=(), split=None, model_config=None,
                       seed=None, checkpoint_lineage=None, eval_params=None, outputs=(),
                       experiment_metadata=None):
    def digest(p):
        try:
            h=hashlib.sha256();
            with open(p,'rb') as f:
                for chunk in iter(lambda:f.read(1<<20), b''): h.update(chunk)
            return h.hexdigest()
        except OSError: return None
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception: commit=None
    payload={'git_commit':commit,'data_hashes':{str(p):digest(p) for p in data_paths},'split':split,
             'environment':{'python':sys.version,'platform':platform.platform(),'cuda':os.environ.get('CUDA_VISIBLE_DEVICES')},
             'command':command or sys.argv,'model_config':model_config,'seed':seed,
             'checkpoint_lineage':checkpoint_lineage,'evaluation':eval_params,'outputs':[str(x) for x in outputs]}
    if experiment_metadata:
        payload['experiment'] = dict(experiment_metadata)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload
