"""Resource estimates use measured rates; hardware predictions are never benchmarks."""
import math


def estimate(profiles,jobs,gpu_counts=(1,4),margin=.25,protocol=None):
    minimum=.1 if protocol in ('supplement_v2','supplement_A_v1') else .25
    if margin<minimum:
        raise ValueError('Resource margin below protocol minimum')
    result=[];gpu_seconds=cpu_seconds=storage=0.;max_vram=max_ram=0.
    for job in jobs:
        p=profiles[job['profile_id']]
        for k in ('seconds_per_update','cpu_core_seconds_per_update','peak_vram_gib','peak_ram_gib','bytes_per_update'):
            if k not in p or not math.isfinite(p[k]) or p[k]<0:
                raise ValueError('Missing/invalid measured profile field: '+k)
        n=job['updates']*job.get('repeats',1)
        seconds=n*p['seconds_per_update']
        gpu_seconds+=seconds*p.get('gpu_count',1)
        cpu_seconds+=n*p['cpu_core_seconds_per_update']
        storage+=n*p['bytes_per_update']+job.get('fixed_bytes',0)
        max_vram=max(max_vram,p['peak_vram_gib']);max_ram=max(max_ram,p['peak_ram_gib'])
        result.append({'job':job['id'],'updates':n,'measured_rate_seconds':p['seconds_per_update'],'estimated_seconds':seconds})
    return {'basis':'stage1_measured_profiles','jobs':result,'margin':margin,
            'gpu_hours':gpu_seconds/3600,'gpu_hours_with_margin':gpu_seconds/3600*(1+margin),
            'cpu_core_hours':cpu_seconds/3600*(1+margin),'peak_vram_gib':max_vram,'peak_ram_gib':max_ram*(1+margin),
            'storage_bytes':math.ceil(storage*(1+margin)),
            'ideal_parallel_wall_hours':{str(g):gpu_seconds/3600/g*(1+margin) for g in gpu_counts},
            'limitations':['Parallel wall time is a lower-bound scheduling estimate; CPU scoring, dependencies, PCIe and NFS can dominate.',
                          'V100 memory capacity and actual allocation must be measured, not inferred from the GPU name.']}
