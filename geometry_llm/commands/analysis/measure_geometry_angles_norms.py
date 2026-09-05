#!/usr/bin/env python
"""Measure absolute norms, rotations, and radial/tangential change without new inference."""
import argparse
import json
from pathlib import Path

import numpy as np

from geometry_llm.commands.audits.audit_composition import read_rows,group_key
from geometry_llm.representation_geometry import vector_components,grouped_summaries


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--root',required=True)
    p.add_argument('--seed',type=int,default=13); args=p.parse_args()
    root=Path(args.root); out=root/'final_geometry'/f'seed-{args.seed}'
    manifest=json.loads((out/'manifest.json').read_text()); chains=read_rows(root/'selected_chains.jsonl')
    groups=np.array([json.dumps(group_key(c)) for c in chains])
    original=np.load(out/'frozen.npz'); layers=original['layers'].tolist(); last=layers[-1]
    report={'n':len(chains),'layers':layers,'conditions':{},
            'normalization':'Absolute L2 norms; relative quantities divide by the paired frozen norm.',
            'angle_policy':'Zero displacement has no update direction; its update-to-base angle is undefined, not zero.',
            'site_policy':'Final prompt token and designated source-span mean, both at the final normalized layer.'}
    raw={}
    for condition in manifest['conditions']:
        data=np.load(out/f'{condition}.npz'); assert np.array_equal(data['chain_ids'],original['chain_ids'])
        report['conditions'][condition]={}
        for site in ['final','entity']:
            values=vector_components(data[f'direct_{site}_{last}'],original[f'direct_{site}_{last}'])
            for name,arr in values.items(): raw[f'{condition}_{site}_{name}']=arr
            for l in layers:
                a=vector_components(data[f'direct_{site}_{l}'],original[f'direct_{site}_{l}'])
                for name in ['state_norm','norm_ratio','relative_step']:
                    values[f'layer_{l}_{name}']=a[name]
            report['conditions'][condition][site]=grouped_summaries(values,groups)
            # Exact per-example identity verifies that the two components exhaust the update.
            r=values['radial_relative_step']; t=values['tangential_relative_step']
            assert np.allclose(values['norm_ratio']**2,(1+r)**2+t**2,atol=1e-10)
        print(json.dumps({'norms_complete':condition}),flush=True)
    conditions=['residual','lora','joint','random','permuted']
    base=original[f'direct_final_{last}'].astype(float)
    displacements={c:np.load(out/f'{c}.npz')[f'direct_final_{last}'].astype(float)-base for c in conditions}
    pairs={}
    for i,a in enumerate(conditions):
        for j,b in enumerate(conditions):
            left,right=displacements[a],displacements[b]
            denom=np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1); valid=denom>1e-12
            angle=np.full(len(chains),np.nan)
            angle[valid]=np.degrees(np.arccos(np.clip(np.sum(left[valid]*right[valid],axis=1)/denom[valid],-1,1)))
            if a==b: angle[valid]=0.
            pairs[f'{i}_{j}']=angle
    summaries=grouped_summaries(pairs,groups)
    report['displacement_angles']={'conditions':conditions,
        'matrix':[[summaries[f'{i}_{j}']['mean'] for j in range(len(conditions))] for i in range(len(conditions))],
        'ci95':[[summaries[f'{i}_{j}']['ci95'] for j in range(len(conditions))] for i in range(len(conditions))],
        'defined_n':[[summaries[f'{i}_{j}']['defined_n'] for j in range(len(conditions))] for i in range(len(conditions))]}
    raw.update({f'displacement_angle_{k}':v for k,v in pairs.items()})
    np.savez_compressed(out/'angles_norms_raw.npz',chain_ids=original['chain_ids'],**raw)
    (out/'angles_norms.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__': main()
