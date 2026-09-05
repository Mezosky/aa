#!/usr/bin/env python
"""Audit unvisited residual rows and check geometry on the common active subset."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from geometry_llm.commands.audits.audit_composition import read_rows
from geometry_llm.data import load_saved_chains
from geometry_llm.representation_geometry import gram,cka,spectrum,cone_angles,repeated_predictive_probe
from geometry_llm.commands.training.train_delta import records,balanced_epoch


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--root',required=True); args=p.parse_args()
    root=Path(args.root); out=root/'final_geometry/seed-13'
    manifest=json.loads((out/'manifest.json').read_text())
    checkpoint=torch.load(manifest['checkpoints']['residual']['path'],map_location='cpu',weights_only=False)
    norms={k:float(v.norm()) for k,v in zip(checkpoint['keys'],checkpoint['state_dict']['delta'])}
    chains=load_saved_chains(root/'selected_chains.jsonl')
    active=np.array([norms['entity:'+c.e1]>0 for c in chains])
    population=records(chains,False,13)
    visited={r['entity'] for epoch in range(checkpoint['metadata']['epochs']) for r in balanced_epoch(population,13+epoch)}
    zero_ids=[c.chain_id for c,a in zip(chains,active) if not a]
    assert all(c.e1 not in visited for c,a in zip(chains,active) if not a)
    arrays={c:np.load(out/f'{c}.npz') for c in manifest['conditions']}
    last=arrays['frozen']['layers'][-1]; original=arrays['frozen'][f'direct_final_{last}'][active]
    report={'all_n':len(chains),'active_n':int(active.sum()),'zero_n':int((~active).sum()),
            'zero_chain_ids':zero_ids,'reason':'Source entities never sampled during the selected eight-epoch replacement-sampling refit.',
            'scope':'Same active-residual-source subset for every compared interface; no retraining.',
            'shape':{},'residual_probes':{}}
    for condition,arr in arrays.items():
        x=arr[f'direct_final_{last}'][active]; k=gram(x)
        report['shape'][condition]={'effective_rank':spectrum(k)['effective_rank'],
                                   'cone_mean':float(cone_angles(k).mean()),'cka_to_frozen':cka(k,gram(original))}
    lookup={str(r['chain_id']):r for r in read_rows(root/'predictions/correct_delta_seed-13.jsonl')}
    selected=[c for c,a in zip(chains,active) if a]
    rr=[lookup[str(c.chain_id)] for c in selected]
    for outcome in ['direct','both_constituents']:
        y=np.array([r['correct_12'] if outcome=='direct' else r['correct_1'] and r['correct_2'] for r in rr],dtype=int)
        result=repeated_predictive_probe(arrays['residual'][f'direct_final_{last}'][active],original,y,
            np.array([c.e3_id or c.e3 for c in selected]),np.array([c.e1_id or c.e1 for c in selected]),
            np.array([c.fact_comp_type for c in selected]),arrays['residual']['direct_lengths'][active])
        report['residual_probes'][outcome]=result
    (out/'active_subset.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'active_n':report['active_n'],'shape':report['shape'],
                      'probes':{k:v['incremental_auc'] for k,v in report['residual_probes'].items()}}),flush=True)


if __name__=='__main__': main()
