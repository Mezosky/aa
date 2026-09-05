#!/usr/bin/env python
"""Analyze final-state shape, label organization, and grouped behavioral prediction."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from umap import UMAP

from geometry_llm.commands.audits.audit_composition import read_rows, group_key
from geometry_llm.representation_geometry import (gram,cka,spectrum,neighbors,overlap,cone_angles,
    group_mean_interval,label_tests,fdr,repeated_predictive_probe)


def run_probes(report,data,manifest,chains,root,seed):
    last=data['frozen']['layers'][-1]; ids=[str(c['chain_id']) for c in chains]
    original=data['frozen'][f'direct_final_{last}']
    # A stricter grouping than the descriptive bridge/answer-cluster intervals:
    # no gold answer identity may occur in both training and validation folds.
    groups=np.array([c['e3_id'] or c['e3'] for c in chains])
    sources=np.array([c['e1_id'] or c['e1'] for c in chains]); relations=np.array([c['fact_comp_type'] for c in chains])
    report['probe_protocol']={'group':'gold answer identity','fold_assignments':[123,321,777],
        'nuisance':'relation-pair one-hot and log prompt length',
        'frozen_geometry':'nuisance plus frozen log norm, cone angle, and 10-neighbor radius',
        'combined':'frozen_geometry plus rotation, relative step, log norm ratio, adapted cone angle, radius, and neighbor retention',
        'scope':'localization of errors in one fitted adapter; not causal or across-adapter generalization'}
    pred_paths={'residual':root/'predictions'/f'correct_delta_seed-{seed}.jsonl',
                'lora':Path(manifest['checkpoints']['lora']['path']).parent/'predictions.jsonl',
                'joint':Path(manifest['checkpoints']['joint']['path']).parent/'predictions.jsonl'}
    for condition,path in pred_paths.items():
        lookup={str(r['chain_id']):r for r in read_rows(path)}; assert set(lookup)==set(ids)
        rr=[lookup[i] for i in ids]; report['probes'][condition]={}
        for outcome in ['direct','both_constituents']:
            y=np.array([r['correct_12'] if outcome=='direct' else r['correct_1'] and r['correct_2'] for r in rr],dtype=int)
            probe=repeated_predictive_probe(data[condition][f'direct_final_{last}'],original,y,groups,sources,relations,data[condition]['direct_lengths'])
            report['probes'][condition][outcome]=probe
            print(json.dumps({'probe_complete':condition,'outcome':outcome,'increment':probe.get('incremental_auc')}),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',required=True); parser.add_argument('--seed',type=int,default=13)
    parser.add_argument('--probes-only',action='store_true')
    args=parser.parse_args(); root=Path(args.root); out=root/'final_geometry'/f'seed-{args.seed}'
    manifest=json.loads((out/'manifest.json').read_text())
    chains=read_rows(root/'selected_chains.jsonl'); ids=[str(c['chain_id']) for c in chains]
    groups=np.array([json.dumps(group_key(c)) for c in chains]); sources=np.array([c['e1_id'] or c['e1'] for c in chains])
    relations=np.array([c['fact_comp_type'] for c in chains]); blocks=[np.where(relations==r)[0] for r in np.unique(relations)]
    data={}
    for condition in manifest['conditions']:
        file=out/f'{condition}.npz'
        if not file.exists(): raise RuntimeError(f'Extraction not complete: {file}')
        arrays=np.load(file); assert arrays['chain_ids'].tolist()==ids
        data[condition]=arrays
    if args.probes_only:
        report=json.loads((out/'analysis.json').read_text())
        run_probes(report,data,manifest,chains,root,args.seed)
        (out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        return
    layers=data['frozen']['layers'].tolist(); last=layers[-1]
    original=data['frozen'][f'direct_final_{last}']; k0=gram(original)
    nn0,_=neighbors(k0,sources)
    report={'n':len(ids),'groups':len(set(groups)),'layers':layers,'seed':args.seed,'conditions':{},'label_tests':{},
            'definition':'Covariance-entropy rank uses eigenvalues, not the singular-value convention in older input analyses.',
            'probes':{},'cka':{},'query_alignment':{},'control_note':'Permuted is row reassignment, not shuffled-label training.'}
    finals={}; kernels={}; neighbor_indexes={}
    # Common frozen references are reused exactly for all layerwise comparisons.
    frozen_kernels={(site,l):gram(data['frozen'][f'direct_{site}_{l}']) for site in ['final','entity'] for l in layers}
    for condition,arrays in data.items():
        x=arrays[f'direct_final_{last}']; k=gram(x); finals[condition]=x; kernels[condition]=k
        nn,_=neighbors(k,sources); neighbor_indexes[condition]=nn
        delta=x-original; dk=gram(delta)
        norm=np.linalg.norm(x,axis=1); norm0=np.linalg.norm(original,axis=1)
        rotations=np.degrees(np.arccos(np.clip(np.sum(x*original,axis=1)/(norm*norm0).clip(1e-12),-1,1)))
        metric={'spectrum':spectrum(k),'displacement_spectrum':spectrum(dk),
                'cone':group_mean_interval(cone_angles(k),groups),
                'rotation':group_mean_interval(rotations,groups),
                'neighbor_retention':group_mean_interval(overlap(nn,nn0),groups),
                'cka_to_frozen':cka(k,k0),
                'mean_pair_cosine':float(((x/norm[:,None]).sum(0).dot((x/norm[:,None]).sum(0))-len(x))/(len(x)*(len(x)-1))),
                'common_displacement_energy':float(len(x)*np.sum(delta.mean(0)**2)/np.sum(delta**2)) if np.sum(delta**2)>0 else None,
                'profiles':[]}
        for site in ['final','entity']:
            for l in layers:
                h=arrays[f'direct_{site}_{l}']; base=data['frozen'][f'direct_{site}_{l}']
                denom=np.linalg.norm(h,axis=1)*np.linalg.norm(base,axis=1)
                angles=np.degrees(np.arccos(np.clip(np.sum(h*base,axis=1)/denom.clip(1e-12),-1,1)))
                metric['profiles'].append({'site':site,'layer':l,'cka':cka(gram(h),frozen_kernels[site,l]),
                                           'rotation':group_mean_interval(angles,groups)})
        report['conditions'][condition]=metric
        report['query_alignment'][condition]={q:cka(k,gram(arrays[f'{q}_final_{last}'])) for q in ['hop1','hop2']}
        print(json.dumps({'shape_complete':condition,'rank':metric['spectrum']['effective_rank'],
                          'cone':metric['cone']['mean'],'cka':metric['cka_to_frozen']}),flush=True)
    conditions=list(data)
    report['cka']={'conditions':conditions,'matrix':[[cka(kernels[a],kernels[b]) for b in conditions] for a in conditions]}
    delta_conditions=[c for c in conditions if c!='frozen']
    delta_grams={c:gram(finals[c]-original) for c in delta_conditions}
    report['delta_cka']={'conditions':delta_conditions,'matrix':[[cka(delta_grams[a],delta_grams[b]) for b in delta_conditions] for a in delta_conditions]}
    for label,values in [('relation',relations),('bridge',[c['e2_id'] or c['e2'] for c in chains]),
                          ('answer',[c['e3_id'] or c['e3'] for c in chains])]:
        tests=label_tests(list(kernels.values()),list(neighbor_indexes.values()),values,
                          blocks=None if label=='relation' else blocks)
        report['label_tests'][label]=dict(zip(conditions,tests))
        print(json.dumps({'label_tests_complete':label}),flush=True)
    # One family per model: all condition/label tests, separately by statistic.
    for statistic in ['cka','purity']:
        tests=[r[statistic] for values in report['label_tests'].values() for r in values.values()]
        for r,q in zip(tests,fdr([r['p'] for r in tests])): r['q']=q
    run_probes(report,data,manifest,chains,root,args.seed)
    (out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    # Shared coordinates within each model. No geometry or prediction analysis uses UMAP.
    main_conditions=['frozen','residual','lora','joint']
    cloud=np.concatenate([finals[c]/np.linalg.norm(finals[c],axis=1,keepdims=True) for c in main_conditions])
    pca=PCA(n_components=50,svd_solver='randomized',random_state=123)
    projected=pca.fit_transform(cloud)
    maps={}; checks={}
    for seed in [123,321]:
        embedding=UMAP(n_neighbors=20,min_dist=.15,metric='cosine',random_state=seed,n_jobs=1).fit_transform(projected)
        maps[f'umap_{seed}']=embedding
        checks[str(seed)]={c:float(trustworthiness(cloud[i*len(ids):(i+1)*len(ids)],embedding[i*len(ids):(i+1)*len(ids)],
                                                 n_neighbors=10,metric='cosine')) for i,c in enumerate(main_conditions)}
    np.savez_compressed(out/'shared_maps.npz',conditions=np.array(main_conditions),chain_ids=np.array(ids),**maps)
    report['umap']={'n_neighbors':20,'min_dist':.15,'metric':'cosine after joint 50-PC projection of unit vectors',
                    'pca_variance_retained':float(pca.explained_variance_ratio_.sum()),
                    'trustworthiness_k10_original_cosine':checks,
                    'scope':'shared coordinates within a model; visualization only, not a manifold or mechanism test'}
    (out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(f'Complete: {out}',flush=True)


if __name__=='__main__': main()
