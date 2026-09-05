import numpy as np
from sklearn.metrics import roc_auc_score

from geometry_llm.representation_geometry import (gram,cka,spectrum,neighbors,overlap,
    label_tests,fdr,auc_bootstrap,predictive_probe,vector_components,grouped_summaries)


def test_shape_and_cka_are_orthogonal_coordinate_invariant():
    rng=np.random.default_rng(8); x=rng.normal(size=(40,7))
    q,_=np.linalg.qr(rng.normal(size=(7,7)))
    assert np.isclose(cka(gram(x),gram(x@q+5)),1)
    assert np.isclose(spectrum(gram(x))['effective_rank'],spectrum(gram(x@q))['effective_rank'])
    assert 1<=spectrum(gram(x))['participation_ratio']<=7.00001


def test_neighbor_queries_exclude_all_same_source_rows():
    x=np.random.default_rng(3).normal(size=(30,5)); sources=np.repeat(np.arange(15),2)
    nn,_=neighbors(gram(x),sources,count=5)
    assert not np.any(sources[nn]==sources[:,None])
    assert np.all(overlap(nn,nn)==1)


def test_relation_restricted_null_does_not_call_relation_only_structure_target_specific():
    labels=np.repeat(np.arange(3),8); x=np.eye(3)[labels]
    k=gram(x); nn,_=neighbors(k,np.arange(24),count=3)
    blocks=[np.flatnonzero(labels==i) for i in range(3)]
    result=label_tests([k],[nn],labels,blocks,permutations=19)[0]
    assert result['exchangeable_rows']==0
    assert result['cka']['p']==1
    assert np.isclose(result['cka']['excess'],0)
    assert np.allclose(fdr([.01,.02,.9]),[.03,.03,.9])


def test_bootstrap_auc_handles_tied_scores():
    y=np.array([0,1,0,1,1,0]); scores=np.array([.1,.3,.3,.8,.8,.8]); groups=np.arange(6)
    draws=auc_bootstrap(y,[scores],groups,samples=20)[0]
    weights=np.random.default_rng(123).multinomial(6,np.ones(6)/6,size=20)
    for got,w in zip(draws,weights):
        if not (w[y==0].sum() and w[y==1].sum()): assert np.isnan(got)
        else: assert np.isclose(got,roc_auc_score(y,scores,sample_weight=w))


def test_behavioral_probe_keeps_groups_in_one_fold():
    rng=np.random.default_rng(10); x0=rng.normal(size=(80,8)); x=x0+.1*rng.normal(size=x0.shape)
    groups=np.repeat(np.arange(40),2); labels=np.tile([0,1],40)
    result=predictive_probe(x,x0,labels,groups,np.arange(80),np.array(['r']*80),np.arange(80)+20)
    assert result['available'] and result['positive_count']==40
    folds=np.array(result['fold_ids'])
    assert all(len(set(folds[groups==g]))==1 for g in set(groups))
    assert all(len(v)==80 for v in result['oof_predictions'].values())


def test_angles_separate_rescaling_rotation_and_zero_update():
    base=np.array([[1.,0.],[1.,0.],[1.,0.]])
    changed=np.array([[2.,0.],[0.,1.],[1.,0.]])
    m=vector_components(changed,base)
    assert np.allclose(m['norm_ratio'],[2,1,1])
    assert np.allclose(m['rotation_deg'],[0,90,0])
    assert np.allclose(m['radial_relative_step'],[1,-1,0])
    assert np.allclose(m['tangential_relative_step'],[0,1,0])
    assert np.allclose(m['update_to_base_angle_deg'][:2],[0,135])
    assert np.isnan(m['update_to_base_angle_deg'][2])
    result=grouped_summaries(m,np.arange(3),samples=30)
    assert result['update_to_base_angle_deg']['defined_n']==2
    assert result['update_to_base_angle_deg']['undefined_n']==1
