"""Sample-space geometry and leakage-aware behavioral probes."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def gram(x):
    x=np.asarray(x,dtype=np.float64)
    return x@x.T


def center(k):
    return k-k.mean(0,keepdims=True)-k.mean(1,keepdims=True)+k.mean()


def cka(k,l):
    a,b=center(k),center(l)
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/den) if den>1e-12 else None


def spectrum(k):
    values=np.maximum(np.linalg.eigvalsh(center(k))[::-1],0)
    if values.sum()<1e-12:
        return {'effective_rank':0.,'participation_ratio':0.,'rank90':0,'cumulative':[]}
    p=values/values.sum(); nz=p>1e-12
    return {'effective_rank':float(np.exp(-np.sum(p[nz]*np.log(p[nz])))),
            'participation_ratio':float(1/np.sum(p*p)),
            'rank90':int(np.searchsorted(np.cumsum(p),.9)+1),'cumulative':np.cumsum(p).tolist()}


def centered_cosine(k,reference=None):
    reference=np.arange(len(k)) if reference is None else np.asarray(reference)
    mean=k[:,reference].mean(1); grand=k[np.ix_(reference,reference)].mean()
    centered=k-mean[:,None]-mean[None,:]+grand
    norm=np.sqrt(np.maximum(np.diag(centered),1e-20))
    return np.clip(centered/(norm[:,None]*norm[None,:]),-1,1)


def neighbors(k,sources,reference=None,count=10):
    reference=np.arange(len(k)) if reference is None else np.asarray(reference)
    sim=centered_cosine(k,reference)[:,reference]
    mask=np.asarray(sources)[:,None]==np.asarray(sources)[reference][None,:]
    sim=np.where(mask,-np.inf,sim)
    if np.any(np.isfinite(sim).sum(1)<count): raise ValueError('Insufficient distinct-source neighbors')
    selected=np.argpartition(-sim,count-1,axis=1)[:,:count]
    radius=1-np.take_along_axis(sim,selected,axis=1).min(1)
    return reference[selected],radius


def overlap(a,b):
    return (a[:,:,None]==b[:,None,:]).any(2).mean(1)


def cone_angles(k,reference=None):
    reference=np.arange(len(k)) if reference is None else np.asarray(reference)
    dot=k[:,reference].mean(1); axis_sq=k[np.ix_(reference,reference)].mean()
    cosine=dot/np.sqrt(np.maximum(np.diag(k)*axis_sq,1e-20))
    return np.degrees(np.arccos(np.clip(cosine,-1,1)))


def vector_components(x,x0):
    """Euclidean change decomposed along and orthogonal to each frozen vector."""
    x=np.asarray(x,dtype=np.float64); x0=np.asarray(x0,dtype=np.float64)
    norm=np.linalg.norm(x,axis=1); norm0=np.linalg.norm(x0,axis=1)
    if np.any(norm0<=1e-12) or np.any(norm<=1e-12): raise ValueError('State angles require nonzero vectors')
    delta=x-x0; update=np.linalg.norm(delta,axis=1); step=update/norm0
    radial=np.sum(delta*x0,axis=1)/norm0**2
    tangential=np.sqrt(np.maximum(step**2-radial**2,0))
    rotation=np.degrees(np.arccos(np.clip(np.sum(x*x0,axis=1)/(norm*norm0),-1,1)))
    rotation[update<=1e-12]=0
    update_angle=np.full(len(x),np.nan); valid=update>1e-12
    update_angle[valid]=np.degrees(np.arccos(np.clip(radial[valid]/step[valid],-1,1)))
    return {'state_norm':norm,'norm_ratio':norm/norm0,'update_norm':update,'relative_step':step,
            'rotation_deg':rotation,'update_to_base_angle_deg':update_angle,
            'radial_relative_step':radial,'tangential_relative_step':tangential}


def grouped_summaries(values,groups,samples=2000):
    """Many paired statistics with one common cluster-bootstrap draw matrix."""
    names=list(values); matrix=np.column_stack([values[k] for k in names]); valid=np.isfinite(matrix)
    _,ids=np.unique(groups,return_inverse=True); ng=ids.max()+1
    totals=np.zeros((ng,len(names))); counts=np.zeros_like(totals)
    np.add.at(totals,ids,np.nan_to_num(matrix)); np.add.at(counts,ids,valid)
    weights=np.random.default_rng(123).multinomial(ng,np.ones(ng)/ng,size=samples)
    denominators=weights@counts
    draws=np.divide(weights@totals,denominators,out=np.full_like(denominators,np.nan),where=denominators>0)
    result={}
    for j,name in enumerate(names):
        good=matrix[valid[:,j],j]
        result[name]={'mean':float(good.mean()) if len(good) else None,
                      'ci95':np.nanquantile(draws[:,j],[.025,.975]).tolist() if len(good) else [None,None],
                      'q10_median_q90':np.quantile(good,[.1,.5,.9]).tolist() if len(good) else [None,None,None],
                      'defined_n':len(good),'undefined_n':len(matrix)-len(good)}
    return result


def group_mean_interval(values,groups,samples=2000):
    values=np.asarray(values); _,ids=np.unique(groups,return_inverse=True)
    counts=np.bincount(ids); sums=np.bincount(ids,weights=values)
    weights=np.random.default_rng(123).multinomial(len(counts),np.ones(len(counts))/len(counts),size=samples)
    draws=(weights@sums)/(weights@counts)
    return {'mean':float(values.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist()}


def label_tests(kernels,neighbor_indices,labels,blocks=None,permutations=999):
    """Bias-calibrate label CKA and neighborhood purity with matched permutations."""
    labels=np.asarray(labels); kinds,encoded=np.unique(labels,return_inverse=True)
    lk=(encoded[:,None]==encoded[None,:]).astype(float)
    denom=np.linalg.norm(center(lk))
    normalized=np.stack([center(k)/max(np.linalg.norm(center(k))*denom,1e-20) for k in kernels])
    def values(y):
        same=(y[:,None]==y[None,:])
        alignment=np.einsum('cij,ij->c',normalized,same,optimize=False)
        purity=np.array([(y[idx]==y[:,None]).mean() for idx in neighbor_indices])
        return alignment,purity
    observed=values(encoded); rng=np.random.default_rng(123)
    blocks=[np.arange(len(encoded))] if blocks is None else blocks
    draws=[[],[]]
    for _ in range(permutations):
        perm=np.arange(len(encoded))
        for block in blocks: perm[block]=rng.permutation(block)
        a,b=values(encoded[perm]); draws[0].append(a); draws[1].append(b)
    reports=[]
    for i in range(len(kernels)):
        report={'n_classes':len(kinds),'exchangeable_rows':sum(len(b) for b in blocks if len(np.unique(encoded[b]))>1)}
        for name,obs,raw in zip(['cka','purity'],observed,draws):
            null=np.array(raw)[:,i]
            report[name]={'observed':float(obs[i]),'null_mean':float(null.mean()),
                          'excess':float(obs[i]-null.mean()),'null95':np.quantile(null,[.025,.975]).tolist(),
                          'p':float((1+np.sum(null>=obs[i]-1e-12))/(permutations+1))}
        reports.append(report)
    return reports


def fdr(pvalues):
    p=np.asarray(pvalues); order=np.argsort(p)
    adjusted=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    result=np.empty(len(p)); result[order]=np.minimum(adjusted,1)
    return result.tolist()


def auc_bootstrap(y,scores,groups,samples=2000):
    """Group bootstrap AUC, retaining score ties and fixed out-of-fold predictions."""
    _,group_ids=np.unique(groups,return_inverse=True); n_groups=group_ids.max()+1
    weights=np.random.default_rng(123).multinomial(n_groups,np.ones(n_groups)/n_groups,size=samples)[:,group_ids]
    output=[]
    for score in scores:
        order=np.argsort(score); starts=np.r_[0,np.flatnonzero(np.diff(score[order]))+1]
        pos=np.add.reduceat(weights[:,order]*y[order],starts,axis=1)
        neg=np.add.reduceat(weights[:,order]*(1-y[order]),starts,axis=1)
        numerator=np.sum(pos*(np.cumsum(neg,axis=1)-.5*neg),axis=1)
        denominator=pos.sum(1)*neg.sum(1)
        output.append(np.divide(numerator,denominator,out=np.full(samples,np.nan),where=denominator>0))
    return np.asarray(output)


def predictive_probe(x,x0,labels,groups,sources,relations,lengths,fold_seed=123):
    """All centroids, neighbors, scalers, and label models use training folds only."""
    y=np.asarray(labels,dtype=int); groups=np.asarray(groups)
    nfolds=min(5,len(set(groups[y==1])),len(set(groups[y==0])))
    if nfolds<3: return {'available':False,'positive_count':int(y.sum()),'reason':'fewer than three outcome groups'}
    k,k0=gram(x),gram(x0)
    norm=np.linalg.norm(x,axis=1); norm0=np.linalg.norm(x0,axis=1)
    angle=np.degrees(np.arccos(np.clip(np.sum(x*x0,axis=1)/(norm*norm0).clip(1e-12),-1,1)))
    fixed=np.column_stack([angle,np.linalg.norm(x-x0,axis=1)/norm0.clip(1e-12),np.log((norm/norm0).clip(1e-12))])
    numeric=np.log1p(np.asarray(lengths))[:,None]
    category=np.asarray(relations)[:,None]
    predictions={key:np.zeros(len(y)) for key in ['nuisance','frozen_geometry','geometry','combined']}
    folds=np.full(len(y),-1,dtype=int)
    cv=StratifiedGroupKFold(n_splits=nfolds,shuffle=True,random_state=fold_seed)
    for fold,(train,test) in enumerate(cv.split(fixed,y,groups)):
        assert set(groups[train]).isdisjoint(groups[test])
        nn,radius=neighbors(k,sources,train); nn0,radius0=neighbors(k0,sources,train)
        features=np.column_stack([fixed,cone_angles(k,train),radius,overlap(nn,nn0)])
        scaler=StandardScaler().fit(features[train]); geometry=scaler.transform(features)
        enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(category[train])
        numeric_scaler=StandardScaler().fit(numeric[train])
        nuisance=np.column_stack([enc.transform(category),numeric_scaler.transform(numeric)])
        frozen_features=np.column_stack([np.log(norm0.clip(1e-12)),cone_angles(k0,train),radius0])
        frozen_scaler=StandardScaler().fit(frozen_features[train])
        frozen_features=frozen_scaler.transform(frozen_features)
        frozen_baseline=np.column_stack([nuisance,frozen_features])
        inputs={'nuisance':nuisance,'frozen_geometry':frozen_baseline,'geometry':geometry,
                'combined':np.column_stack([frozen_baseline,geometry])}
        for name,data in inputs.items():
            classifier=LogisticRegression(C=1,max_iter=2000,class_weight='balanced',solver='liblinear',random_state=123)
            classifier.fit(data[train],y[train])
            predictions[name][test]=classifier.predict_proba(data[test])[:,1]
        folds[test]=fold
    names=list(predictions); draws=auc_bootstrap(y,[predictions[k] for k in names],groups)
    result={'available':True,'n':len(y),'positive_count':int(y.sum()),'groups':len(set(groups)),
            'folds':nfolds,'fold_seed':fold_seed,'fold_ids':folds.tolist(),'prevalence':float(y.mean()),'scores':{},
            'ci_scope':'group resampling of fixed out-of-fold predictions; not refitting uncertainty'}
    for i,name in enumerate(names):
        result['scores'][name]={'auc':float(roc_auc_score(y,predictions[name])),
                               'auc_ci95':np.nanquantile(draws[i],[.025,.975]).tolist(),
                               'average_precision':float(average_precision_score(y,predictions[name]))}
    delta=draws[names.index('combined')]-draws[names.index('frozen_geometry')]
    result['incremental_auc']={'estimate':result['scores']['combined']['auc']-result['scores']['frozen_geometry']['auc'],
                               'ci95':np.nanquantile(delta,[.025,.975]).tolist()}
    result['oof_predictions']={k:v.tolist() for k,v in predictions.items()}
    return result


def repeated_predictive_probe(x,x0,labels,groups,sources,relations,lengths):
    """Average three fixed grouped fold assignments, without tuning to outcomes."""
    fits=[predictive_probe(x,x0,labels,groups,sources,relations,lengths,s) for s in [123,321,777]]
    if not all(r['available'] for r in fits): return fits[0]
    result=fits[0].copy(); y=np.asarray(labels,dtype=int)
    names=list(fits[0]['oof_predictions'])
    predictions={name:np.mean([r['oof_predictions'][name] for r in fits],axis=0) for name in names}
    draws=auc_bootstrap(y,[predictions[k] for k in names],groups)
    result['scores']={}
    for i,name in enumerate(names):
        result['scores'][name]={'auc':float(roc_auc_score(y,predictions[name])),
                               'auc_ci95':np.nanquantile(draws[i],[.025,.975]).tolist(),
                               'average_precision':float(average_precision_score(y,predictions[name])),
                               'split_aucs':[r['scores'][name]['auc'] for r in fits]}
    delta=draws[names.index('combined')]-draws[names.index('frozen_geometry')]
    result['incremental_auc']={'estimate':result['scores']['combined']['auc']-result['scores']['frozen_geometry']['auc'],
                              'ci95':np.nanquantile(delta,[.025,.975]).tolist(),
                              'split_estimates':[r['incremental_auc']['estimate'] for r in fits]}
    result['fold_assignments']={str(r['fold_seed']):r['fold_ids'] for r in fits}
    result.pop('fold_ids'); result.pop('fold_seed')
    result['oof_predictions']={k:v.tolist() for k,v in predictions.items()}
    return result
