#!/usr/bin/env python
"""Audit coverage, paired prompts, and canonical single-edit target metadata.

The initial runner inherited chain ID fields when replacing target strings.
Repair those descriptive IDs in generated single-edit files only; predictions,
prompts, target strings, aliases, and correctness must remain identical.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from geometry_llm.data import load_saved_chains
from geometry_llm.evaluation import write_json
from geometry_llm.text import answer_is_correct
from geometry_llm.commands.training.run_final_confirmation import controlled_chains


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--root',required=True)
    p.add_argument('--source',default='/tmp/MQuAKE-CF-3k-v2.json'); args=p.parse_args()
    root=Path(args.root); work=root/'confirmation_v1'
    raw={str(r['case_id']):r for r in json.loads(Path(args.source).read_text())}
    chains=load_saved_chains(root/'selected_chains.jsonl')
    audit={'coverage':{},'single_edit':{},'metadata_only_repairs':0}
    for seed in [13,37,71]:
        dest=work/'both'/f'seed-{seed}'
        meta=json.loads((dest/'final.json').read_text())
        assert meta['nonzero_rows']==meta['eligible_rows']==839
        assert all(h['unique_facts']==h['visited']==892 for h in meta['history'])
        rows=[json.loads(l) for l in (dest/'access.jsonl').read_text().splitlines()]
        assert len(rows)==533
        for r in rows:
            assert r['correct_pipeline']==answer_is_correct(r['prediction_pipeline'],r['e3_aliases'])
            assert r['correct_pipeline_stage2_off']==answer_is_correct(r['prediction_pipeline_stage2_off'],r['e3_aliases'])
        audit['coverage'][str(seed)]={'epochs':meta['epochs'],'all_facts_each_epoch':True,'active_rows':839,'paired_cases':len(rows)}
    for scope in ['source','bridge']:
        expected={c.chain_id:asdict(c) for c in controlled_chains(chains,raw,scope)}
        assert len(expected)==525
        dest=work/scope/'seed-13'; manifests=[]
        for filename in ['consistent.jsonl','consistent_frozen.jsonl','consistent_access.jsonl']:
            path=dest/filename; before=path.read_bytes()
            rows=[json.loads(l) for l in before.decode().splitlines()]
            assert {r['chain_id'] for r in rows}==set(expected)
            changes=0
            for r in rows:
                e=expected[r['chain_id']]
                for field in ['e1','e2','e3','prompt_1','prompt_2','prompt_12','e2_aliases','e3_aliases']:
                    assert r[field]==e[field],(scope,r['chain_id'],field)
                for field in ['e1_id','e2_id','e3_id']:
                    if r[field]!=e[field]: changes+=1; r[field]=e[field]
                for suffix,aliases in [('1','e2_aliases'),('2','e3_aliases'),('12','e3_aliases')]:
                    assert r['correct_'+suffix]==answer_is_correct(r['prediction_'+suffix],r[aliases])
            if changes:
                path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            audit['metadata_only_repairs']+=changes
            manifests.append({'file':filename,'changed_id_fields':changes,
                'sha256_before':hashlib.sha256(before).hexdigest(),'sha256_after':hashlib.sha256(path.read_bytes()).hexdigest()})
        adapted=[json.loads(l) for l in (dest/'consistent.jsonl').read_text().splitlines()]
        frozen=[json.loads(l) for l in (dest/'consistent_frozen.jsonl').read_text().splitlines()]
        change=sum(a['prediction_12']!=b['prediction_12'] for a,b in zip(adapted,frozen))
        first_change=sum(a['prediction_1']!=b['prediction_1'] for a,b in zip(adapted,frozen))
        second_change=sum(a['prediction_2']!=b['prediction_2'] for a,b in zip(adapted,frozen))
        if scope=='bridge': assert change==0
        if scope=='bridge': assert first_change==0
        if scope=='source': assert second_change==0
        write_json(dest/'target_manifest.json',list(expected.values()))
        audit['single_edit'][scope]={'n':525,'direct_string_changes':change,
            'first_probe_string_changes':first_change,'second_probe_string_changes':second_change,
            'files':manifests,'target_strings_and_scores_verified':True}
    write_json(work/'final_audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__': main()
