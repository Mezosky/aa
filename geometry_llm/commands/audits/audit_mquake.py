#!/usr/bin/env python
"""Audit the source-to-selection path and compatibility of a shared entity table."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from geometry_llm.data import mquake_cf_row_to_chain
from geometry_llm.text import normalize_answer, find_entity_spans


def conflicts(records, key_fields, target_field):
    grouped = defaultdict(list)
    for row in records:
        grouped[tuple(row[k] for k in key_fields)].append(row)
    bad = [{"key": list(k), "records": v} for k,v in sorted(grouped.items())
           if len({r[target_field] for r in v}) > 1]
    return {"unique_keys": len(grouped), "conflicting_keys": len(bad),
            "affected_case_ids": sorted({r['case_id'] for g in bad for r in g['records']}), "conflicts": bad}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", default="/tmp/MQuAKE-CF-3k-v2.json")
    p.add_argument("--root", default="outputs/265b7bb1723b")
    args = p.parse_args()
    source, root = Path(args.source), Path(args.root)
    raw = json.loads(source.read_text())
    selected = [json.loads(line) for line in (root / "selected_chains.jsonl").open()]
    selected_ids = {r['chain_id'] for r in selected}
    candidates = [c for i,r in enumerate(raw) if (c := mquake_cf_row_to_chain(r,i)) is not None]
    facts, checks = [], []
    for r in raw:
        if str(r['case_id']) not in selected_ids:
            continue
        triples = r['orig']['new_triples']
        labeled = r['orig']['new_triples_labeled']
        for hop,(rw,triple,label) in enumerate(zip(r['requested_rewrite'],triples,labeled)):
            h = r['new_single_hops'][hop]
            checks.append(rw['subject'] == label[0] and rw['relation_id'] == triple[1]
                          and rw['target_new']['id'] == triple[2]
                          and normalize_answer(h['answer']) == normalize_answer(rw['target_new']['str']))
            facts.append({'case_id':str(r['case_id']), 'hop':hop+1,
                          'entity':normalize_answer(rw['subject']), 'subject_id':triple[0],
                          'relation':triple[1], 'target_id':triple[2],
                          'answer':normalize_answer(h['answer']), 'prompt':normalize_answer(h['cloze'])})
    duplicates = Counter((f['subject_id'],f['relation'],f['target_id']) for f in facts)
    surface_ids = defaultdict(set)
    for f in facts:
        surface_ids[f['entity']].add(f['subject_id'])
    report = {
        'source':str(source), 'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_cases':len(raw), 'two_hop_cases':sum(len(r.get('new_single_hops',[]))==2 for r in raw),
        'two_hop_two_edit_candidates':len(candidates), 'selected_cases':len(selected),
        'excluded_after_adapter':[c.chain_id for c in candidates if c.chain_id not in selected_ids],
        'selection_policy':'two new single hops and two requested rewrites; unique token span in all three formatted prompts',
        'adapter_scope':'one entity table shared across all selected cases per model and seed',
        'one_hop_records':len(facts), 'unique_fact_triples':len(duplicates),
        'repeated_fact_records':sum(v-1 for v in duplicates.values()),
        'duplicate_case_ids':len(selected)-len(selected_ids),
        'duplicates_policy':'repeated constituent facts retained; no case-specific tables or answer-based exclusions',
        'structural_consistency_checks':{'passed':sum(checks),'total':len(checks)},
        'identity_relation':conflicts(facts,['subject_id','relation'],'target_id'),
        'surface_relation':conflicts(facts,['entity','relation'],'answer'),
        'exact_prompt':conflicts(facts,['prompt'],'answer'),
        'surface_identity_collisions':{k:sorted(v) for k,v in surface_ids.items() if len(v)>1},
        'bridge_surface_present_direct':sum(bool(find_entity_spans(r['prompt_12'],r['e2'])) for r in selected),
        'relation_pairs':len({r['fact_comp_type'] for r in selected}),
    }
    destination = root/'analysis/dataset_audit.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
