#!/usr/bin/env python
"""Materialize SOCRATES as typed paths without using the graph in the loss."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import networkx as nx
from datasets import load_dataset

from geometry_llm.config import common_parser, load_config, output_path
from geometry_llm.data import row_to_chain


def node_id(chain, role):
    qid = getattr(chain, f"{role}_id")
    value = getattr(chain, role)
    return qid or f"{getattr(chain, f'{role}_type')}:{value}"


def main():
    parser = common_parser("Build typed entity graphs from SOCRATES")
    parser.add_argument("--selected-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    ds = load_dataset(cfg["data"]["dataset"], split=cfg["data"]["split"])
    chains = [row_to_chain(dict(row), i) for i, row in enumerate(ds)]
    if args.selected_only and cfg["data"]["composition_type"] != "auto":
        chains = [c for c in chains if c.fact_comp_type == cfg["data"]["composition_type"]]

    nodes, edges, degree = {}, [], Counter()
    for chain in chains:
        for role in ("e1", "e2", "e3"):
            key = node_id(chain, role)
            nodes.setdefault(key, {
                "node_id": key, "label": getattr(chain, role),
                "entity_type": getattr(chain, f"{role}_type"),
                "rough_type": getattr(chain, f"{role}_rough_type"),
            })
        for hop, source, target, relation, rough_relation, template_id in (
            (1, "e1", "e2", chain.r1_type, chain.r1_rough_type, chain.r1_template_id),
            (2, "e2", "e3", chain.r2_type, chain.r2_rough_type, chain.r2_template_id),
        ):
            source_id, target_id = node_id(chain, source), node_id(chain, target)
            edges.append({
                "chain_id": chain.chain_id, "source": source_id, "target": target_id,
                "source_label": getattr(chain, source), "target_label": getattr(chain, target),
                "relation": relation, "rough_relation": rough_relation, "hop": hop,
                "template_id": template_id, "fact_comp_type": chain.fact_comp_type,
                "rough_fact_comp_type": chain.rough_fact_comp_type,
            })
            degree[source_id] += 1; degree[target_id] += 1
    for key in nodes: nodes[key]["degree"] = degree[key]

    out = output_path(cfg, "graph_all" if not args.selected_only else "graph_selected")
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("nodes.csv", list(nodes.values())), ("edges.csv", edges)):
        with (out / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((edge["source"], edge["target"]) for edge in edges)
    relation_counts = Counter(edge["relation"] for edge in edges)
    type_counts = Counter(node["entity_type"] for node in nodes.values())
    comp_counts = Counter(c.fact_comp_type for c in chains)
    summary = {
        "chains_length_two_paths": len(chains), "nodes": len(nodes), "directed_edge_records": len(edges),
        "unique_directed_edges": len({(e['source'], e['relation'], e['target']) for e in edges}),
        "relation_types": len(relation_counts), "entity_types": len(type_counts),
        "connected_components_undirected": nx.number_connected_components(graph),
        "largest_component_nodes": max(map(len, nx.connected_components(graph)), default=0),
        "entity_type_counts": dict(type_counts), "relation_counts": dict(relation_counts),
        "composition_counts": dict(comp_counts),
        "interpretation": "Each chain is e1 -[r1]-> e2 -[r2]-> e3. Only the two edges supervise the residual; the composed prompt tests the held-out length-two path.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # An intuitive overview: relation families and their number of edge records.
    top = relation_counts.most_common(20)
    plt.figure(figsize=(10, 6)); plt.barh([x[0] for x in reversed(top)], [x[1] for x in reversed(top)])
    plt.xlabel("edge records"); plt.title("Largest SOCRATES relation types"); plt.tight_layout()
    plt.savefig(out / "relation_counts.png", dpi=180); plt.close()
    print(json.dumps({k: summary[k] for k in list(summary)[:7]}, indent=2))
    print(f"Graph artifacts: {out}")


if __name__ == "__main__":
    main()
