from geometry_llm.data import clutrr_row_to_example, mquake_cf_row_to_chain, row_to_chain
from geometry_llm.evaluation import accuracy_summary
from geometry_llm.metrics import grouped_bootstrap_metrics, grouped_paired_difference
from geometry_llm.commands.evaluation.run_residual_swaps import different_target_donors


def test_twohopfact_alias_fields_are_supported():
    row = {
        "uid": 1, "e1.value": "Book", "e2.value": "Author", "e3.value": "City",
        "e2.aliases": "(('Writer',),)", "e3.aliases": "(('Town',),)",
        "r1(e1).prompt": "The author of Book is",
        "r2(e2).prompt": "Author was born in",
        "r2(r1(e1)).prompt": "The author of Book was born in",
        "fact_comp_type": "birth city of author",
    }
    chain = row_to_chain(row, 0)
    assert "Writer" in chain.e2_aliases
    assert "Town" in chain.e3_aliases


def test_mquake_counterfactual_two_hop_adapter():
    row = {
        "case_id": 7,
        "requested_rewrite": [
            {"subject": "Ada", "relation_id": "r1", "prompt": "{} lives in",
             "target_new": {"str": "Paris", "id": "q2"}},
            {"relation_id": "r2", "prompt": "The mayor of {} is",
             "target_new": {"str": "Bea", "id": "q3"}},
        ],
        "questions": ["Who is the mayor of the city where Ada lives?"],
        "new_answer": "Bea", "new_answer_alias": ["B"],
        "new_single_hops": [
            {"cloze": "Ada lives in", "answer": "Paris", "answer_alias": ["City of Paris"]},
            {"cloze": "The mayor of Paris is", "answer": "Bea", "answer_alias": ["B"]},
        ],
        "orig": {"new_triples_labeled": [["Ada", "lives in", "Paris"],
                                             ["Paris", "mayor", "Bea"]]},
    }
    chain = mquake_cf_row_to_chain(row, 0)
    assert chain is not None
    assert (chain.e1, chain.e2, chain.e3) == ("Ada", "Paris", "Bea")
    assert chain.prompt_12 == row["questions"][0]
    assert chain.rough_fact_comp_type == "counterfactual two hop"


def test_mquake_longer_cases_are_explicitly_excluded():
    assert mquake_cf_row_to_chain({"new_single_hops": [{}, {}, {}]}, 0) is None


def test_clutrr_adapter_recovers_relation_and_approximate_length():
    row = {"sentence1": "[A] is [B]'s mother. [B] is [C]'s brother.",
           "sentence2": "('A', 'C')", "labels": 4}
    labels = ["x0", "x1", "x2", "x3", "grandmother"]
    item = clutrr_row_to_example(row, 9, labels)
    assert item.query_subject == "A"
    assert item.query_object == "C"
    assert item.answer_relation == "grandmother"
    assert item.approximate_hops == 2


def test_composition_metrics_are_reported_explicitly():
    rows = [
        {"chain_id": "a", "correct_1": True, "correct_2": True,
         "correct_12": True, "correct_explicit": True},
        {"chain_id": "b", "correct_1": True, "correct_2": False,
         "correct_12": False, "correct_explicit": False},
    ]
    result = accuracy_summary(rows)["all"]
    assert result["A_2"] == .5
    assert result["A_explicit"] == .5
    assert result["J_1"] == .5
    assert result["A_1_independent"] == .5
    assert result["A_2_given_adapted_one_hops"] == 1.0


def test_targeted_swaps_always_change_bridge_target():
    base = {
        "e2_aliases": [], "e3_aliases": [], "prompt_1": "", "prompt_2": "",
        "prompt_12": "", "fact_comp_type": "x", "rough_fact_comp_type": "x",
    }
    from geometry_llm.data import Chain
    chains = [Chain(chain_id=str(i), e1=f"s{i}", e2=f"b{i}", e3=f"a{i}", **base)
              for i in range(4)]
    mapping, representatives = different_target_donors(chains)
    assert all(representatives[source].e2 != representatives[donor].e2
               for source, donor in mapping.items())


def test_grouped_bootstrap_and_paired_difference():
    base = [
        {"chain_id": "1", "bridge_answer_group": "a", "correct_1": False,
         "correct_2": True, "correct_12": False},
        {"chain_id": "2", "bridge_answer_group": "a", "correct_1": True,
         "correct_2": True, "correct_12": False},
        {"chain_id": "3", "bridge_answer_group": "b", "correct_1": False,
         "correct_2": False, "correct_12": False},
    ]
    adapted = [dict(row, correct_1=True, correct_12=True) for row in base]
    summary = grouped_bootstrap_metrics(adapted, samples=20, seed=1)
    paired = grouped_paired_difference(base, adapted, samples=20, seed=1)
    assert summary["n_groups"] == 2
    assert summary["metrics"]["A_2"]["estimate"] == 1.0
    assert paired["differences"]["A_2"]["estimate"] == 1.0
