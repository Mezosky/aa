from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from geometry_llm.confirmation import coverage_epoch, unique_weighted_facts, weighted_answer_loss, single_edit_targets
from geometry_llm.modeling import ResidualTable


def chain(source="s",bridge="b",answer="a"):
    return SimpleNamespace(e1=source,e2=bridge,e3=answer,r1_type="r1",r2_type="r2",
                           prompt_1=source+" r1",prompt_2=bridge+" r2")


def test_unique_coverage_and_weights():
    rows=unique_weighted_facts([chain(),chain(),chain("s2","b","a")])
    assert len(rows)==3
    assert sum(r["weight"] for r in rows)==pytest.approx(3)
    for seed in [13,37,71]:
        order=coverage_epoch(rows,seed)
        assert sorted(order)==list(range(3))
        assert len(set(order))==len(rows)
    assert len(unique_weighted_facts([chain(),chain()],"source"))==1
    assert len(unique_weighted_facts([chain(),chain()],"bridge"))==1


def test_conditional_targets_follow_the_remaining_fact():
    raw=dict(new_single_hops=[{"answer":"new_bridge"}],new_answer="new_answer",
             answer="original_answer",single_hops=[{"answer":"old_bridge"}],
             requested_rewrite=[{},dict(target_true={"str":"new_bridges_old_answer"})],
             orig=dict(triples=[["source","r1","old_id"],["old_id","r2","answer_id"]]))
    assert single_edit_targets(raw,"source",{})==("new_bridge","new_bridges_old_answer")
    assert single_edit_targets(raw,"bridge",{})==("old_bridge","original_answer")
    assert single_edit_targets(raw,"bridge",{("old_id","r2"):"elsewhere_edit"})==("old_bridge","elsewhere_edit")
    assert single_edit_targets(raw,"both",{})==("new_bridge","new_answer")


def test_sparse_answer_head_matches_full_weighted_ce_and_gradient():
    class Backbone(nn.Module):
        def forward(self,inputs_embeds,**kwargs):
            return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(1))
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.embedding=nn.Embedding(11,4); self.model=Backbone(); self.lm_head=nn.Linear(4,11)
        def get_input_embeddings(self): return self.embedding
    torch.manual_seed(3)
    model=Model(); table=ResidualTable(["entity:x"],4)
    batch=dict(input_ids=torch.tensor([[1,2,3,4],[4,5,6,0]]),
        labels=torch.tensor([[-100,-100,3,4],[-100,5,6,-100]]),
        delta_indices=torch.tensor([[0,-1,-1,-1],[0,-1,-1,-1]]),attention_mask=torch.ones(2,4))
    weights=torch.tensor([.5,1.5])
    sparse=weighted_answer_loss(model,table,batch,weights)
    hidden=model.model(inputs_embeds=table(model.embedding(batch["input_ids"]),batch["delta_indices"])).last_hidden_state
    logits=model.lm_head(hidden[:,:-1]).float(); labels=batch["labels"][:,1:]
    full=F.cross_entropy(logits.reshape(-1,11),labels.reshape(-1),reduction="none").reshape_as(labels)
    expected=((full.sum(1)/(labels!=-100).sum(1))*weights).mean()
    assert torch.allclose(sparse,expected)
    assert torch.allclose(torch.autograd.grad(sparse,table.delta,retain_graph=True)[0],
                          torch.autograd.grad(expected,table.delta)[0])
