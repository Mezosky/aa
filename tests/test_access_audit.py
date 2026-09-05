from types import SimpleNamespace

import pytest
import torch

from geometry_llm.commands.audits.audit_composition import paired_audit
from geometry_llm.commands.audits.audit_mquake import conflicts
from geometry_llm.commands.evaluation.evaluate_access import alias_lookup, stage_two_prompt
from geometry_llm.commands.training.run_lora_baseline import LoRALinear
from geometry_llm.modeling import ResidualTable


def chain(e1='Ada', e2='Paris', e3='Bea'):
    return SimpleNamespace(chain_id='1',e1=e1,e2=e2,e3=e3,
                           e2_aliases=[e2,'City'],e3_aliases=[e3],
                           prompt_2=f'The mayor of {e2} is')


def test_pipeline_uses_generated_bridge_even_when_wrong():
    c=chain(); other=chain('Bob','Rome','Dan')
    lookup=alias_lookup([c,other])
    prompt,resolved=stage_two_prompt(c,'Rome',lookup)
    assert prompt=='The mayor of Rome is' and resolved=='Rome'
    prompt,resolved=stage_two_prompt(c,'Unknown',lookup)
    assert prompt=='The mayor of Unknown is' and resolved is None
    # Ambiguous alias cannot be resolved by consulting the gold bridge.
    assert stage_two_prompt(c,'City',lookup)==('The mayor of City is',None)


def test_common_population_is_intersection_and_empty_is_undefined():
    def row(i,joint,direct):
        return dict(chain_id=str(i),e2='b'+str(i),e3='a',correct_1=joint,
                    correct_2=joint,correct_12=direct)
    b=[row(1,True,True),row(2,False,False),row(3,True,False)]
    a=[row(1,True,False),row(2,True,True),row(3,False,True)]
    report=paired_audit(b,a,samples=100)
    assert report['base_C']['denominator']==2
    assert report['adapted_C']['denominator']==2
    assert report['common_change']['denominator']==1
    assert report['common_change']['estimate']==-1
    report=paired_audit([row(1,False,False)],[row(1,True,True)],samples=100)
    assert report['base_C']['estimate'] is None
    assert report['common_adapted']['estimate'] is None
    with pytest.raises(ValueError): paired_audit(b,a[:1])


def test_audit_catches_conflicting_repeated_supervision():
    facts=[dict(case_id='1',s='x',r='r',a='a'),dict(case_id='2',s='x',r='r',a='b')]
    report=conflicts(facts,['s','r'],'a')
    assert report['conflicting_keys']==1 and report['affected_case_ids']==['1','2']


def test_lora_starts_identical_and_only_adapter_gets_gradient():
    base=torch.nn.Linear(5,3)
    for p in base.parameters(): p.requires_grad_(False)
    layer=LoRALinear(base,2); x=torch.randn(4,5)
    assert torch.equal(layer(x),base(x))
    layer(x).square().sum().backward()
    assert layer.b.grad is not None and layer.b.grad.abs().sum()>0
    assert base.weight.grad is None
    with torch.no_grad(): layer.b.fill_(.1)
    assert not torch.allclose(layer(x),base(x))
    layer.scaling=0
    assert torch.equal(layer(x),base(x))


def test_joint_adapter_updates_both_components_and_keeps_base_frozen():
    torch.manual_seed(7)
    base=torch.nn.Linear(5,3)
    for p in base.parameters(): p.requires_grad_(False)
    layer=LoRALinear(base,2)
    table=ResidualTable(['entity:A'],5,mode='entity_span')
    x=torch.randn(2,3,5); indices=torch.tensor([[0,-1,-1],[0,0,-1]])
    original=base.weight.detach().clone()
    optimizer=torch.optim.AdamW([table.delta,layer.a,layer.b],lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        layer(table(x,indices)).square().mean().backward()
        assert table.delta.grad.abs().sum()>0
        assert layer.b.grad.abs().sum()>0
        assert base.weight.grad is None
        optimizer.step()
    assert table.delta.norm()>0 and layer.b.norm()>0
    assert torch.equal(base.weight,original)
    table.alpha=0; layer.scaling=0
    assert torch.equal(layer(table(x,indices)),base(x))
