import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from geometry_llm.automatic_access import Span, build_batch, audit_cache, parse_answer, resolve_generated
from geometry_llm.modeling import ResidualTable


def test_occurrences_normalize_independently_and_keep_padding_unchanged():
    model=LlamaForCausalLM(LlamaConfig(vocab_size=20,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2))
    table=ResidualTable(['entity:a','entity:b'],16,mode='entity_span')
    table.delta.data[0].fill_(1); table.delta.data[1].fill_(2)
    ids,out,mask=build_batch(model,table,[[1,2,3,4,5],[6,7]],
        [[Span((0,1),0),Span((3,),1)],[Span((1,),0)]],0)
    base=model.get_input_embeddings()(ids)
    assert torch.allclose(out[0,0]-base[0,0],torch.full((16,),1/2**.5))
    assert torch.allclose(out[0,3]-base[0,3],torch.full((16,),2.))
    assert torch.equal(out[1,:4],base[1,:4])
    assert torch.allclose(out[1,4]-base[1,4],torch.ones(16))


@pytest.mark.parametrize('config,model_class',[(LlamaConfig,LlamaForCausalLM),(Qwen2Config,Qwen2ForCausalLM)])
def test_replayed_entity_cache_equals_uncached_reference(config,model_class):
    torch.manual_seed(123)
    model=model_class(config(vocab_size=31,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2)).eval()
    table=ResidualTable(['entity:source','entity:generated'],16,mode='entity_span')
    table.delta.data.normal_(std=.2)
    tokenizer=type('Tokenizer',(),{'pad_token_id':0})()
    spans=[Span((1,),0),Span((3,4),1)]
    result=audit_cache(model,tokenizer,table,[1,2,3,4,5,6],spans)
    assert all(result['greedy_matches'])
    assert max(result['max_abs_logit_differences'])<1e-5
    # A generated-row update actually changes the final prediction distribution.
    _,on,mask=build_batch(model,table,[[1,2,3,4,5,6]],[spans],0)
    _,off,_=build_batch(model,table,[[1,2,3,4,5,6]],[spans[:1]],0)
    with torch.no_grad():
        a=model(inputs_embeds=on,attention_mask=mask).logits[:,-1]
        b=model(inputs_embeds=off,attention_mask=mask).logits[:,-1]
    assert not torch.allclose(a,b)


def test_strict_answer_parser():
    assert parse_answer(' Nepal\nAnswer: Kathmandu')=='Kathmandu'
    assert parse_answer('Nepal\nThe answer is Kathmandu')==''
    assert parse_answer('Nepal')==''
    assert parse_answer(' Nepal\nAnswer: Kathmandu\nExplanation: extra')=='Kathmandu'


class CharacterTokenizer:
    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)

    def __call__(self, text, **kwargs):
        return {'input_ids':list(map(ord,text)),
                'offset_mapping':[(i,i+1) for i in range(len(text))]}


def test_resolver_only_uses_complete_unique_alias_and_original_generated_tokens():
    tokenizer=CharacterTokenizer()
    table=ResidualTable(['entity:New York'],4,mode='entity_span')
    table.delta.data.fill_(1)
    prefix=list(map(ord,'Intermediate:'))
    lookup={'new york':'New York'}
    result,span=resolve_generated(tokenizer,prefix,list(map(ord,' New York\n')),lookup,table)
    assert result['active_nonzero']
    assert span.positions==tuple(range(len(prefix)+1,len(prefix)+9))
    for text in [' New York',' New\n',' New York is a city\n']:
        result,span=resolve_generated(tokenizer,prefix,list(map(ord,text)),lookup,table)
        assert span is None
        assert not result['active_nonzero']


def test_non_roundtrip_generated_ids_are_not_silently_retokenized():
    class ChangedTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            result=super().__call__(text,**kwargs)
            result['input_ids'][0]+=1
            return result
    table=ResidualTable(['entity:Nepal'],4,mode='entity_span')
    result,span=resolve_generated(ChangedTokenizer(),list(map(ord,'Intermediate:')),
        list(map(ord,' Nepal\n')),{'nepal':'Nepal'},table)
    assert result['resolution']=='non_roundtrip_tokens'
    assert span is None
