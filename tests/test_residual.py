import pytest

torch = pytest.importorskip("torch")

from geometry_llm.modeling import EncodedExample, ResidualTable


def test_delta_mask_changes_only_selected_positions():
    table = ResidualTable(["entity:x"], 3, mode="entity_span")
    table.delta.data.fill_(2)
    base = torch.zeros(1, 4, 3)
    indices = torch.tensor([[-1, 0, 0, -1]])
    result = table(base, indices)
    assert torch.equal(result[0, 0], base[0, 0])
    assert torch.equal(result[0, 3], base[0, 3])
    assert torch.allclose(result[0, 1], torch.full((3,), 2 / 2**0.5))


def test_entity_span_preserves_embedding_dtype():
    table = ResidualTable(["entity:x"], 3, mode="entity_span")
    base = torch.zeros(1, 2, 3, dtype=torch.bfloat16)
    indices = torch.tensor([[0, 0]])
    assert table(base, indices).dtype == base.dtype


def test_empty_residual_table_is_a_noop():
    table = ResidualTable([], 3, alpha=0, mode="entity_span")
    base = torch.randn(1, 2, 3)
    result = table(base, torch.full((1, 2), -1))
    assert result is base


def test_prompt_and_answer_masks_are_disjoint():
    item = EncodedExample([1, 2, 3, 4], [-100, -100, 3, 4], [-1, 0, -1, -1], 2)
    assert all(x == -100 for x in item.labels[:item.prompt_length])
    assert all(x == -1 for x in item.delta_indices[item.prompt_length:])
