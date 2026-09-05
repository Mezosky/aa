from geometry_llm.paper_tables import style_table


def test_headers_and_model_rows_keep_semantic_colours():
    source = (
        r"Model & Value\\\midrule" + "\n"
        + r"Llama & 1 \\" + "\n"
        + r"Data & Qwen & 2 \\" + "\n"
        + r"\bottomrule" + "\n"
    )
    result = style_table(source)
    assert r"\rowcolor{headerrow}\color{white}Model & \color{white}Value" in result
    assert r"\rowcolor{llamarow}Llama & 1" in result
    assert r"\rowcolor{qwenrow}Data & Qwen & 2" in result
    assert result.endswith("\\bottomrule\n")
    assert style_table(result) == result


def test_neutral_rows_and_prose():
    source = "Unchanged prose.\n" + r"P1 & 10 \\" + "\n" + r"P2 & 20 \\" + "\n"
    result = style_table(source)
    assert result.startswith("Unchanged prose.\n")
    assert r"\rowcolor{neutralrow}P1" in result
    assert r"\rowcolor{neutralalt}P2" in result


def test_existing_model_shading_is_preserved():
    source = r"\rowcolor{llamarow}Llama & $\Delta$ & 1 \\" + "\n"
    assert style_table(source) == source
