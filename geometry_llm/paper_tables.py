"""Shared colour treatment for generated, line-oriented LaTeX tables."""
import re


def style_table(text: str) -> str:
    """Keep model colours semantic; use neutral stripes for mixed-model rows.

    Generated rows occupy one line. Headers end in ``\\midrule``; existing
    colours are preserved so this transformation is safe to apply again.
    """
    output = []
    neutral_index = 0
    for line in text.splitlines(keepends=True):
        if "&" not in line or r"\\" not in line:
            output.append(line)
            continue
        if r"\rowcolor{" in line:
            output.append(line)
            continue
        if r"\midrule" in line:
            line = r"\rowcolor{headerrow}\color{white}" + line.replace(
                " & ", r" & \color{white}"
            )
            neutral_index = 0
        else:
            cells = [cell.strip() for cell in re.split(r"(?<!\\)&", line)]
            if "Llama" in cells:
                colour = "llamarow"
            elif "Qwen" in cells:
                colour = "qwenrow"
            else:
                colour = "neutralrow" if neutral_index % 2 == 0 else "neutralalt"
                neutral_index += 1
            line = r"\rowcolor{" + colour + "}" + line
        output.append(line)
    return "".join(output)
