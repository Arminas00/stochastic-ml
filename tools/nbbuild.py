"""Piccolo aiuto per scrivere i notebook come sorgenti Python leggibili.

Uso: in un file `build_XX.py` si definisce CELLS = [("md", "..."), ("code", "..."), ...]
e si chiama write_notebook(CELLS, "XX.ipynb").
"""
import textwrap

import nbformat


def write_notebook(cells, path):
    nb = nbformat.v4.new_notebook()
    for kind, src in cells:
        src = textwrap.dedent(src).strip("\n")
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["accelerator"] = "GPU"
    nbformat.write(nb, path)
