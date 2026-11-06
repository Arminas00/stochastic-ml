"""Piccolo aiuto per scrivere i notebook come sorgenti Python leggibili.

Uso: in un file `build_XX.py` si definisce CELLS = [("md", "..."), ("code", "..."), ...]
e si chiama write_notebook(CELLS, "XX.ipynb").
"""
import textwrap

import nbformat

# Prima cella di codice di ogni notebook: su Colab clona/aggiorna il repository,
# ovunque installa i pacchetti mancanti. Nessun passo manuale.
COLAB_SETUP = r'''
# --- Setup: su Colab clona il repository; ovunque installa ciò che manca ---
import os, sys, subprocess, importlib.util

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/Arminas00/stochastic-ml.git"
if IN_COLAB:
    repo = "/content/stochastic-ml"
    if os.path.exists(repo):
        subprocess.run(["git", "-C", repo, "pull", "-q"], check=True)
    else:
        subprocess.run(["git", "clone", "-q", REPO_URL, repo], check=True)
    os.chdir(repo)

needed = ["numpy", "scipy", "pandas", "matplotlib", "sklearn", "ts2vg"]
missing = [m for m in needed if importlib.util.find_spec(m) is None]
if missing:
    print("Installo:", missing)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
sys.path.insert(0, os.getcwd())
print("Cartella di lavoro:", os.getcwd(), "| Colab:", IN_COLAB)
'''


def write_notebook(cells, path):
    nb = nbformat.v4.new_notebook()
    for kind, src in cells:
        src = textwrap.dedent(src).strip("\n")
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["accelerator"] = "GPU"
    nbformat.write(nb, path)
