"""Sorgente del notebook colab_full_run.ipynb: esegue i notebook 01-05 in modalità full su Colab (GPU)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbbuild import write_notebook

CELLS = [
("md", r"""
# Esecuzione completa (modalità `full`) su Colab

Questo notebook non contiene analisi: esegue in sequenza i notebook 01–05 con `SP_MODE=full`, sul dataset
definitivo (N = 1024, 40 000 serie di train) e con la GPU di Colab.

- I notebook eseguiti si salvano in `risultati_full/` e le figure in `figures_full/`. I notebook in `quick`
  del repository restano invariati.
- Con Google Drive montato, dati intermedi e risultati vanno in `MyDrive/stochastic-ml-full`. Se il runtime si
  disconnette non si perde nulla: rieseguendo, i notebook già completati si saltano e i dati in cache si
  ricaricano.
- Prima di lanciarlo: *Runtime → Cambia tipo di runtime → GPU*.
"""),
("code", r'''
import os, sys, subprocess, time, shutil, importlib.util
REPO_URL = "https://github.com/Arminas00/stochastic-ml.git"
REPO = "/content/stochastic-ml"
if os.path.exists(REPO):
    subprocess.run(["git", "-C", REPO, "pull", "-q"], check=True)
else:
    subprocess.run(["git", "clone", "-q", REPO_URL, REPO], check=True)
os.chdir(REPO); sys.path.insert(0, REPO)
needed = ["numpy", "scipy", "pandas", "matplotlib", "sklearn", "ts2vg", "torch", "torch_geometric"]
if any(importlib.util.find_spec(m) is None for m in needed):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
import psutil; print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
'''),
("code", r'''
USE_DRIVE = True    # False: tutto resta in /content e si perde alla disconnessione
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/stochastic-ml-full"
else:
    OUT = "/content/stochastic-ml-full"
for d in ["data", "risultati_full", "figures_full"]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
    link = f"{REPO}/{d}"
    if os.path.islink(link):
        continue
    if os.path.isdir(link):            # cartella locale già creata: si sposta il contenuto su Drive
        for f in os.listdir(link):
            shutil.move(f"{link}/{f}", f"{OUT}/{d}/{f}")
        os.rmdir(link)
    os.symlink(f"{OUT}/{d}", link)
print("Risultati in", OUT, "|", {d: len(os.listdir(f"{OUT}/{d}")) for d in ["data", "risultati_full", "figures_full"]})
'''),
("code", r'''
TIMES = {}
def run(nb, timeout_h=10, force=False):
    """Esegue un notebook con SP_MODE=full; salta se il risultato esiste già (ripresa dopo disconnessione)."""
    out = f"{REPO}/risultati_full/{nb}"
    if os.path.exists(out) and not force:
        print(f"{nb}: già eseguito, salto (force=True per rifarlo)"); return
    env = dict(os.environ, SP_MODE="full")
    t0 = time.time()
    r = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", nb,
                        "--output-dir", "/content/tmp_out", f"--ExecutePreprocessor.timeout={int(timeout_h * 3600)}"],
                       env=env, capture_output=True, text=True)
    TIMES[nb] = (time.time() - t0) / 60
    if r.returncode != 0:
        print(r.stderr[-4000:]); raise RuntimeError(f"{nb} fallito dopo {TIMES[nb]:.1f} min")
    shutil.copy(f"/content/tmp_out/{nb}", out)
    print(f"{nb}: completato in {TIMES[nb]:.1f} min")

def show_outputs(nb, last=40):
    """Ultime righe di testo prodotte da un notebook eseguito."""
    import nbformat
    lines = []
    for c in nbformat.read(f"{REPO}/risultati_full/{nb}", as_version=4).cells:
        for o in c.get("outputs", []):
            t = o.get("text") or o.get("data", {}).get("text/plain", "")
            lines += (t if isinstance(t, str) else "".join(t)).splitlines()
    print("\n".join(lines[-last:]))
'''),
("md", r"""
## 01–03: simulazione, stimatori classici, CNN e modelli sulle feature
"""),
("code", r'''
run("01_simulazione_validazione.ipynb"); show_outputs("01_simulazione_validazione.ipynb", 20)
'''),
("code", r'''
run("02_classici_e_grafi.ipynb"); show_outputs("02_classici_e_grafi.ipynb", 30)
'''),
("code", r'''
run("03_cnn_e_feature.ipynb"); show_outputs("03_cnn_e_feature.ipynb", 40)
'''),
("md", r"""
## Stima del costo del notebook 04 (GNN)

Prima di lanciare la GNN in `full` misuriamo il tempo di un'epoca su 2000 grafi e lo estrapoliamo a 40 000.
Il limite superiore conta tutte le epoche massime: 2 configurazioni per la scelta + 4 varianti × 3 seed,
fino a 25 epoche ciascuno (D-37). Con l'early stopping il tempo reale è di solito minore.
"""),
("code", r'''
import numpy as np, torch, sp_utils as u
ds = u.load_or_make_fbm_dataset("full", 2026, "data")
n = 2000; X, Hs = ds["train"]["X"][:n], ds["train"]["H"][:n]
est = {}
for kind, series in [("nvg", "fbm"), ("hvg", "fgn")]:
    t0 = time.time(); e, k, v = u.build_graphs(X, kind, series); tb = time.time() - t0
    lk = np.log(np.concatenate(k)); G = u.GraphSet(e, u.node_features(k, v, "grado+valori", lk.mean(), lk.std()))
    for hp in [dict(hidden=32, n_layers=3), dict(hidden=64, n_layers=4)]:
        m = u.build_gnn(n_in=2, **hp)
        t0 = time.time()
        u.train_regressor(m, G, Hs, G, Hs, max_epochs=2, patience=5, batch_size=128, lr=1e-3, seed=0)
        te = (time.time() - t0) / 2 / 2      # 2 epoche, train + "val" sugli stessi grafi ≈ 2× un'epoca
        est[(kind, hp["hidden"])] = te * 40_000 / n
        print(f"{kind} {hp}: costruzione grafi {tb * 66_000 / n / 60:.1f} min (tutti gli split), "
              f"epoca su 40k ≈ {te * 40_000 / n / 60:.2f} min")
ep = np.mean([est[("nvg", 64)], est[("hvg", 64)]])
print(f"Limite superiore notebook 04 (3 seed, 25 epoche, D-37): ≈ "
      f"{(25 * (est[('nvg', 32)] + est[('nvg', 64)]) + 2 * 3 * 25 * (est[('nvg', 64)] + est[('hvg', 64)])) / 3600:.1f} ore")
'''),
("md", r"""
## 04–05: GNN e modello ibrido
"""),
("code", r'''
run("04_gnn.ipynb", timeout_h=12); show_outputs("04_gnn.ipynb", 40)
'''),
("code", r'''
run("05_ibrido_ablazione.ipynb"); show_outputs("05_ibrido_ablazione.ipynb", 40)
'''),
("md", r"""
## Archivio dei risultati
"""),
("code", r'''
import zipfile, glob
zpath = f"{OUT}/risultati_full.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{REPO}/risultati_full/*.ipynb") + glob.glob(f"{REPO}/figures_full/*.png") + glob.glob(f"{REPO}/data/results*_full.csv"):
        z.write(f, os.path.relpath(os.path.realpath(f), OUT))
print(zpath, f"{os.path.getsize(zpath) / 1e6:.1f} MB", "| tempi (min):", {k: round(v, 1) for k, v in TIMES.items()})
'''),
]

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "colab_full_run.ipynb"
    write_notebook(CELLS, out)
    print("Scritto", out)
