"""Sorgente del notebook 04_gnn.ipynb."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbbuild import COLAB_SETUP, write_notebook

SETUP = COLAB_SETUP.replace(
    'needed = ["numpy", "scipy", "pandas", "matplotlib", "sklearn", "ts2vg"]',
    'needed = ["numpy", "scipy", "pandas", "matplotlib", "sklearn", "ts2vg", "torch", "torch_geometric"]')

CELLS = [
("md", r"""
# 04 — Graph Neural Network sui grafi di visibilità

**Compito.** Stimare $H$ da una traiettoria di fBm, dando alla rete **il grafo di visibilità** della serie
invece della serie stessa.

**Domande**
1. Una GNN che vede il grafo intero recupera l'informazione su $H$ che la relazione analitica di Lacasa
   perdeva (notebook 02)? Fa meglio delle 16 feature del grafo costruite a mano (notebook 03)?
2. **Cosa conserva e cosa perde la rappresentazione a grafo?** Con il solo grado come feature dei nodi,
   la rete vede solo la topologia, invariante per riscalamento (e, per la NVG, per trend lineare): lo
   verifichiamo direttamente sulle previsioni della rete.
3. Aggiungere i **valori della serie** come feature dei nodi aiuta?

**Varianti** (stesso modello, stesso protocollo)

| Grafo | Feature dei nodi |
|---|---|
| NVG del fBm (come Lacasa et al.) | grado · grado + valori |
| HVG del fGn (il grafo con l'esponente $\lambda$ più informativo nel notebook 02) | grado · grado + valori |

**Contenuto:** 1. Configurazione · 2. Costruzione dei grafi · 3. Modello e scelta degli iperparametri ·
4. Ensemble delle quattro varianti · 5. Confronto sul test · 6. Invarianze: verifica sulle previsioni ·
7. Robustezza rispetto a $N$ · 8. Salvataggio · 9. Riepilogo
"""),
("code", SETUP),
("md", r"""
## 1. Configurazione

La GNN è il modello più costoso del progetto. In `quick` (CPU) usiamo meno epoche e 2 seed: serve solo a
verificare che tutto giri (~30 minuti su un portatile). I numeri definitivi vengono dalla modalità `full` su GPU.
"""),
("code", r'''
MODE = os.environ.get("SP_MODE", "quick")   # "quick" (CPU, verifica) o "full" (definitivo); il runner Colab imposta SP_MODE=full
SEED = 2026
DATA_DIR = "data"
FIG_DIR = "figures" if MODE == "quick" else f"figures_{MODE}"   # figure separate per modalità
N_BOOT = 1000

if MODE == "quick":
    N_SEEDS = 2
    GNN_GRID = [dict(hidden=32, n_layers=3), dict(hidden=64, n_layers=3)]
    TRAIN_KW = dict(max_epochs=12, patience=4, batch_size=128, lr=1e-3)
else:
    N_SEEDS = 3          # D-37: budget ridotto per i limiti di Colab (CNN: 5 seed, 40 epoche)
    GNN_GRID = [dict(hidden=32, n_layers=3), dict(hidden=64, n_layers=4)]
    TRAIN_KW = dict(max_epochs=25, patience=6, batch_size=128, lr=1e-3)
'''),
("code", r'''
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import sp_utils as u

u.set_seed(SEED)
Path(FIG_DIR).mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
def savefig(name):
    plt.savefig(f"{FIG_DIR}/04_{name}.png", bbox_inches="tight", dpi=150)

DEV = u.get_device()
print("Dispositivo:", DEV)
CFG = u.MODES[MODE]
ds = u.load_or_make_fbm_dataset(MODE, SEED, DATA_DIR)
SPLITS = list(ds)
H = {s: ds[s]["H"].astype(np.float64) for s in SPLITS}
'''),
("md", r"""
## 2. Costruzione dei grafi

Per ogni serie costruiamo la NVG del fBm e l'HVG del fGn con `ts2vg` (verificata contro la definizione nel
notebook 01). Gli archi si conservano in forma compatta (interi a 32 bit, una sola direzione) e diventano
tensori solo quando si forma il batch (D-32).
"""),
("code", r'''
GRAPHS = {}
KINDS = {"NVG(fBm)": ("nvg", "fbm"), "HVG(fGn)": ("hvg", "fgn")}
rows = []
for gname, (kind, series) in KINDS.items():
    GRAPHS[gname] = {}
    t0 = time.time()
    for s in SPLITS:
        GRAPHS[gname][s] = u.build_graphs(ds[s]["X"], kind, series)
    e_tr = GRAPHS[gname]["train"][0]
    rows.append({"grafo": gname, "secondi": time.time() - t0,
                 "archi medi (train)": np.mean([len(e) for e in e_tr]),
                 "grado medio (train)": 2 * np.mean([len(e) for e in e_tr]) / CFG["N"],
                 "MB archi (tutti gli split)": sum(e.nbytes for s in SPLITS for e in GRAPHS[gname][s][0]) / 1e6})
display(pd.DataFrame(rows).round(2))
'''),
("md", r"""
**Feature dei nodi** (D-33):
- `grado`: $\log k$ del nodo, standardizzato con media e deviazione standard calcolate sui nodi del **train**;
- `grado+valori`: in più il valore della serie nel nodo, normalizzato per serie (media 0, deviazione 1),
  coerente con la scelta della CNN (D-26): non reintroduce l'ampiezza.
"""),
("code", r'''
DATA = {}
for gname in KINDS:
    lk = np.log(np.concatenate(GRAPHS[gname]["train"][1]))
    mu, sd = lk.mean(), lk.std()
    for which in ["grado", "grado+valori"]:
        name = f"4 GNN {gname} {which}"
        DATA[name] = {s: u.GraphSet(GRAPHS[gname][s][0],
                                    u.node_features(GRAPHS[gname][s][1], GRAPHS[gname][s][2], which, mu, sd))
                      for s in SPLITS}
print(list(DATA))
'''),
("md", r"""
## 3. Modello e scelta degli iperparametri

**GIN** (Graph Isomorphism Network, Xu et al. 2019), `sp_utils.build_gnn`: strati di passaggio di messaggi con
aggregazione a **somma** sui vicini seguita da un MLP; dopo ogni strato si legge il grafo con media e massimo
sui nodi; le letture di tutti gli strati vanno in una testa densa. La somma (e non la media, come nella GCN)
permette alla rete di "contare" i vicini, cioè di vedere i gradi e le loro correlazioni: è l'informazione che
contiene $H$. Motivazioni e alternative in D-31.

Stesso protocollo della CNN (D-27): due configurazioni, scelte sul validation (sulla variante principale
NVG(fBm) con il solo grado, un seed); la configurazione scelta si usa per tutte e quattro le varianti.
"""),
("code", r'''
import re
def fit_gnn(name, hp, seed, verbose=False):
    n_in = DATA[name]["train"].x[0].shape[1]
    m = u.build_gnn(n_in=n_in, **hp)
    # in full ogni modello addestrato si salva: dopo una disconnessione di Colab si riprende da lì (D-39)
    ck = (Path(DATA_DIR) / f"ckpt04_{MODE}" / f"{re.sub(r'[^A-Za-z0-9]+', '_', name)}_h{hp['hidden']}"
          f"_l{hp['n_layers']}_s{seed}.pt") if MODE == "full" else None
    return u.train_regressor(m, DATA[name]["train"], H["train"], DATA[name]["val"], H["val"],
                             seed=seed, verbose=verbose, checkpoint=ck, **TRAIN_KW)

MAIN = "4 GNN NVG(fBm) grado"
search_rows, best = [], None
for hp in GNN_GRID:
    t0 = time.time()
    m, hist = fit_gnn(MAIN, hp, SEED)
    val = min(h["rmse_val"] for h in hist)
    search_rows.append({**hp, "parametri": u.count_params(m), "epoche": len(hist), "RMSE val": val,
                        "s/epoca": np.mean([h["s"] for h in hist]), "secondi": time.time() - t0})
    if best is None or val < best[1]:
        best = (hp, val, hist)
display(pd.DataFrame(search_rows).round(4))
GNN_HP = best[0]
print("Configurazione scelta:", GNN_HP)
'''),
("md", r"""
## 4. Ensemble delle quattro varianti
"""),
("code", r'''
PRED, PRED_STD, MEMBER_RMSE, HISTS, MODELS, TIMES = {}, {}, {}, {}, {}, {}
for name in DATA:
    t0 = time.time()
    members, hists = [], []
    for k in range(N_SEEDS):
        m, h = fit_gnn(name, GNN_HP, SEED + 100 * k)
        members.append(m); hists.append(h)
    TIMES[name] = time.time() - t0
    MODELS[name], HISTS[name] = members, hists
    P = {s: np.stack([u.predict(m, DATA[name][s]) for m in members]) for s in SPLITS}
    PRED[name] = {s: P[s].mean(0) for s in SPLITS}
    PRED_STD[name] = {s: P[s].std(0) for s in SPLITS}
    MEMBER_RMSE[name] = [u.regression_metrics(H["val"], P["val"][k])["RMSE"] for k in range(N_SEEDS)]
    print(f"{name:<34} membri (val) {np.round(MEMBER_RMSE[name], 4)}  ensemble (val) "
          f"{u.regression_metrics(H['val'], PRED[name]['val'])['RMSE']:.4f}  ({TIMES[name]:.0f} s)")
'''),
("code", r'''
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
for ax, name in zip(axes, DATA):
    h = pd.DataFrame(HISTS[name][0])
    ax.plot(h["epoch"], h["rmse_train"], "o-", ms=3, label="train")
    ax.plot(h["epoch"], h["rmse_val"], "o-", ms=3, label="validation")
    ax.set(title=name.replace("4 GNN ", ""), xlabel="epoca", yscale="log")
axes[0].set_ylabel("RMSE su H"); axes[0].legend()
plt.tight_layout(); savefig("curve_apprendimento"); plt.show()
'''),
("md", r"""
## 5. Confronto sul test set

Riferimenti dagli altri notebook, sullo stesso test set: baseline, DFA, relazione di Lacasa (approccio 2a),
gradient boosting sulle 16 feature del grafo, CNN (normalizzata per serie).
"""),
("code", r'''
p02 = np.load(Path(DATA_DIR) / f"preds02_{MODE}_seed{SEED}.npz")
p03 = np.load(Path(DATA_DIR) / f"preds03_{MODE}_seed{SEED}.npz")
REF = {"0 baseline (media)": p02, "1a DFA": p02, "2a NVG(fBm) Lacasa": p02,
       "3-GBT feature grafo": p03, "3 CNN (per_serie)": p03}
for ref, src in REF.items():
    PRED[ref] = {s: src[f"{ref}__{s}"] for s in SPLITS}

order = list(REF) + list(DATA)
rows = []
for name in order:
    m = u.regression_metrics(H["test"], PRED[name]["test"])
    lo, hi = u.bootstrap_ci(H["test"], PRED[name]["test"], "RMSE", n_boot=N_BOOT, seed=SEED)
    rows.append({"metodo": name, "RMSE": m["RMSE"], "IC 95% RMSE": f"[{lo:.3f}, {hi:.3f}]", "MAE": m["MAE"],
                 "bias": m["bias"], "R2": m["R2"], "std previsioni": m["std previsioni"],
                 "RMSE val membri": (f"{np.mean(MEMBER_RMSE[name]):.4f} ± {np.std(MEMBER_RMSE[name]):.4f}"
                                     if name in MEMBER_RMSE else "")})
df_res = pd.DataFrame(rows).set_index("metodo")
display(df_res.round(4))
'''),
("code", r'''
show = ["1a DFA", "3-GBT feature grafo", "3 CNN (per_serie)"] + list(DATA)
fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), sharex=True, sharey=True)
for ax, name in zip(axes.flat, show):
    ax.scatter(H["test"], PRED[name]["test"], s=2, alpha=0.3)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(f"{name}\nRMSE={df_res.loc[name, 'RMSE']:.4f}", fontsize=9)
axes.flat[-1].axis("off")
for ax in axes[-1]: ax.set_xlabel("H vero")
for ax in axes[:, 0]: ax.set_ylabel("H previsto")
plt.tight_layout(); savefig("scatter"); plt.show()

bins = np.linspace(0.05, 0.95, 10)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
axes[0].hist(H["test"], bins=40, range=(-0.1, 1.1), histtype="stepfilled", color="lightgrey", label="H vero")
for name in show:
    axes[0].hist(PRED[name]["test"], bins=40, range=(-0.1, 1.1), histtype="step", lw=1.2, label=name)
    be = u.binned_errors(H["test"], PRED[name]["test"], bins)
    axes[1].plot(be["centers"], be["bias"], "o-", ms=3, label=name)
    axes[2].plot(be["centers"], be["rmse"], "o-", ms=3, label=name)
axes[0].set(xlabel="H", title="Distribuzione delle previsioni (test)"); axes[0].legend(fontsize=6)
axes[1].axhline(0, color="k", lw=0.7); axes[1].set(xlabel="H vero", ylabel="bias", title="Bias in funzione di H")
axes[2].set(xlabel="H vero", ylabel="RMSE", title="RMSE in funzione di H"); axes[2].legend(fontsize=6)
plt.tight_layout(); savefig("errore_vs_H"); plt.show()
'''),
("md", r"""
## 6. Invarianze: verifica sulle previsioni dei modelli

Nel notebook 01 abbiamo verificato le invarianze **dei grafi**. Qui verifichiamo che passino ai **modelli
addestrati**: trasformiamo 500 serie del test e misuriamo di quanto cambiano le previsioni.

- **riscalamento** dell'ampiezza: fGn → $3.7\cdot$fGn (equivale a fBm → $3.7\cdot$fBm);
- **trend lineare** sul fBm: $W_H(t)\to W_H(t) + b\,t$, cioè fGn → fGn $+\,b$ (deriva costante), con $b$ pari a
  due deviazioni standard della serie distribuite su tutta la lunghezza.

Attenzione a **su quale serie è costruito il grafo**: un trend lineare del fBm è una *costante* aggiunta al fGn.

| Modello | riscalamento | deriva | perché |
|---|---|---|---|
| GNN NVG(fBm), grado | invariante | invariante | la NVG è invariante per $a x + b t + c$ (notebook 01) |
| GNN NVG(fBm), grado+valori | invariante | **cambia** | i valori del fBm normalizzati per serie cambiano forma con il trend |
| GNN HVG(fGn), grado | invariante | invariante | sul fGn la deriva è una costante: trasformazione monotona |
| GNN HVG(fGn), grado+valori | invariante | invariante | la normalizzazione per serie toglie la costante |
| CNN per serie | invariante | invariante | la normalizzazione toglie media e scala del fGn |
| CNN globale | **cambia** | **cambia** | vede l'ampiezza e la media |

Per i modelli "invarianti" ci aspettiamo variazioni nulle (a meno dell'arrotondamento in virgola mobile).
"""),
("code", r'''
n_inv = 500
X0 = ds["test"]["X"][:n_inv].astype(np.float64)
b = 2 * u.fbm_from_fgn(X0).std(axis=1, keepdims=True) / CFG["N"]
TRANSF = {"riscalamento ×3.7": 3.7 * X0, "deriva (trend lineare del fBm)": X0 + b}

def predict_models(name, X):
    """Ricostruisce la rappresentazione della serie trasformata e predice con l'ensemble del modello."""
    if name.startswith("4 GNN"):
        gname = "NVG(fBm)" if "NVG" in name else "HVG(fGn)"
        kind, series = KINDS[gname]
        e, d, v = u.build_graphs(X, kind, series)
        lk = np.log(np.concatenate(GRAPHS[gname]["train"][1]))
        which = "grado+valori" if "valori" in name else "grado"
        G = u.GraphSet(e, u.node_features(d, v, which, lk.mean(), lk.std()))
        return np.mean([u.predict(m, G) for m in MODELS[name]], axis=0)
    how = "per_serie" if "per_serie" in name else "globale"
    return np.mean([u.predict(m, u.normalize_series(X, how, CNN_SCALE)) for m in CNN_MODELS[how]], axis=0)

# modelli CNN del notebook 03
st = torch.load(Path(DATA_DIR) / f"cnn03_{MODE}_seed{SEED}.pt", map_location=DEV)
cnn_hp = {how: (st[how][0]["features.0.weight"].shape[0]) for how in st}
CNN_MODELS = {}
for how, states in st.items():
    ms = []
    for sd_ in states:
        m = u.build_model("cnn", width=cnn_hp[how]).to(DEV); m.load_state_dict(sd_)
        m.y_mu, m.y_sd = float(H["train"].mean()), float(H["train"].std()); ms.append(m)
    CNN_MODELS[how] = ms
CNN_SCALE = float(ds["train"]["X"].std())

rows = []
for name in list(DATA) + ["3 CNN (per_serie)", "3 CNN (globale)"]:
    p0 = predict_models(name, X0)
    r = {"modello": name}
    for tname, Xt in TRANSF.items():
        r[f"{tname}: max |Δ previsione|"] = np.abs(predict_models(name, Xt) - p0).max()
        r[f"{tname}: RMSE dopo"] = np.sqrt(np.mean((predict_models(name, Xt) - H["test"][:n_inv]) ** 2))
    r["RMSE prima"] = np.sqrt(np.mean((p0 - H["test"][:n_inv]) ** 2))
    rows.append(r)
df_inv = pd.DataFrame(rows).set_index("modello")
display(df_inv.round(4))
'''),
("md", r"""
*Nota tecnica.* La CNN salvata nel notebook 03 contiene solo i pesi; media e deviazione del target usate per
standardizzarlo sono quelle del train ($\bar H$, $\sigma_H$), ricalcolate qui identiche.

## 7. Robustezza rispetto alla lunghezza $N$

Come la CNN, la GNN ha pooling globale e accetta grafi di qualunque dimensione. La standardizzazione di
$\log k$ però usa la media del train, e nella NVG il grado medio cresce con $N$.
"""),
("code", r'''
lengths = sorted([CFG["N"]] + list(CFG["robust_lengths"]))
split_of = {L: ("test" if L == CFG["N"] else f"test_N{L}") for L in lengths}
rob = ["1a DFA", "3 CNN (per_serie)", "3-GBT feature grafo"] + list(DATA)
rows = [{"metodo": n, "N": L, "RMSE": u.regression_metrics(H[split_of[L]], PRED[n][split_of[L]])["RMSE"]}
        for n in rob for L in lengths]
df_rob = pd.DataFrame(rows).pivot(index="metodo", columns="N", values="RMSE")
display(df_rob.round(3))
fig, ax = plt.subplots(figsize=(7, 4.2))
for n in rob:
    ax.plot(lengths, df_rob.loc[n, lengths], "o-", ms=4, label=n)
ax.axvline(CFG["N"], color="grey", lw=0.8, ls=":")
ax.set(xscale="log", yscale="log", xlabel="lunghezza N", ylabel="RMSE (test)", title="Errore in funzione della lunghezza")
ax.legend(fontsize=7); plt.tight_layout(); savefig("rmse_vs_N"); plt.show()
'''),
("md", r"""
## 8. Salvataggio
"""),
("code", r'''
out = Path(DATA_DIR) / f"preds04_{MODE}_seed{SEED}.npz"
np.savez_compressed(out, **{f"{n}__{s}": v for n, d in PRED.items() if n in DATA for s, v in d.items()},
                    **{f"STD::{n}__{s}": v for n, d in PRED_STD.items() for s, v in d.items()})
df_res.to_csv(Path(DATA_DIR) / f"results04_{MODE}.csv")
torch.save({n: [m.state_dict() for m in ms] for n, ms in MODELS.items()}, Path(DATA_DIR) / f"gnn04_{MODE}_seed{SEED}.pt")
print("Salvato", out, "| tempi (s):", {k: round(v) for k, v in TIMES.items()})
'''),
("md", r"""
## 9. Riepilogo
"""),
("code", r'''
for name in DATA:
    r = df_res.loc[name, "RMSE"]
    print(f"{name:<34} RMSE {r:.4f} | vs DFA {100 * (r / df_res.loc['1a DFA', 'RMSE'] - 1):+.0f}% "
          f"| vs feature grafo {100 * (r / df_res.loc['3-GBT feature grafo', 'RMSE'] - 1):+.0f}% "
          f"| vs CNN {100 * (r / df_res.loc['3 CNN (per_serie)', 'RMSE'] - 1):+.0f}%")
'''),
("md", r"""
### Cosa abbiamo imparato (modalità quick: N = 512, 4000 serie di train, 2 semi per variante)

**Prestazioni sul test (RMSE su H, IC 95% bootstrap).**
- La variante migliore è **GNN HVG(fGn) grado+valori: 0.036 [0.035, 0.037]**, scelta anche dal validation (0.038). È alla pari con il GBT sulle 16 feature di grafo (0.037), meglio della DFA (0.053, −32%), ma peggio della CNN sulla serie grezza (0.031, +17%).
- Con il **solo grado** (cioè solo la topologia del grafo) le GNN fanno peggio: NVG 0.052 (come la DFA), HVG 0.091. Il grafo da solo contiene informazione su H, ma una GNN con 3 strati di message passing non la estrae meglio delle feature fatte a mano: il GBT sulle feature di grafo arriva a 0.037.
- Aggiungere i valori della serie ai nodi aiuta molto per l'HVG (0.091 → 0.036) e poco per l'NVG (0.052 → 0.045). L'HVG(fGn) è un grafo sparso (grado medio ≈ 4): con i valori diventa sostanzialmente una rete convoluzionale su un supporto irregolare.
- Costo: 6–13 minuti per variante su CPU, contro circa 2 minuti per la CNN. A parità di dati, la GNN costa di più e non è migliore.

**Invarianze (test fisico chiave, 500 serie di test).**
- Con nodi = solo grado, le previsioni restano **identiche** (max |Δ| = 0) sia dopo un riscalamento ×3.7 sia dopo l'aggiunta di una deriva lineare al fBm. Questo conferma sul modello addestrato le invarianze del grafo verificate nel notebook 01.
- Con i valori della serie (normalizzati per serie), il riscalamento non cambia nulla. La deriva cambia le previsioni della GNN NVG grado+valori (max |Δ| = 0.083): i valori sui nodi rompono l'invarianza dell'NVG rispetto al trend.
- Per confronto, la CNN con normalizzazione globale cambia previsione sotto riscalamento (max |Δ| = 0.54, RMSE 0.029 → 0.36). Usa l'ampiezza, che non dovrebbe contenere informazione su H (D-19, D-26).

**Robustezza alla lunghezza N** (addestramento solo a N = 512).
- Le GNN generalizzano peggio della CNN e del GBT sulle feature di grafo: a N = 2048 l'RMSE **sale** (NVG 0.063–0.070, HVG+valori 0.045), mentre il GBT sulle feature di grafo scende a 0.023.
- Lettura plausibile: grado e readout (media + max) dipendono dalla dimensione del grafo, e il grado massimo cresce con N. La GNN impara la scala di N = 512 invece di una proprietà intensiva.

**Note di onestà.**
- La scelta LayerNorm al posto di BatchNorm (D-35) è stata fatta dopo aver visto i numeri di test della prima esecuzione. È motivata dal validation instabile, ma è registrata.
- Gli iperparametri sono stati scelti sulla sola variante principale (D-34), con una griglia di 2 configurazioni: lo stesso sforzo dato alla CNN.
- In modalità quick ci sono solo 2 semi: la dispersione tra i membri (es. 0.055 ± 0.004 per NVG grado) è indicativa. I numeri definitivi verranno dalla modalità full su Colab.

**Conclusione.** Nella sua forma migliore (HVG + valori), la GNN sul grafo di visibilità è competitiva con le feature di grafo fatte a mano, ma non batte la CNN, costa di più e generalizza peggio in N. Per la regola del corso ("a parità di prestazioni, il più semplice") non è il modello principale. Il suo valore è soprattutto fisico: con il solo grado è un estimatore **dimostrabilmente invariante** per ampiezza e deriva. Nel notebook 05 la migliore variante (HVG grado+valori) viene provata come ibrido con le stime fisiche.
"""),
]

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "04_gnn.ipynb"
    write_notebook(CELLS, out)
    print("Scritto", out)
