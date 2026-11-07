"""Sorgente del notebook 05_ibrido_ablazione.ipynb."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbbuild import COLAB_SETUP, write_notebook

SETUP = COLAB_SETUP.replace(
    'needed = ["numpy", "scipy", "pandas", "matplotlib", "sklearn", "ts2vg"]',
    'needed = ["numpy", "scipy", "pandas", "matplotlib", "sklearn", "ts2vg", "torch", "torch_geometric"]')

CELLS = [
("md", r"""
# 05 — Modello ibrido: fisica + apprendimento

**Idea** (dalla fisica delle particelle: calcolare a mano le masse invarianti prima della regressione rende
il lavoro della rete più facile). La rete riceve la serie grezza **e** le stime fisiche, e impara a
**correggerle** invece di partire da zero.

**Disegno 2×2 (D-12)** — tutte le celle sullo stesso test set:

| | senza feature fisiche | con feature fisiche |
|---|---|---|
| **senza serie grezza** | 0 baseline | modello sulle feature (GBT/MLP, notebook 03; qui anche sui soli 4 stimatori) |
| **con serie grezza** | CNN (notebook 03) | **CNN ibrida** (questo notebook) |

più gli stimatori fisici da soli (DFA grezza e ricalibrata).

**Varianti dell'ibrido**
- **ibrido-min**: la CNN riceve le 4 stime fisiche del piano originale — $\hat H_{\rm DFA}$, $\hat H_{R/S}$,
  $\gamma$ della NVG, $\lambda$ della HVG;
- **ibrido-full**: la CNN riceve tutte le 42 feature (curve DFA e R/S + 16 feature del grafo);
- **ibrido-full-int**: come ibrido-full, ma con le 32 feature indipendenti da $N$ e dall'ampiezza (D-30,
  notebook 03: stime, pendenze locali, livelli a scale assolute $s = 8, 16, 32$, grafo);
- **ibrido-residuo**: come ibrido-min, ma la rete predice la correzione $H - \hat H_{\rm DFA}$;
- **GNN ibrida**: la variante migliore della GNN (notebook 04) con le 4 stime fisiche.

Le feature entrano concatenate alla rappresentazione appresa, subito prima della testa densa (fusione tardiva).
Architettura, iperparametri e seed sono **identici** a quelli della rete senza feature: l'unica differenza è
l'input aggiuntivo (D-36). È questo che rende l'ablazione pulita.

**Contenuto:** 1. Configurazione · 2. Feature fisiche · 3. Modelli sulle sole 4 stime · 4. CNN ibride ·
5. GNN ibrida · 6. Tabella 2×2 e confronto · 7. Quanto si appoggia l'ibrido alla fisica? · 8. Robustezza in $N$ ·
9. Salvataggio · 10. Riepilogo
"""),
("code", SETUP),
("md", r"""
## 1. Configurazione

Gli iperparametri non si cercano di nuovo: si riusano quelli scelti nei notebook 03 e 04 (letti dai pesi salvati).
"""),
("code", r'''
MODE = os.environ.get("SP_MODE", "quick")   # "quick" (CPU, verifica) o "full" (definitivo); il runner Colab imposta SP_MODE=full
SEED = 2026
DATA_DIR = "data"
FIG_DIR = "figures" if MODE == "quick" else f"figures_{MODE}"   # figure separate per modalità
N_BOOT = 1000

if MODE == "quick":
    N_SEEDS, N_SEEDS_GNN = 3, 2
    TRAIN_KW = dict(max_epochs=15, patience=5, batch_size=128, lr=2e-3)       # come notebook 03
    GNN_TRAIN_KW = dict(max_epochs=12, patience=4, batch_size=128, lr=1e-3)   # come notebook 04
else:
    N_SEEDS, N_SEEDS_GNN = 5, 3                                               # GNN come notebook 04 (D-37)
    TRAIN_KW = dict(max_epochs=40, patience=6, batch_size=128, lr=1e-3)
    GNN_TRAIN_KW = dict(max_epochs=25, patience=6, batch_size=128, lr=1e-3)
GBT_GRID = [dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15),
            dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=31)]
'''),
("code", r'''
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

import sp_utils as u

u.set_seed(SEED)
Path(FIG_DIR).mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
def savefig(name):
    plt.savefig(f"{FIG_DIR}/05_{name}.png", bbox_inches="tight", dpi=150)

DEV = u.get_device()
CFG = u.MODES[MODE]
ds = u.load_or_make_fbm_dataset(MODE, SEED, DATA_DIR)
feats = u.load_or_make_features(ds, MODE, SEED, DATA_DIR)
SPLITS = list(ds)
H = {s: ds[s]["H"].astype(np.float64) for s in SPLITS}

# Iperparametri dei notebook precedenti, letti dai pesi salvati
cnn_states = torch.load(Path(DATA_DIR) / f"cnn03_{MODE}_seed{SEED}.pt", map_location="cpu")
CNN_HP = dict(width=int(cnn_states["per_serie"][0]["features.0.weight"].shape[0]))
gnn_states = torch.load(Path(DATA_DIR) / f"gnn04_{MODE}_seed{SEED}.pt", map_location="cpu")
print("Dispositivo:", DEV, "| CNN:", CNN_HP)
'''),
("md", r"""
## 2. Feature fisiche

Le stesse del notebook 02/03, standardizzate con media e deviazione del **train** (valori mancanti → mediana del train).
Sono funzioni della sola serie: usarle come input non introduce alcuna informazione sul vero $H$.
"""),
("code", r'''
col = {n: j for j, n in enumerate(u.GRAPH_FEATURES)}
def phys(f, X, group):
    if group == "min":
        return np.column_stack([f["dfa"], f["rs"], f["graph"][:, col["nvg_fbm_gamma"]], f["graph"][:, col["hvg_fgn_lambda"]]])
    if group == "full-int":     # versione indipendente da N (D-30): come "tutte-int" del notebook 03
        return np.column_stack([u.classic_intensive_features(f, X), f["graph"]])
    return np.column_stack([f["dfa"], f["rs"], f["logF"], f["logRS"], f["graph"]])

E = {}
for group in ["min", "full", "full-int"]:
    raw = {s: phys(feats[s], ds[s]["X"], group) for s in SPLITS}
    sc = u.FeatureScaler().fit(raw["train"])
    E[group] = {s: sc.transform(raw[s]) for s in SPLITS}
    print(f"ibrido-{group}: {E[group]['train'].shape[1]} feature fisiche")
XN = {s: u.normalize_series(ds[s]["X"], "per_serie") for s in SPLITS}
DFA = {s: feats[s]["dfa"] for s in SPLITS}
'''),
("md", r"""
## 3. Modello sulle sole 4 stime fisiche

Per completare la cella "senza serie, con feature" anche per ibrido-min: gradient boosting sulle 4 stime.
Risponde a "quanto si guadagna combinando gli stimatori fisici, senza guardare la serie?".
"""),
("code", r'''
PRED, PRED_STD, MEMBER_RMSE, TIMES = {}, {}, {}, {}
best = None
for hp in GBT_GRID:
    g = HistGradientBoostingRegressor(early_stopping=False, random_state=SEED, **hp).fit(E["min"]["train"], H["train"])
    val = u.regression_metrics(H["val"], g.predict(E["min"]["val"]))["RMSE"]
    print(hp, f"RMSE val {val:.4f}")
    if best is None or val < best[1]:
        best = (hp, val, g)
PRED["5-GBT 4 stime fisiche"] = {s: best[2].predict(E["min"][s]) for s in SPLITS}
'''),
("md", r"""
## 4. CNN ibride

Stessa CNN del notebook 03 (stessa larghezza, stessi seed, stesso addestramento), con `n_extra` feature
concatenate prima della testa densa. Per la variante a residuo il target è $H - \hat H_{\rm DFA}$ e la previsione
finale è $\hat H_{\rm DFA}$ + uscita della rete; l'RMSE sul residuo coincide con l'RMSE su $H$.
"""),
("code", r'''
def fit_ensemble(name, make_model, Xs, target, extra, n_seeds, train_kw, offset=None):
    """Addestra n_seeds modelli; salva media e deviazione delle previsioni (+ offset per il residuo)."""
    t0 = time.time()
    members = []
    for k in range(n_seeds):
        m, _ = u.train_regressor(make_model(), Xs["train"], target["train"], Xs["val"], target["val"],
                                 E_tr=extra["train"] if extra else None, E_val=extra["val"] if extra else None,
                                 seed=SEED + 100 * k, **train_kw)
        members.append(m)
    TIMES[name] = time.time() - t0
    P = {s: np.stack([u.predict(m, Xs[s], extra[s] if extra else None) for m in members])
         + (offset[s] if offset else 0) for s in SPLITS}
    PRED[name] = {s: P[s].mean(0) for s in SPLITS}
    PRED_STD[name] = {s: P[s].std(0) for s in SPLITS}
    MEMBER_RMSE[name] = [u.regression_metrics(H["val"], P["val"][k])["RMSE"] for k in range(n_seeds)]
    print(f"{name:<32} membri (val) {np.round(MEMBER_RMSE[name], 4)}  ensemble (val) "
          f"{u.regression_metrics(H['val'], PRED[name]['val'])['RMSE']:.4f}  ({TIMES[name]:.0f} s)")
    return members

HYB = {}
for group in ["min", "full", "full-int"]:
    n_extra = E[group]["train"].shape[1]
    HYB[group] = fit_ensemble(f"5 CNN ibrida-{group}", lambda: u.build_model("cnn", n_extra=n_extra, **CNN_HP),
                              XN, H, E[group], N_SEEDS, TRAIN_KW)
resid = {s: H[s] - DFA[s] for s in SPLITS}
HYB["residuo"] = fit_ensemble("5 CNN ibrida-residuo", lambda: u.build_model("cnn", n_extra=4, **CNN_HP),
                              XN, resid, E["min"], N_SEEDS, TRAIN_KW, offset=DFA)
'''),
("md", r"""
## 5. GNN ibrida

Si sceglie la variante della GNN migliore **sul validation** (notebook 04) e la si riaddestra identica con in più
le 4 stime fisiche.
"""),
("code", r'''
p04 = np.load(Path(DATA_DIR) / f"preds04_{MODE}_seed{SEED}.npz")
gnn_names = sorted({k.split("__")[0] for k in p04.files if not k.startswith("STD::")})
val_rmse = {n: u.regression_metrics(H["val"], p04[f"{n}__val"])["RMSE"] for n in gnn_names}
display(pd.Series(val_rmse, name="RMSE val").round(4))
BEST_GNN = min(val_rmse, key=val_rmse.get)
print("Variante scelta:", BEST_GNN)

kind, series = ("nvg", "fbm") if "NVG" in BEST_GNN else ("hvg", "fgn")
which = "grado+valori" if "valori" in BEST_GNN else "grado"
st = gnn_states[BEST_GNN][0]
GNN_HP = dict(hidden=int(st["convs.0.nn.0.weight"].shape[0]),
              n_layers=len({k.split(".")[1] for k in st if k.startswith("convs.")}))
print("Iperparametri GNN:", GNN_HP)

t0 = time.time()
graphs = {s: u.build_graphs(ds[s]["X"], kind, series) for s in SPLITS}
lk = np.log(np.concatenate(graphs["train"][1]))
G = {s: u.GraphSet(graphs[s][0], u.node_features(graphs[s][1], graphs[s][2], which, lk.mean(), lk.std())) for s in SPLITS}
n_in = G["train"].x[0].shape[1]
print(f"Grafi costruiti in {time.time() - t0:.0f} s")
fit_ensemble(f"5 GNN ibrida ({BEST_GNN.replace('4 GNN ', '')})",
             lambda: u.build_gnn(n_in=n_in, n_extra=4, **GNN_HP), G, H, E["min"], N_SEEDS_GNN, GNN_TRAIN_KW)
GNN_HYB = f"5 GNN ibrida ({BEST_GNN.replace('4 GNN ', '')})"
'''),
("md", r"""
## 6. Tabella 2×2 e confronto sul test
"""),
("code", r'''
p02 = np.load(Path(DATA_DIR) / f"preds02_{MODE}_seed{SEED}.npz")
p03 = np.load(Path(DATA_DIR) / f"preds03_{MODE}_seed{SEED}.npz")
for ref, src in [("0 baseline (media)", p02), ("1a DFA", p02), ("1a-cal DFA ricalibrata", p02),
                 ("3-GBT feature tutte", p03), ("3-GBT feature tutte-int", p03), ("3 CNN (per_serie)", p03)]:
    PRED[ref] = {s: src[f"{ref}__{s}"] for s in SPLITS}
PRED[BEST_GNN] = {s: p04[f"{BEST_GNN}__{s}"] for s in SPLITS}

order = ["0 baseline (media)", "1a DFA", "1a-cal DFA ricalibrata", "5-GBT 4 stime fisiche", "3-GBT feature tutte",
         "3-GBT feature tutte-int", "3 CNN (per_serie)", "5 CNN ibrida-min", "5 CNN ibrida-full", "5 CNN ibrida-full-int",
         "5 CNN ibrida-residuo", BEST_GNN, GNN_HYB]
rows = []
for name in order:
    m = u.regression_metrics(H["test"], PRED[name]["test"])
    lo, hi = u.bootstrap_ci(H["test"], PRED[name]["test"], "RMSE", n_boot=N_BOOT, seed=SEED)
    rows.append({"metodo": name, "RMSE": m["RMSE"], "IC 95% RMSE": f"[{lo:.4f}, {hi:.4f}]", "MAE": m["MAE"],
                 "bias": m["bias"], "R2": m["R2"], "std previsioni": m["std previsioni"]})
df_res = pd.DataFrame(rows).set_index("metodo")
display(df_res.round(4))
'''),
("code", r'''
R = df_res["RMSE"]
table = pd.DataFrame({
    "senza feature fisiche": [f"baseline: {R['0 baseline (media)']:.4f}",
                              f"CNN: {R['3 CNN (per_serie)']:.4f}"],
    "con 4 stime fisiche": [f"GBT su 4 stime: {R['5-GBT 4 stime fisiche']:.4f}",
                            f"CNN ibrida-min: {R['5 CNN ibrida-min']:.4f}  (residuo: {R['5 CNN ibrida-residuo']:.4f})"],
    "con 42 feature fisiche": [f"GBT su 42 feature: {R['3-GBT feature tutte']:.4f}",
                               f"CNN ibrida-full: {R['5 CNN ibrida-full']:.4f}"],
    "con 32 feature indipendenti da N": [f"GBT su 32 feature: {R['3-GBT feature tutte-int']:.4f}",
                                         f"CNN ibrida-full-int: {R['5 CNN ibrida-full-int']:.4f}"],
}, index=["senza serie grezza", "con serie grezza"])
print(f"Solo fisica, nessun apprendimento: DFA {R['1a DFA']:.4f} (ricalibrata {R['1a-cal DFA ricalibrata']:.4f})")
print(f"GNN: {BEST_GNN} {R[BEST_GNN]:.4f}  ->  ibrida {R[GNN_HYB]:.4f}")
display(table)
'''),
("code", r'''
show = ["1a DFA", "3 CNN (per_serie)", "5 CNN ibrida-min", "5 CNN ibrida-full", "5 CNN ibrida-residuo", GNN_HYB]
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
for ax, name in zip(axes.flat, show):
    ax.scatter(H["test"], PRED[name]["test"], s=2, alpha=0.3)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(f"{name}\nRMSE={R[name]:.4f}", fontsize=9)
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
## 7. Quanto si appoggia l'ibrido alla fisica?

**Test di permutazione.** Sul test set rimescoliamo a caso, tra le serie, una feature fisica alla volta (la serie
grezza resta al suo posto) e misuriamo quanto peggiora l'RMSE. Se l'ibrido ignorasse la fisica, l'RMSE non
cambierebbe; se dipendesse solo da essa, crescerebbe molto. Rimescoliamo anche tutte le feature insieme:
l'ibrido degrada con grazia (la serie grezza basta ancora) o crolla?
"""),
("code", r'''
rng_p = np.random.default_rng(SEED + 55)
names_min = ["H DFA", "H R/S", "gamma NVG(fBm)", "lambda HVG(fGn)"]
def ens_rmse(members, Xs, Es, offset=0):
    p = np.mean([u.predict(m, Xs, Es) for m in members], axis=0) + offset
    return u.regression_metrics(H["test"], p)["RMSE"]

rows = []
for label, key, off in [("ibrida-min", "min", 0), ("ibrida-residuo", "residuo", DFA["test"])]:
    Et = E["min"]["test"]
    base = ens_rmse(HYB[key], XN["test"], Et, off)
    r = {"modello": label, "RMSE": base}
    for j, fn in enumerate(names_min):
        Ep = Et.copy(); Ep[:, j] = rng_p.permutation(Ep[:, j])
        r[f"rimescolata: {fn}"] = ens_rmse(HYB[key], XN["test"], Ep, off)
    r["rimescolate tutte"] = ens_rmse(HYB[key], XN["test"], rng_p.permutation(Et), off)
    rows.append(r)
df_perm = pd.DataFrame(rows).set_index("modello")
display(df_perm.round(4))
'''),
("md", r"""
*Nota sulla variante a residuo.* Rimescolando le feature non si rimescola l'offset $\hat H_{\rm DFA}$ aggiunto
all'uscita: la variante a residuo resta comunque ancorata alla DFA della serie giusta.

## 8. Robustezza rispetto a $N$

Le feature dell'ibrido-full contengono le curve DFA/R-S su scale relative a $N$ (D-30): ci aspettiamo che
l'ibrido-full erediti questo limite. L'ibrido-min usa solo pendenze ed esponenti, l'ibrido-full-int le feature
costruite per essere indipendenti da $N$.
"""),
("code", r'''
lengths = sorted([CFG["N"]] + list(CFG["robust_lengths"]))
split_of = {L: ("test" if L == CFG["N"] else f"test_N{L}") for L in lengths}
rob = ["1a DFA", "3 CNN (per_serie)", "3-GBT feature tutte-int", "5 CNN ibrida-min", "5 CNN ibrida-full",
       "5 CNN ibrida-full-int", "5 CNN ibrida-residuo", GNN_HYB]
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
## 9. Salvataggio
"""),
("code", r'''
mine = [n for n in PRED if n.startswith("5")]
out = Path(DATA_DIR) / f"preds05_{MODE}_seed{SEED}.npz"
np.savez_compressed(out, **{f"{n}__{s}": PRED[n][s] for n in mine for s in SPLITS},
                    **{f"STD::{n}__{s}": v for n, d in PRED_STD.items() for s, v in d.items()})
df_res.to_csv(Path(DATA_DIR) / f"results05_{MODE}.csv")
print("Salvato", out, "| tempi (s):", {k: round(v) for k, v in TIMES.items()})
'''),
("md", r"""
## 10. Riepilogo
"""),
("code", r'''
for a, b in [("3 CNN (per_serie)", "5 CNN ibrida-min"), ("3 CNN (per_serie)", "5 CNN ibrida-full"),
             ("3 CNN (per_serie)", "5 CNN ibrida-residuo"), ("1a DFA", "5 CNN ibrida-min"),
             ("3-GBT feature tutte", "5 CNN ibrida-full"), ("3-GBT feature tutte-int", "5 CNN ibrida-full-int"),
             ("3 CNN (per_serie)", "5 CNN ibrida-full-int"), (BEST_GNN, GNN_HYB)]:
    print(f"{b:<40} vs {a:<26}: {100 * (R[b] / R[a] - 1):+5.0f}%")
'''),
("md", r"""
### Cosa abbiamo imparato (modalità quick: N = 512, 3 semi per le CNN, 2 per la GNN)

**Tabella 2×2 (RMSE di test su H).**

| | senza feature fisiche | 4 stime fisiche | 42 feature (curve, legate a N) | 32 feature indipendenti da N |
|---|---|---|---|---|
| senza serie grezza | baseline 0.259 | GBT 0.045 | GBT 0.026 | GBT 0.030 |
| con serie grezza | CNN 0.031 | CNN ibrida-min 0.032 | CNN ibrida-full 0.026 | CNN ibrida-full-int 0.029 |

- **Le 4 stime fisiche non aggiungono nulla alla CNN a N = 512.** L'ibrida-min (0.032 [0.031, 0.033]) è alla pari con la CNN da sola (0.031 [0.030, 0.032]). La CNN estrae già dalla serie l'informazione contenuta in DFA e R/S.
- **L'ibrida-full (0.0256 [0.025, 0.026]) è la migliore in assoluto**, ma gli intervalli si sovrappongono con il GBT sulle stesse 42 feature senza serie grezza (0.0263 [0.025, 0.027]). Per la regola del corso, a parità di prestazioni vince il modello più semplice: il GBT sulle feature, senza rete convoluzionale.
- **La variante residuo** (la rete impara la correzione H − H_DFA) è peggiore (0.034). Partire dalla DFA come offset non aiuta: la correzione da imparare non è più semplice di H stesso.
- **GNN ibrida:** 0.036 → 0.035 (−3%), entro l'incertezza. Nessun guadagno chiaro.

**Il modello usa davvero la fisica? (test di permutazione).**
- Rimescolando una stima alla volta sul test, l'RMSE dell'ibrida-min sale molto per H_DFA (0.032 → 0.060) e H_R/S (→ 0.062), un po' per λ HVG (→ 0.041), e non cambia per γ NVG (0.032). Lo stimatore di Lacasa, già fallito nel notebook 02, viene ignorato.
- Rimescolandole tutte, l'RMSE arriva a 0.121, molto peggio della CNN senza stime (0.031). La rete **si appoggia** sulle stime fisiche ed è ridondante con la serie, ma non ne trae un'informazione nuova: a parità di prestazioni, ha spostato parte del lavoro sulle stime.

**Robustezza in N: qui l'ibrido fa la differenza.**
- L'ibrida-min generalizza **meglio di tutti** verso serie più lunghe: 0.023 a N = 1024 e 0.020 a N = 2048, contro 0.028 e 0.031 della CNN e 0.034 della DFA. Le stime DFA/R-S sono intensive (migliorano con N), e la rete che si appoggia su di esse eredita questa proprietà.
- L'ibrida-full **collassa** (0.148 a N = 2048), come il GBT sulle feature "tutte" (0.136, notebook 03). Le curve log F(s) e log R/S sono calcolate su scale che dipendono da N (D-30), quindi il modello impara il loro aspetto a N = 512.
- **Con le feature indipendenti da N (D-30) il collasso sparisce.** L'ibrida-full-int vale 0.068, 0.043, 0.029, 0.022, 0.021 per N = 128 … 2048: è tra i migliori a ogni lunghezza (il migliore per N ≤ 1024), anche verso serie più corte, dove ibrida-min (0.088) e CNN (0.079) sono peggiori.
- Il GBT sulle stesse 32 feature, senza serie grezza, è quasi identico: 0.072, 0.045, 0.030, 0.023, 0.018. A N = 512 l'ibrida-full-int è meglio del 4% (0.0286 [0.028, 0.030] contro 0.0299 [0.029, 0.031]), con intervalli che si toccano.

**Conclusione.**
- A lunghezza fissa, aggiungere la fisica come input migliora la CNN di poco: −7% con le 32 feature indipendenti da N, −17% con le curve legate a N, che però crollano appena N cambia. In ogni cella con feature, il GBT sulle sole feature è a pochi punti percentuali dalla CNN ibrida corrispondente.
- **Scelta per la regola del corso** (prestazioni simili → il più semplice): il **GBT sulle 32 feature indipendenti da N**. Ha la stessa accuratezza della CNN a N = 512, è più robusto della CNN a ogni altra lunghezza, si addestra in secondi e non usa l'ampiezza. La CNN ibrida-full-int è il miglior modello neurale, con un vantaggio piccolo.
- Il valore dell'informazione fisica sta soprattutto nella **robustezza**. Le stime e le pendenze sono grandezze intensive, e i modelli che le usano generalizzano a lunghezze mai viste; le curve su scale relative a N no. Il punto non è "fisica sì o no", ma **quale rappresentazione** della fisica.
- Caveat: modalità quick, pochi semi; i confronti entro ~5% non sono significativi. Numeri definitivi dalla modalità full. La forma delle feature indipendenti da N è stata scelta dopo una prova preliminare in cui si vedeva anche il test (D-30).
"""),
]

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "05_ibrido_ablazione.ipynb"
    write_notebook(CELLS, out)
    print("Scritto", out)
