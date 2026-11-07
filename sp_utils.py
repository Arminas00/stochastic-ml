"""
sp_utils.py — codice condiviso dal progetto.

Notazione come negli appunti di Econofisica (par. 6.1):
  W_H(t)   moto browniano frazionario (fBm), <W_H(t) W_H(s)> = (t^2H + s^2H - |t-s|^2H) / 2
  dW_H(t)  rumore gaussiano frazionario (fGn), dW_H(t) = W_H(t+1) - W_H(t)

Sezioni:
  1. Configurazione e riproducibilità
  2. Simulazione: fGn / fBm (metodo di Davies–Harte)
  3. Grafi di visibilità (NVG, HVG) e verifiche
  4. Dataset: generazione, split, salvataggio
  5. Stimatori classici di H: DFA, R/S
  6. Stime e feature dai grafi di visibilità
  7. Metriche
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# =============================================================================
# 1. Configurazione e riproducibilità
# =============================================================================

# Dimensioni dei dataset per le due modalità (vedi DECISION_LOG D-03).
MODES = {
    "quick": dict(N=512, n_train=4_000, n_val=1_000, n_test=2_000,
                  robust_lengths=(128, 256, 1024, 2048), n_robust=500),
    "full": dict(N=1024, n_train=40_000, n_val=5_000, n_test=10_000,
                 robust_lengths=(128, 256, 512, 2048, 4096), n_robust=2_000),
}

H_RANGE = (0.05, 0.95)  # D-10


def set_seed(seed: int) -> None:
    """Fissa i seed di random, numpy e (se presente) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# =============================================================================
# 2. Simulazione: rumore gaussiano frazionario e moto browniano frazionario
# =============================================================================

def fgn_autocov(k, H):
    """Autocovarianza esatta del fGn a varianza unitaria:
    gamma(k) = ( |k+1|^2H - 2|k|^2H + |k-1|^2H ) / 2   (appunti, par. 6.1)."""
    k = np.abs(np.asarray(k, dtype=float))
    H = np.asarray(H, dtype=float)
    return 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))


def davies_harte_eigenvalues(n: int, H) -> np.ndarray:
    """Autovalori della matrice circolante di dimensione 2n che contiene la
    covarianza di Toeplitz del fGn. Il metodo è esatto se sono tutti >= 0,
    cosa dimostrata per il fGn con H in (0,1) (verificata numericamente in 01)."""
    H = np.atleast_1d(np.asarray(H, dtype=float))[:, None]          # (B,1)
    k = np.arange(n + 1)[None, :]                                    # (1,n+1)
    g = fgn_autocov(k, H)                                            # (B,n+1)
    c = np.concatenate([g, g[:, -2:0:-1]], axis=1)                   # (B,2n): g0..gn, g(n-1)..g1
    return np.fft.fft(c, axis=1).real


def simulate_fgn(n: int, H, rng: np.random.Generator, batch: int = 2_000) -> np.ndarray:
    """Simula fGn esatto a varianza unitaria con il metodo di Davies–Harte.

    H può essere uno scalare o un array (un valore per serie). Restituisce un
    array (len(H), n) in float64.

    Con z complesso a parti reale e immaginaria N(0,1) indipendenti,
    y = FFT( sqrt(lambda/2n) z ) ha parte reale con covarianza esattamente pari
    alla covarianza circolante, quindi i primi n valori sono fGn esatto.
    (La parte immaginaria sarebbe una seconda serie indipendente con lo stesso H;
    poiché ogni serie ha il suo H, la scartiamo: costo x2, ma codice semplice.)
    """
    H = np.atleast_1d(np.asarray(H, dtype=float))
    out = np.empty((len(H), n))
    m = 2 * n
    for start in range(0, len(H), batch):
        Hb = H[start:start + batch]
        lam = davies_harte_eigenvalues(n, Hb)
        if lam.min() < -1e-8 * lam.max():
            raise ValueError("Autovalori negativi: embedding circolante non valido")
        lam = np.clip(lam, 0.0, None)
        nb = len(Hb)
        z = rng.standard_normal((nb, m)) + 1j * rng.standard_normal((nb, m))
        y = np.fft.fft(np.sqrt(lam / m) * z, axis=1)
        out[start:start + nb] = y.real[:, :n]
    return out


def fbm_from_fgn(fgn: np.ndarray) -> np.ndarray:
    """fBm come somma cumulata del fGn, con W_H(0) = 0 incluso:
    restituisce N+1 punti W_H(0), W_H(1), ..., W_H(N)."""
    zeros = np.zeros(fgn.shape[:-1] + (1,), dtype=fgn.dtype)
    return np.concatenate([zeros, np.cumsum(fgn, axis=-1)], axis=-1)


def sample_autocov(x: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Autocovarianza d'insieme <x_t x_{t+h}> (media nota = 0), mediata sui tempi
    e sulle serie. Restituisce stima e errore standard (dalla dispersione tra
    le serie, che sono indipendenti)."""
    per_series = np.stack(
        [(x[:, : x.shape[1] - h] * x[:, h:]).mean(axis=1) for h in range(max_lag + 1)], axis=1
    )
    return per_series.mean(axis=0), per_series.std(axis=0, ddof=1) / np.sqrt(len(x))


# =============================================================================
# 3. Grafi di visibilità
# =============================================================================

def visibility_graph(x: np.ndarray, kind: str = "hvg"):
    """Costruisce il grafo di visibilità naturale ('nvg') od orizzontale ('hvg')
    con ts2vg. Restituisce (edges array (E,2) int, degrees array (N,))."""
    from ts2vg import HorizontalVG, NaturalVG
    g = HorizontalVG() if kind == "hvg" else NaturalVG()
    g.build(np.asarray(x, dtype=float))
    return np.asarray(g.edges, dtype=np.int64).reshape(-1, 2), np.asarray(g.degrees)


def visibility_edges_bruteforce(x: np.ndarray, kind: str = "hvg") -> set:
    """Implementazione diretta O(N^3) dalla definizione (Lacasa et al.),
    usata solo per verificare ts2vg su serie corte.
      NVG: (i,j) visibili se  x_k < x_j + (x_i - x_j) (j-k)/(j-i)  per ogni i<k<j
      HVG: (i,j) visibili se  x_k < min(x_i, x_j)                 per ogni i<k<j
    """
    n = len(x)
    edges = set()
    for i in range(n - 1):
        for j in range(i + 1, n):
            ks = np.arange(i + 1, j)
            if kind == "hvg":
                ok = np.all(x[ks] < min(x[i], x[j]))
            else:
                ok = np.all(x[ks] < x[j] + (x[i] - x[j]) * (j - ks) / (j - i))
            if ok:
                edges.add((i, j))
    return edges


def hvg_degree_pmf_iid(k):
    """Distribuzione esatta del grado dell'HVG per rumore i.i.d. continuo
    (Luque, Lacasa et al. 2009): P(k) = (1/3) (2/3)^(k-2), k >= 2.
    Vale per i nodi interni di una serie infinita; non dipende dalla
    distribuzione del rumore."""
    k = np.asarray(k)
    return np.where(k >= 2, (1 / 3) * (2 / 3) ** (k - 2.0), 0.0)


def hvg_mean_degree_iid(n: int) -> float:
    """Grado medio esatto dell'HVG di N valori i.i.d. continui (con bordi).
    Due nodi a distanza d sono visibili se sono i due massimi tra i d+1 valori
    da i a j: probabilità 2/(d(d+1)). Quindi E[archi] = sum_d (N-d) 2/(d(d+1))."""
    d = np.arange(1, n)
    n_edges = np.sum((n - d) * 2.0 / (d * (d + 1)))
    return 2 * n_edges / n


# =============================================================================
# 4. Dataset
# =============================================================================

def make_fgn_split(n_series: int, N: int, seed: int) -> dict:
    """Genera uno split: H ~ U(H_RANGE), fGn esatto di lunghezza N (float32)."""
    rng = np.random.default_rng(seed)
    H = rng.uniform(*H_RANGE, size=n_series)
    X = simulate_fgn(N, H, rng).astype(np.float32)
    return {"H": H.astype(np.float32), "X": X}


def load_or_make_fbm_dataset(mode: str, seed: int, data_dir: str | Path = "data") -> dict:
    """Carica il dataset fGn/fBm della modalità richiesta, o lo genera e lo salva.

    I tre split usano seed diversi (seed, seed+1, seed+2): sono generati in modo
    indipendente, quindi nessuna serie può comparire in due split.
    I set di robustezza (altre lunghezze) usano seed+100+L.
    """
    cfg = MODES[mode]
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"fbm_{mode}_seed{seed}.npz"
    if path.exists():
        z = np.load(path)
        ds = {}
        for key in z.files:
            split, field = key.split("__")
            ds.setdefault(split, {})[field] = z[key]
        return ds

    ds = {
        "train": make_fgn_split(cfg["n_train"], cfg["N"], seed),
        "val": make_fgn_split(cfg["n_val"], cfg["N"], seed + 1),
        "test": make_fgn_split(cfg["n_test"], cfg["N"], seed + 2),
    }
    for L in cfg["robust_lengths"]:
        ds[f"test_N{L}"] = make_fgn_split(cfg["n_robust"], L, seed + 100 + L)
    np.savez_compressed(path, **{f"{s}__{f}": v for s, d in ds.items() for f, v in d.items()})
    return ds


# =============================================================================
# 5. Stimatori classici di H
# =============================================================================

def log_scales(N: int, s_min: int = 8, s_max_frac: float = 0.25, n_scales: int = 12) -> np.ndarray:
    """Scale (lunghezze di finestra) equispaziate in log tra s_min e N*s_max_frac, intere e distinte."""
    s = np.unique(np.round(np.geomspace(s_min, max(s_min * 2, N * s_max_frac), n_scales)).astype(int))
    return s


def _slope(logx: np.ndarray, logy: np.ndarray) -> np.ndarray:
    """Pendenza dei minimi quadrati di logy (B, S) contro logx (S,), riga per riga."""
    xc = logx - logx.mean()
    return (logy - logy.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)


def dfa_fluctuation(x: np.ndarray, scales: np.ndarray, order: int = 1) -> np.ndarray:
    """Funzione di fluttuazione F(s) della DFA di ordine `order` (Peng et al. 1994).

    x: (B, N) incrementi (fGn). Il profilo è y = cumsum(x - media). Per ogni scala s
    il profilo si divide in finestre non sovrapposte di lunghezza s (da sinistra e da
    destra, per usare tutti i punti), si sottrae il polinomio di grado `order` fittato
    in ogni finestra e si calcola la varianza residua media. F(s) ~ s^alpha, alpha = H.
    Restituisce (B, len(scales)).
    """
    x = np.asarray(x, dtype=np.float64)
    B, N = x.shape
    y = np.cumsum(x - x.mean(axis=1, keepdims=True), axis=1)
    F = np.empty((B, len(scales)))
    for j, s in enumerate(scales):
        nw = N // s
        segs = np.concatenate([y[:, : nw * s].reshape(B, nw, s),
                               y[:, N - nw * s:].reshape(B, nw, s)], axis=1)       # (B, 2nw, s)
        t = np.arange(s, dtype=np.float64)
        V = np.vander((t - t.mean()) / s, order + 1)                                # (s, order+1)
        P = V @ np.linalg.pinv(V)                                                   # proiettore sui polinomi
        resid = segs - segs @ P.T
        F[:, j] = np.sqrt((resid ** 2).mean(axis=(1, 2)))
    return F


def dfa_hurst(x: np.ndarray, scales: np.ndarray | None = None, order: int = 1) -> np.ndarray:
    """Stima di H come pendenza di log F(s) contro log s."""
    if scales is None:
        scales = log_scales(x.shape[1])
    F = dfa_fluctuation(x, scales, order)
    return _slope(np.log(scales), np.log(F))


def rs_statistic(x: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Statistica R/S media (Hurst 1951; Mandelbrot–Wallis) per ogni scala n.

    Per ogni finestra di lunghezza n: Z = somma cumulata degli scarti dalla media della
    finestra, R = max Z - min Z, S = deviazione standard della finestra; si media R/S
    sulle finestre. <R/S>(n) ~ n^H. Restituisce (B, len(scales))."""
    x = np.asarray(x, dtype=np.float64)
    B, N = x.shape
    RS = np.empty((B, len(scales)))
    for j, n in enumerate(scales):
        nw = N // n
        segs = x[:, : nw * n].reshape(B, nw, n)
        dev = segs - segs.mean(axis=2, keepdims=True)
        Z = np.cumsum(dev, axis=2)
        R = Z.max(axis=2) - Z.min(axis=2)
        S = segs.std(axis=2)
        RS[:, j] = (R / S).mean(axis=1)
    return RS


def rs_hurst(x: np.ndarray, scales: np.ndarray | None = None) -> np.ndarray:
    """Stima di H come pendenza di log <R/S>(n) contro log n."""
    if scales is None:
        scales = log_scales(x.shape[1])
    return _slope(np.log(scales), np.log(rs_statistic(x, scales)))


# =============================================================================
# 6. Stime e feature dai grafi di visibilità
# =============================================================================

def powerlaw_mle(k: np.ndarray, min_tail: int = 50, max_kmin_quantile: float = 0.9):
    """Esponente gamma di una coda a legge di potenza P(k) ~ k^-gamma (Clauset, Shalizi, Newman 2009).

    Per ogni k_min candidato: stima di massima verosimiglianza (approssimazione continua
    con correzione discreta k_min - 1/2)
        gamma = 1 + n / sum ln( k_i / (k_min - 1/2) ),
    e distanza di Kolmogorov–Smirnov tra coda empirica e legge fittata; si sceglie il
    k_min con KS minima, purché la coda abbia almeno `min_tail` punti.
    Restituisce (gamma, k_min, KS); NaN se nessun k_min è ammissibile."""
    k = np.sort(np.asarray(k, dtype=np.float64))
    cands = np.unique(k[k <= np.quantile(k, max_kmin_quantile)])
    cands = cands[cands >= 2]
    best = (np.nan, np.nan, np.inf)
    for kmin in cands:
        tail = k[k >= kmin]
        n = len(tail)
        if n < min_tail:
            break
        g = 1 + n / np.log(tail / (kmin - 0.5)).sum()
        # CDF empirica e teorica (continua) sui valori distinti della coda
        vals, counts = np.unique(tail, return_counts=True)
        emp = np.cumsum(counts) / n
        theo = 1 - ((vals + 0.5) / (kmin - 0.5)) ** (1 - g)
        ks = np.abs(emp - theo).max()
        if ks < best[2]:
            best = (g, kmin, ks)
    return best


def exponential_tail_mle(k: np.ndarray, kmin: int = 4) -> float:
    """Esponente lambda di una coda esponenziale P(k) ~ exp(-lambda k) per k >= kmin.
    k - kmin è geometrica: MLE p = 1/(1 + media(k - kmin)), lambda = -ln(1 - p).
    Per rumore i.i.d. l'HVG ha lambda = ln(3/2) ≈ 0.405 esattamente."""
    tail = np.asarray(k, dtype=np.float64)
    tail = tail[tail >= kmin] - kmin
    if len(tail) == 0:
        return np.nan
    p = 1.0 / (1.0 + tail.mean())
    return -np.log(1.0 - p) if p < 1 else np.nan   # coda degenere (tutti k = kmin)


def hurst_from_nvg_gamma(gamma, series: str = "fbm"):
    """Relazione di Lacasa et al. (EPL 86, 30001, 2009): NVG di fBm gamma = 3 - 2H;
    per il fGn (incrementi) gamma = 5 - 2H. Si inverte per H."""
    gamma = np.asarray(gamma, dtype=float)
    return (3.0 - gamma) / 2.0 if series == "fbm" else (5.0 - gamma) / 2.0


GRAPH_FEATURES = [
    "nvg_fbm_gamma", "nvg_fbm_kmean", "nvg_fbm_kstd", "nvg_fbm_logkmax",
    "nvg_fgn_gamma", "nvg_fgn_kmean", "nvg_fgn_kstd", "nvg_fgn_logkmax",
    "hvg_fbm_lambda", "hvg_fbm_kstd", "hvg_fbm_p2", "hvg_fbm_p3",
    "hvg_fgn_lambda", "hvg_fgn_kstd", "hvg_fgn_p2", "hvg_fgn_p3",
]


def graph_features_one(x_fgn: np.ndarray) -> np.ndarray:
    """Feature costruite a mano dai quattro grafi di una serie: NVG e HVG del fBm
    (profilo) e del fGn (incrementi). Ordine come GRAPH_FEATURES."""
    from ts2vg import HorizontalVG, NaturalVG
    out = []
    w = fbm_from_fgn(np.asarray(x_fgn, dtype=np.float64))
    for kind, cls in (("nvg", NaturalVG), ("hvg", HorizontalVG)):
        for series in (w, x_fgn):
            g = cls(); g.build(np.asarray(series, dtype=np.float64))
            k = np.asarray(g.degrees)
            if kind == "nvg":
                gamma, _, _ = powerlaw_mle(k)
                out += [gamma, k.mean(), k.std(), np.log(k.max())]
            else:
                out += [exponential_tail_mle(k), k.std(), (k == 2).mean(), (k == 3).mean()]
    return np.array(out)


def graph_features(X: np.ndarray, verbose_every: int = 0) -> np.ndarray:
    """graph_features_one su tutte le righe di X (B, N) -> (B, len(GRAPH_FEATURES))."""
    F = np.empty((len(X), len(GRAPH_FEATURES)))
    for i, x in enumerate(X):
        F[i] = graph_features_one(x)
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  {i + 1}/{len(X)}")
    return F


# =============================================================================
# 7. Metriche
# =============================================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE, bias medio, R^2 e correlazione di Pearson. NaN nelle previsioni: errore."""
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    if np.isnan(y_pred).any():
        raise ValueError("Previsioni con NaN: gestire i fallimenti prima di calcolare le metriche")
    e = y_pred - y_true
    return {"RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)), "bias": np.mean(e),
            "R2": 1 - np.sum(e ** 2) / np.sum((y_true - y_true.mean()) ** 2),
            "std previsioni": np.std(y_pred)}


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric: str = "RMSE",
                 n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Intervallo di confidenza bootstrap (percentili) di una metrica sul test set."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = regression_metrics(y_true[idx], y_pred[idx])[metric]
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def binned_errors(y_true: np.ndarray, y_pred: np.ndarray, bins: np.ndarray) -> dict:
    """Bias e RMSE in classi del parametro vero (errore in funzione del valore vero)."""
    idx = np.digitize(y_true, bins) - 1
    centers = (bins[:-1] + bins[1:]) / 2
    bias = np.array([np.mean(y_pred[idx == i] - y_true[idx == i]) for i in range(len(centers))])
    rmse = np.array([np.sqrt(np.mean((y_pred[idx == i] - y_true[idx == i]) ** 2)) for i in range(len(centers))])
    return {"centers": centers, "bias": bias, "rmse": rmse}


# =============================================================================
# 8. Feature fisiche per split (usate da 02, 03, 05) con cache su disco
# =============================================================================

def physics_features(X: np.ndarray) -> dict:
    """Tutte le quantità "fisiche" di una matrice di fGn (B, N):
    stime DFA e R/S di H, curve log F(s) e log R/S(n), feature dei grafi."""
    X = np.asarray(X, dtype=np.float64)
    scales = log_scales(X.shape[1])
    logF = np.log(dfa_fluctuation(X, scales))
    logRS = np.log(rs_statistic(X, scales))
    ls = np.log(scales)
    return {"dfa": _slope(ls, logF), "rs": _slope(ls, logRS),
            "logF": logF, "logRS": logRS, "scales": scales,
            "graph": graph_features(X)}


ABS_SCALES = (8, 16, 32)     # scale assolute comuni a tutte le lunghezze N >= 128 (s_max = N/4)
N_SEG = 4                    # segmenti per le pendenze locali
CLASSIC_INT_FEATURES = (["dfa", "rs"]
                        + [f"dfa_pend{j}" for j in range(N_SEG)] + [f"rs_pend{j}" for j in range(N_SEG)]
                        + [f"logF/sd_s{s}" for s in ABS_SCALES] + [f"logRS_s{s}" for s in ABS_SCALES])


def local_slopes(scales: np.ndarray, logY: np.ndarray, n_seg: int = N_SEG) -> np.ndarray:
    """Pendenze di logY contro log s su n_seg segmenti consecutivi (estremi condivisi) delle scale.
    Una pendenza è adimensionale e, per un processo autosimile, vale ~H a ogni scala: non dipende
    da N, a differenza dei valori log F(s) sulle scale relative (D-30)."""
    ls = np.log(scales)
    edges = np.linspace(0, len(ls) - 1, n_seg + 1).round().astype(int)
    return np.column_stack([_slope(ls[a:b + 1], logY[:, a:b + 1]) for a, b in zip(edges[:-1], edges[1:])])


def classic_intensive_features(f: dict, X: np.ndarray) -> np.ndarray:
    """Feature classiche indipendenti da N e dall'ampiezza (D-30), dalle curve già calcolate:
    stime globali DFA e R/S, pendenze locali su N_SEG segmenti, e i livelli a scale ASSOLUTE
    s in ABS_SCALES (interpolati in log-log): log(F(s)/sd) e log<R/S>(s). F si divide per la
    deviazione standard della serie, così la feature non usa l'ampiezza (cfr. D-19, D-26);
    R/S è già adimensionale. Ordine come CLASSIC_INT_FEATURES."""
    scales = np.asarray(f["scales"])
    if scales[0] > ABS_SCALES[0] or scales[-1] < ABS_SCALES[-1]:
        raise ValueError(f"Scale {scales[0]}..{scales[-1]} non coprono {ABS_SCALES}: serve N >= 128")
    ls, la = np.log(scales), np.log(np.asarray(ABS_SCALES, dtype=float))
    interp = lambda Y: np.stack([np.interp(la, ls, row) for row in Y])
    sd = np.asarray(X, dtype=np.float64).std(axis=1)
    return np.column_stack([f["dfa"], f["rs"], local_slopes(scales, f["logF"]), local_slopes(scales, f["logRS"]),
                            interp(f["logF"]) - np.log(sd)[:, None], interp(f["logRS"])])


def load_or_make_features(ds: dict, mode: str, seed: int, data_dir: str | Path = "data",
                          verbose: bool = True) -> dict:
    """Calcola (o carica dalla cache) physics_features per ogni split del dataset."""
    import time
    path = Path(data_dir) / f"features_{mode}_seed{seed}.npz"
    if path.exists():
        z = np.load(path)
        out = {}
        for key in z.files:
            split, field = key.split("__")
            out.setdefault(split, {})[field] = z[key]
        return out
    out = {}
    for split, d in ds.items():
        t0 = time.time()
        out[split] = physics_features(d["X"])
        if verbose:
            print(f"  feature {split:>11}: {len(d['X'])} serie in {time.time() - t0:.1f} s")
    np.savez_compressed(path, **{f"{s}__{f}": v for s, d in out.items() for f, v in d.items()})
    return out


# =============================================================================
# 9. Reti neurali (PyTorch): modelli, normalizzazione, training con early stopping
# =============================================================================

def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_deterministic(seed: int) -> None:
    """Seed ovunque + algoritmi deterministici dove possibile (su GPU alcune
    operazioni restano non bit-esatte: lo dichiariamo, D-11)."""
    import torch
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def normalize_series(X: np.ndarray, how: str, train_scale: float | None = None) -> np.ndarray:
    """Normalizzazione degli input della CNN (D-19, D-26).
    'per_serie': ogni serie a media 0 e deviazione standard 1 (toglie l'ampiezza).
    'globale'  : divisione per un'unica scala calcolata sul train (conserva l'ampiezza)."""
    X = np.asarray(X, dtype=np.float32)
    if how == "per_serie":
        return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    if how == "globale":
        return X / np.float32(train_scale)
    raise ValueError(how)


class FeatureScaler:
    """Standardizzazione delle feature con media/deviazione del train; NaN -> mediana del train."""

    def fit(self, F: np.ndarray):
        F = np.asarray(F, dtype=np.float64)
        self.median = np.nanmedian(F, axis=0)
        Ff = np.where(np.isfinite(F), F, self.median)
        self.mean, self.std = Ff.mean(axis=0), Ff.std(axis=0) + 1e-12
        return self

    def transform(self, F: np.ndarray) -> np.ndarray:
        F = np.asarray(F, dtype=np.float64)
        Ff = np.where(np.isfinite(F), F, self.median)
        return ((Ff - self.mean) / self.std).astype(np.float32)


def _torch_modules():
    import torch
    import torch.nn as nn

    class CNN1D(nn.Module):
        """CNN 1D: blocchi [Conv-BN-ReLU-Conv-BN-ReLU-MaxPool], pooling globale (media e max
        sul tempo) -> funziona con qualsiasi lunghezza N. `n_extra` > 0 aggiunge feature
        esterne concatenate prima della testa densa (fusione tardiva, modello ibrido D-12)."""

        def __init__(self, width=32, n_blocks=4, kernel=7, n_extra=0, dropout=0.1):
            super().__init__()
            layers, c_in = [], 1
            for b in range(n_blocks):
                c_out = width * min(2 ** b, 4)
                layers += [nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2), nn.BatchNorm1d(c_out), nn.ReLU(),
                           nn.Conv1d(c_out, c_out, kernel, padding=kernel // 2), nn.BatchNorm1d(c_out), nn.ReLU(),
                           nn.MaxPool1d(2)]
                c_in = c_out
            self.features = nn.Sequential(*layers)
            self.n_extra = n_extra
            self.head = nn.Sequential(nn.Linear(2 * c_in + n_extra, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

        def forward(self, x, extra=None):
            h = self.features(x.unsqueeze(1))
            z = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
            if self.n_extra:
                z = torch.cat([z, extra], dim=1)
            return self.head(z).squeeze(1)

    class MLP(nn.Module):
        """Percettrone multistrato su un vettore di feature."""

        def __init__(self, n_in, hidden=64, n_layers=2, dropout=0.1):
            super().__init__()
            layers, d = [], n_in
            for _ in range(n_layers):
                layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
                d = hidden
            layers.append(nn.Linear(d, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x, extra=None):
            return self.net(x).squeeze(1)

    return CNN1D, MLP


def build_model(kind: str, **kw):
    CNN1D, MLP = _torch_modules()
    return {"cnn": CNN1D, "mlp": MLP}[kind](**kw)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def train_regressor(model, X_tr, y_tr, X_val, y_val, *, E_tr=None, E_val=None, lr=1e-3,
                    batch_size=128, max_epochs=40, patience=6, weight_decay=1e-4, seed=0,
                    verbose=False):
    """Addestra un modello di regressione con perdita MSE sul target standardizzato.

    - Target standardizzato con media/deviazione del train (y_mu, y_sd salvati nel modello).
    - Adam + riduzione del learning rate a metà se l'RMSE di validazione non migliora per 3 epoche.
    - Early stopping sull'RMSE di validazione (pazienza `patience`), ripristino dei pesi migliori.
    Il validation set serve SOLO a questo (D-15). E_tr/E_val: feature extra (modello ibrido).
    Restituisce (modello, storia)."""
    import copy
    import time

    import torch

    make_deterministic(seed)
    dev = get_device()
    model = model.to(dev)
    y_mu, y_sd = float(np.mean(y_tr)), float(np.std(y_tr))
    model.y_mu, model.y_sd = y_mu, y_sd

    def T(a):
        return None if a is None else torch.as_tensor(np.asarray(a, dtype=np.float32))

    is_graph = isinstance(X_tr, GraphSet)
    Xt = X_tr if is_graph else T(X_tr)
    Yt, Et = T((np.asarray(y_tr) - y_mu) / y_sd), T(E_tr)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, wait, hist = np.inf, None, 0, []
    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        tot = 0.0
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            xb = Xt.batch(idx.numpy(), dev) if is_graph else Xt[idx].to(dev)
            yb = Yt[idx].to(dev)
            eb = Et[idx].to(dev) if Et is not None else None
            opt.zero_grad()
            loss = torch.mean((model(xb, eb) - yb) ** 2)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        p_val = predict(model, X_val, E_val)
        rmse_val = float(np.sqrt(np.mean((p_val - np.asarray(y_val)) ** 2)))
        rmse_tr = float(np.sqrt(tot / len(Xt))) * y_sd
        sched.step(rmse_val)
        hist.append({"epoch": epoch + 1, "rmse_train": rmse_tr, "rmse_val": rmse_val,
                     "lr": opt.param_groups[0]["lr"], "s": time.time() - t0})
        if verbose:
            print(f"  epoca {epoch + 1:3d}  train {rmse_tr:.4f}  val {rmse_val:.4f}  ({hist[-1]['s']:.1f} s)")
        if rmse_val < best - 1e-5:
            best, best_state, wait = rmse_val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist


def predict(model, X, E=None, batch_size=512) -> np.ndarray:
    """Previsioni nella scala originale del target."""
    import torch
    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            if isinstance(X, GraphSet):
                xb = X.batch(np.arange(i, min(i + batch_size, len(X))), dev)
            else:
                xb = torch.as_tensor(np.asarray(X[i:i + batch_size], dtype=np.float32)).to(dev)
            eb = None if E is None else torch.as_tensor(np.asarray(E[i:i + batch_size], dtype=np.float32)).to(dev)
            out.append(model(xb, eb).cpu().numpy())
    return np.concatenate(out) * model.y_sd + model.y_mu


# =============================================================================
# 10. Grafi per la GNN: insieme di grafi compatto, feature dei nodi, modello GIN
# =============================================================================

class GraphSet:
    """Insieme di grafi di visibilità in forma compatta (archi non orientati in uint16, una sola
    direzione), convertiti in tensori PyTorch solo al momento di formare il batch.
    Serve a stare nella RAM di Colab con 40 000 grafi da 1024 nodi (D-32, D-38).

    edges[i]: (E_i, 2) uint16;  x[i]: (N_i, F) float32 feature dei nodi (x può essere una
    lista o un oggetto NodeFeatures che le calcola al volo)."""

    def __init__(self, edges: list, x: list):
        self.edges, self.x = edges, x

    def __len__(self):
        return len(self.x)

    def n_edges(self) -> np.ndarray:
        return np.array([len(e) for e in self.edges])

    def batch(self, idx, dev):
        """Unisce i grafi `idx` in un unico grafo disconnesso (convenzione di PyG):
        restituisce (x, edge_index orientato nei due versi, vettore batch)."""
        import torch
        xs, es, bs, off = [], [], [], 0
        for j, i in enumerate(idx):
            xi, ei = self.x[i], self.edges[i].astype(np.int64) + off
            xs.append(xi); es.append(ei); bs.append(np.full(len(xi), j, dtype=np.int64))
            off += len(xi)
        e = np.concatenate(es)
        edge_index = np.concatenate([e, e[:, ::-1]]).T            # archi nei due versi
        return (torch.as_tensor(np.concatenate(xs)).to(dev),
                torch.as_tensor(np.ascontiguousarray(edge_index)).to(dev),
                torch.as_tensor(np.concatenate(bs)).to(dev))


def build_graphs(X_fgn: np.ndarray, kind: str, series: str) -> tuple[list, list, list]:
    """Costruisce i grafi di visibilità ('nvg'/'hvg') del fBm o del fGn di ogni riga di X_fgn.
    Restituisce (lista archi uint16, lista gradi uint16, lista valori della serie normalizzati per serie).
    Indici e gradi stanno in 16 bit perché N <= 65535: metà della memoria di int32 (D-38)."""
    from ts2vg import HorizontalVG, NaturalVG
    cls = NaturalVG if kind == "nvg" else HorizontalVG
    X_fgn = np.asarray(X_fgn)
    if X_fgn.shape[1] > np.iinfo(np.uint16).max:
        raise ValueError("Serie troppo lunghe per indici a 16 bit")
    edges, degs, vals = [], [], []
    for x in X_fgn:
        x = np.asarray(x, dtype=np.float64)
        s = fbm_from_fgn(x) if series == "fbm" else x
        g = cls(); g.build(s)
        edges.append(np.asarray(g.edges, dtype=np.uint16).reshape(-1, 2))
        degs.append(np.asarray(g.degrees, dtype=np.uint16))
        vals.append(((s - s.mean()) / s.std()).astype(np.float32))
    return edges, degs, vals


class NodeFeatures:
    """Feature dei nodi calcolate al volo da gradi e valori (si comporta come una lista).
    Evita di tenere in memoria una copia delle feature per ogni variante (D-38)."""

    def __init__(self, degs: list, vals: list, which: str, deg_mu: float, deg_sd: float):
        if which not in ("grado", "grado+valori"):
            raise ValueError(which)
        self.degs, self.vals, self.which, self.mu, self.sd = degs, vals, which, deg_mu, deg_sd

    def __len__(self):
        return len(self.degs)

    def __getitem__(self, i):
        f = ((np.log(self.degs[i].astype(np.float32)) - self.mu) / self.sd).astype(np.float32)
        if self.which == "grado":
            return f[:, None]
        return np.column_stack([f, self.vals[i]])


def node_features(degs: list, vals: list, which: str, deg_mu: float, deg_sd: float) -> NodeFeatures:
    """Feature dei nodi (D-33):
      'grado'        -> log(k), standardizzato con media/dev. del train;
      'grado+valori' -> in più il valore della serie nel nodo (normalizzato per serie).
    Con il solo grado la GNN vede solo la topologia del grafo, che è invariante per
    riscalamento (NVG e HVG) e per trend lineare (NVG): non può vedere l'ampiezza."""
    return NodeFeatures(degs, vals, which, deg_mu, deg_sd)


def _gin_module():
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool

    class GIN(nn.Module):
        """Graph Isomorphism Network (Xu et al. 2019) per regressione a livello di grafo.
        `n_layers` strati GINConv (aggregazione a somma + MLP), poi lettura per grafo con media e
        massimo dei nodi a OGNI strato (concatenati), testa densa 64 -> 1.
        `n_extra` > 0: feature esterne concatenate alla lettura (modello ibrido)."""

        def __init__(self, n_in, hidden=32, n_layers=3, dropout=0.1, n_extra=0, norm="layer"):
            super().__init__()
            # norm: 'batch' (BatchNorm sui nodi del batch) o 'layer' (LayerNorm per nodo, D-35)
            Norm = (lambda: nn.BatchNorm1d(hidden)) if norm == "batch" else (lambda: nn.LayerNorm(hidden))
            self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
            d = n_in
            for _ in range(n_layers):
                mlp = nn.Sequential(nn.Linear(d, hidden), Norm(), nn.ReLU(), nn.Linear(hidden, hidden))
                self.convs.append(GINConv(mlp, train_eps=True))
                self.bns.append(Norm())
                d = hidden
            self.n_extra = n_extra
            self.head = nn.Sequential(nn.Linear(2 * hidden * n_layers + n_extra, 64), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(64, 1))

        def forward(self, g, extra=None):
            x, edge_index, batch = g
            reads = []
            for conv, bn in zip(self.convs, self.bns):
                x = torch.relu(bn(conv(x, edge_index)))
                reads += [global_mean_pool(x, batch), global_max_pool(x, batch)]
            z = torch.cat(reads, dim=1)
            if self.n_extra:
                z = torch.cat([z, extra], dim=1)
            return self.head(z).squeeze(1)

    return GIN


def build_gnn(**kw):
    return _gin_module()(**kw)
