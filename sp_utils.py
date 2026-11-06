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
