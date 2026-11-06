# Stima di parametri e riconoscimento di processi stocastici

Progetto d'esame per il corso *AI for Experimental and Applied Physics*.

**Domanda centrale.** Quanto bene si stimano i parametri di un processo stocastico, e si riconosce
quale processo ha generato una serie temporale, usando:

0. una baseline banale (media / classe più frequente);
1. stimatori statistici classici (DFA, R/S, massima verosimiglianza per OU);
2. il grafo di visibilità della serie (relazione analitica esponente–H, senza apprendimento);
3. una CNN 1D sulla serie grezza (con MLP / alberi su feature riassuntive come baseline neurale);
4. una GNN sul grafo di visibilità;
5. un modello ibrido che riceve anche le stime fisiche e impara a correggerle.

## Struttura

| Notebook | Contenuto |
|---|---|
| `01_simulazione_validazione.ipynb` | Simulazione fGn/fBm, verifica contro risultati esatti, costruzione degli split |
| `02_classici_e_grafi.ipynb` | Approcci 0, 1, 2 |
| `03_cnn_e_feature.ipynb` | CNN 1D, MLP e alberi su feature |
| `04_gnn.ipynb` | GNN sui grafi di visibilità |
| `05_ibrido_ablazione.ipynb` | Disegno 2×2 serie grezza × feature fisiche |
| `06_risultati.ipynb` | Tabella comparativa e figure |

`sp_utils.py` contiene il codice condiviso. Ogni notebook clona questo repository se gira su Colab,
e rigenera i dati dai seed se non li trova: nessun passo manuale.

Le scelte e le loro motivazioni sono in [`DECISION_LOG.md`](DECISION_LOG.md).

## Esecuzione

- **Colab:** aprire il notebook da GitHub, scegliere un runtime GPU, *Esegui tutto*.
- **Locale (Windows/Linux):** `pip install -r requirements.txt`, poi eseguire il notebook.
  Con `MODE = "quick"` nella cella di configurazione tutto gira su CPU in pochi minuti.
