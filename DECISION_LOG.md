# Registro delle decisioni

Ogni scelta del progetto, con la motivazione e le alternative scartate.
Regola del corso: *provare approcci diversi, scegliere il migliore; se le prestazioni sono vicine, scegliere il più semplice.*

Formato: **D-numero — titolo** · data · *Scelta* / *Perché* / *Alternative scartate*.

---

## Organizzazione e infrastruttura

### D-01 — Framework: PyTorch + PyTorch Geometric per tutte le reti · 2026-11-06
- **Scelta:** CNN, MLP e GNN scritte tutte in PyTorch; GNN con PyTorch Geometric (PyG).
- **Perché:** un unico framework significa stesso ottimizzatore, stessa inizializzazione e stesso ciclo di training per tutti i modelli, quindi il confronto "a parità di sforzo di tuning" è difendibile. PyG è lo standard per le GNN.
- **Scartate:** Keras (usato nel corso) per CNN + PyG per GNN → due framework, confronto meno pulito; Keras + Spektral → libreria poco mantenuta e problemi con Keras 3.

### D-02 — Codice condiviso in `sp_utils.py`, notebook che clonano il repository · 2026-11-06
- **Scelta:** funzioni comuni (simulazione, stimatori, grafi, modelli, metriche) in un unico modulo; ogni notebook, su Colab, clona il repository pubblico e lo importa. I dati non sono nel repository: ogni notebook li rigenera dai seed se non li trova.
- **Perché:** nessun passo manuale, nessuna copia divergente del codice, storia delle modifiche tracciata da git.
- **Scartate:** Google Drive montato (richiede autorizzazione manuale a ogni sessione); notebook completamente autonomi con codice duplicato (rischio di versioni incoerenti).

### D-03 — Due modalità: `quick` (CPU locale) e `full` (GPU Colab) · 2026-11-06
- **Scelta:** `MODE = "quick"` → N = 512, 4k/1k/2k serie; `MODE = "full"` → N = 1024, 40k/5k/10k serie.
- **Perché:** il portatile di sviluppo non ha GPU; la modalità quick verifica che tutto giri dall'inizio alla fine, la full produce i numeri finali. N = 1024 ≈ 4 anni di dati giornalieri, scala rilevante per il capitolo sui dati reali.
- **Scartate:** N = 2048 con 100k serie → troppo pesante per la GNN su Colab gratuito.

### D-04 — Il repository contiene solo il codice del progetto · 2026-11-06
- **Scelta:** il repository è la sottocartella `stochastic-ml/`; il materiale del corso (PDF, notebook, dataset) resta fuori ed è anche escluso dal `.gitignore`.
- **Perché:** il repository è pubblico; il materiale del corso non è nostro da pubblicare ed è pesante.
- **Scartate:** repository nella cartella principale con `.gitignore` → un errore basterebbe a pubblicare il materiale.

### D-05 — Identità git con indirizzo `noreply` di GitHub · 2026-11-06
- **Scelta:** i commit usano `<id>+<utente>@users.noreply.github.com`.
- **Perché:** in un repository pubblico l'indirizzo email dei commit è visibile a chiunque.

## Fase 1 — fBm e stima di H

### D-10 — Intervallo dei parametri: H ~ Uniforme(0.05, 0.95) · 2026-11-06
- **Scelta:** prior uniforme su H, estremi esclusi.
- **Perché:** copre sia il regime anti-persistente/rugoso (H ≈ 0.1, rilevante per la volatilità rugosa) sia quello persistente; il prior uniforme evita che la rete impari un prior sbilanciato. Gli estremi 0 e 1 sono degeneri.

### D-11 — Incertezze: ensemble di 5 seed + bootstrap sulle metriche di test · 2026-11-06
- **Scelta:** ogni rete è addestrata con 5 seed diversi; la previsione è la media, la dispersione dà l'incertezza; le metriche di test hanno intervalli di confidenza bootstrap.
- **Perché:** è il metodo più semplice che dà sia la variabilità tra run sia un'incertezza per singola serie.
- **Scartate (per ora):** testa gaussiana con perdita NLL (possibile estensione per il capitolo sui dati reali); perdita quantile.

### D-12 — Modello ibrido: disegno 2×2 e fusione tardiva · 2026-11-06
- **Scelta:** quattro celle {serie grezza sì/no} × {feature fisiche sì/no}, più gli stimatori classici da soli e la loro versione ricalibrata. Le feature fisiche entrano concatenate nella testa densa della rete (fusione tardiva). Variante: predire il residuo H − H_DFA.
- **Perché:** separa il contributo della serie grezza da quello delle feature fisiche; la DFA ricalibrata impedisce di scambiare una semplice correzione del bias per un vantaggio dell'ibrido.
