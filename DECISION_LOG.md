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

### D-06 — I notebook si generano da sorgenti Python in `tools/` · 2026-11-06
- **Scelta:** ogni notebook `XX_*.ipynb` è scritto da `tools/build_XX.py` (celle Markdown e codice in un file leggibile), poi eseguito per intero con `jupyter nbconvert --execute`. Il notebook eseguito, con gli output, è nel repository.
- **Perché:** le modifiche ai notebook diventano leggibili in git (un `.ipynb` è JSON con output incorporati); l'esecuzione completa da zero garantisce "nessun errore, nessun passo manuale".
- **Scartate:** modificare i `.ipynb` a mano → diff illeggibili, rischio di celle eseguite fuori ordine.

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

### D-13 — Simulazione del fGn con il metodo di Davies–Harte · 2026-11-06 · notebook 01
- **Scelta:** embedding circolante della covarianza di Toeplitz e FFT; fBm come somma cumulata del fGn.
- **Perché:** è **esatto** (la covarianza simulata coincide con quella teorica, verificato nel notebook 01), costa O(N log N) e si vettorizza su migliaia di serie (10 000 serie di 1024 punti in ~3 s su CPU). Gli autovalori dell'embedding risultano ≥ 0 per H ∈ [0.01, 0.99] e N ≤ 4096 (verificato).
- **Scartate:** decomposizione di Cholesky (esatta ma O(N³) e una matrice per ogni H); metodo di Hosking (esatto ma O(N²)); metodi spettrali approssimati o per ondine (introducono errori sistematici proprio sulla grandezza che vogliamo stimare).

### D-14 — Si salva il fGn; ogni approccio riceve la rappresentazione naturale · 2026-11-06 · notebook 01
- **Scelta:** il dataset contiene il fGn $dW_H$; il fBm $W_H$ si ottiene con la somma cumulata (con $W_H(0)=0$).
- **Perché:** fGn e fBm contengono la stessa informazione (corrispondenza biunivoca), quindi dare a un approccio l'uno e a un altro l'altro non rende il confronto ingiusto. La scelta concreta (CNN sugli incrementi, grafi sul fBm come in Lacasa et al., DFA sul profilo) si motiva nei notebook successivi.

### D-15 — Split indipendenti generati con seed diversi · 2026-11-06 · notebook 01
- **Scelta:** train, validation e test generati separatamente con seed `SEED`, `SEED+1`, `SEED+2`; nessuna serie in comune (verificato con hash).
- **Perché:** separazione stretta per costruzione, non per partizione a posteriori; ogni split è un campione i.i.d. dello stesso prior su H (verificato con test di Kolmogorov–Smirnov).

### D-16 — Set di robustezza a lunghezze diverse, generati subito · 2026-11-06 · notebook 01
- **Scelta:** set di solo test con N ∈ {128, 256, 512, 2048, 4096} (quick: 128, 256, 1024, 2048), seed `SEED+100+N`.
- **Perché:** generarli insieme al dataset principale garantisce che nessuna scelta di modello sia stata fatta guardandoli.

### D-17 — Verifiche con soglie esplicite e arresto del notebook · 2026-11-06 · notebook 01
- **Scelta:** ogni confronto con un risultato esatto usa lo scarto normalizzato z = (stima − teoria)/SE, con SE dalla dispersione tra serie indipendenti, e soglia |z| < 4 (|z| < 5 dove i confronti sono ~1000). Il notebook termina con un `assert`: se una verifica fallisce, non si procede.
- **Perché:** una soglia su z è indipendente dalla scala della grandezza e ha un significato statistico (con ~100 confronti, |z| > 4 ha probabilità ~0.6% se la simulazione è corretta).

### D-18 — Correzione dell'ipotesi sulle invarianze dei grafi · 2026-11-06 · notebook 01
- **Risultato:** la NVG è invariante per x → a x + b t + c (a > 0), quindi anche per un trend lineare; la HVG è invariante per qualunque trasformazione monotona crescente, ma **non** per il trend (0% dei grafi identici, Jaccard 0.82). L'ipotesi iniziale "il grafo di visibilità non cambia con un trend lineare" vale quindi solo per la NVG.
- **Conseguenza:** previsioni da verificare più avanti: nessun grafo vede l'ampiezza (σ di OU, livello di volatilità); la NVG non vede la deriva di un moto browniano, la HVG sì.

### D-19 — Attenzione alla normalizzazione per serie · 2026-11-06 · notebook 01
- **Osservazione:** la varianza campionaria di una serie di fGn vale in media 1 − N^(2H−2) (verificato): per H = 0.9 e N = 512 è ~0.69. Dividere ogni serie per la propria deviazione standard quindi non è neutro rispetto ad H.
- **Da decidere nel notebook 03:** normalizzazione per serie (rimuove l'ampiezza, che per il fBm non è informativa) contro normalizzazione globale con statistiche del train.

### D-20 — Scale di DFA e R/S: 12 scale log-equispaziate tra 8 e N/4 · 2026-11-06 · notebook 02
- **Scelta:** stesse scale per DFA-1 e R/S; finestre non sovrapposte, prese da sinistra e da destra (DFA).
- **Perché:** sotto ~8 punti la regressione locale e la deviazione standard per finestra sono troppo rumorose; sopra N/4 ci sono meno di 4 finestre per scala. Le stesse scale per i due metodi rendono il confronto pulito.
- **Scartate:** scale fino a N/2 (troppo poche finestre); DFA di ordine 2 (inutile per fBm senza trend, più varianza).

### D-21 — Esponente della NVG con lo stimatore di Clauset–Shalizi–Newman · 2026-11-06 · notebook 02
- **Scelta:** massima verosimiglianza per una coda a legge di potenza, k_min scelto minimizzando la distanza di Kolmogorov–Smirnov, coda di almeno 50 punti. Per l'HVG, massima verosimiglianza della coda esponenziale (geometrica) per k ≥ 4. Se l'esponente non è stimabile, la previsione analitica usa la media del train (conteggio riportato).
- **Perché:** è lo stimatore standard e non distorto; la regressione lineare sull'istogramma log-log è distorta e dipende dal binning.
- **Risultato:** la relazione di Lacasa γ = 3 − 2H non è riprodotta né a N = 512 né a N = 16384 con questo stimatore (vedi notebook 02). Approccio 2a peggiore della baseline: risultato negativo, spiegato.

### D-22 — Ricalibrazione lineare sul train come baseline "fisica + 2 parametri" · 2026-11-06 · notebook 02
- **Scelta:** per ogni stimatore Ĥ, retta H = a + b Ĥ fittata sul solo train.
- **Perché:** separa due cose diverse: l'informazione contenuta nella stima (misurata dopo la ricalibrazione) e il suo bias (corretto dalla retta). Un modello ibrido deve battere la versione ricalibrata, non solo quella grezza.
- **Scartate:** ricalibrazione non lineare (isotonica, polinomiale) → più parametri; la scelta lineare è la più semplice e, per DFA e R/S, sufficiente (residui senza struttura nel grafico bias-vs-H).

### D-23 — Le previsioni non vengono tagliate in [0, 1] · 2026-11-06 · notebook 02
- **Scelta:** le stime grezze possono uscire dall'intervallo fisico e restano così nelle metriche.
- **Perché:** tagliarle migliorerebbe artificialmente metodi che sbagliano molto (es. Lacasa) e nasconderebbe il problema nei grafici.

### D-24 — DFA grezza come riferimento classico · 2026-11-06 · notebook 02
- **Scelta:** il riferimento "solo fisica" per le reti è la DFA grezza.
- **Perché:** DFA grezza, DFA ricalibrata e R/S ricalibrato hanno lo stesso RMSE entro l'intervallo di confidenza (0.053–0.054 in quick); per la regola del corso si sceglie il più semplice, che non impara alcun parametro.

### D-25 — Architettura della CNN 1D · 2026-11-06 · notebook 03
- **Scelta:** 4 blocchi [Conv1d(k=7)–BN–ReLU ×2, MaxPool(2)], canali che raddoppiano fino a 4× la larghezza base, pooling globale (media + massimo sul tempo), testa densa 64→1. Input: incrementi (fGn).
- **Perché:** il campo recettivo finale (~200 punti) copre le scale su cui DFA e R/S misurano la legge di potenza; il pooling globale rende la rete indipendente da N (necessario per la prova di robustezza in lunghezza) e invariante per traslazione temporale, come deve essere per un processo a incrementi stazionari. Gli incrementi sono stazionari, il fBm no: la rete non deve imparare a gestire una deriva della scala lungo la serie.
- **Scartate:** testa densa sull'intera serie appiattita (dipende da N, molti più parametri); reti ricorrenti LSTM/GRU (più lente da addestrare, nessun vantaggio atteso per una statistica di scala); input fBm (non stazionario).

### D-26 — Normalizzazione per serie degli input della CNN · 2026-11-06 · notebook 03
- **Scelta:** ogni serie a media 0 e deviazione standard 1, anche se la normalizzazione globale dovesse dare un RMSE leggermente migliore.
- **Perché:** nelle simulazioni la varianza del fGn è sempre 1, quindi con la normalizzazione globale la varianza campionaria (1 − N^(2H−2), notebook 01) diventa un indizio di H che esiste solo nella simulazione: su dati reali l'ampiezza è arbitraria. Un modello che lo usa non si trasferisce. H è un esponente invariante di scala: l'input giusto è invariante di scala.
- **Verifica:** entrambe le normalizzazioni sono addestrate e confrontate nel notebook 03, per misurare quanto vale l'indizio.

### D-27 — Stesso sforzo di tuning per tutti i modelli · 2026-11-06 · notebook 03
- **Scelta:** per CNN, MLP e gradient boosting, due configurazioni di iperparametri ciascuno, scelta sul validation con un seed fisso, poi ensemble di 5 seed (3 in quick) sulla configurazione scelta. Il gradient boosting è deterministico: un solo modello.
- **Perché:** la regola del corso confronta approcci; un modello più ottimizzato degli altri vincerebbe per lo sforzo, non per la rappresentazione.
- **Scartate:** ricerca estesa degli iperparametri (costo, e rischio di sovradattare il validation).

### D-28 — Modelli sulle feature: MLP e gradient boosting, tre gruppi di feature · 2026-11-06 · notebook 03
- **Scelta:** feature classiche (DFA, R/S e curve su 12 scale), feature del grafo (16), tutte (42); MLP e `HistGradientBoostingRegressor`.
- **Perché:** il gruppo "grafo" misura quanto della informazione su H sopravvive nei grafi quando si usa più di un esponente (risposta al fallimento di Lacasa, notebook 02); il gruppo "classiche" è la cella "solo fisica + apprendimento" del disegno 2×2. Il gradient boosting è il modello su feature tabellari visto nel corso, robusto e quasi senza tuning.

### D-29 — Punto aperto: l'incertezza dell'ensemble non è calibrata · 2026-11-06 · notebook 03
- **Risultato:** copertura di ±2σ tra 25% e 60% (attesa 95%), correlazione tra σ dell'ensemble ed errore ≈ 0. L'ensemble misura solo l'incertezza del modello, non quella della traiettoria.
- **Da decidere:** se aggiungere una testa che predice anche la varianza (perdita gaussiana NLL) per avere barre d'errore per singola serie, utile nel capitolo sui dati reali.

### D-30 — Limite noto: i modelli sulle feature sono legati alla lunghezza N del train · 2026-11-06 · notebook 03
- **Risultato:** RMSE 0.13–0.29 appena N cambia, contro una CNN che resta stabile.
- **Perché:** le scale di DFA/R-S sono scelte relative a N (fino a N/4) e le feature del grafo (grado medio, grado massimo) crescono con N: lo stesso numero di colonna ha un significato diverso a N diversi.
- **Alternativa possibile (non fatta):** scale assolute fisse comuni a tutte le lunghezze e feature del grafo normalizzate per N. Per ora si dichiara il limite.
