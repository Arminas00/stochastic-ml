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

### D-30 — Limite noto: i modelli sulle curve DFA/R-S sono legati alla lunghezza N del train · 2026-11-06 · notebook 03 (corretto il 2026-11-07)
- **Risultato:** i modelli con le feature "classiche" o "tutte" hanno RMSE 0.13–0.29 appena N cambia; il modello sulle sole feature del grafo invece regge e migliora con N (0.023 a N = 2048). La CNN resta stabile.
- **Perché:** le scale di DFA/R-S sono scelte relative a N (fino a N/4): la stessa colonna ha un significato diverso a N diversi. Le feature del grafo sono quasi tutte intensive (esponente della coda, frazione di nodi con grado 2 o 3, dispersione dei gradi).
- **Correzione:** una prima versione di questa voce attribuiva il problema a tutte le feature, compreso il grafo; il notebook 04 ha mostrato che non è così.
- **Risolto il 2026-11-07 (scelta dell'utente: "Fix D-30 features"):** nuovo gruppo di feature classiche
  indipendenti da N, `sp_utils.classic_intensive_features` (16 numeri): stime DFA e R/S; pendenze locali di
  log F e log R/S su 4 tratti consecutivi delle scale; livelli log(F(s)/σ̂) e log⟨R/S⟩(s) a scale **assolute**
  s = 8, 16, 32 (interpolati in log-log; coperte da ogni N ≥ 128). Gruppi "classiche-int" e "tutte-int"
  (+ 16 feature del grafo = 32) nel notebook 03; ibrido "full-int" nel notebook 05. Le vecchie curve restano
  nei notebook come confronto (ablazione del difetto).
- **Perché così:** una pendenza è adimensionale e per un processo autosimile vale ~H a ogni scala; un livello a
  scala assoluta fissa ha lo stesso significato a ogni N. F si divide per la deviazione standard della serie
  per non usare l'ampiezza (artefatto delle simulazioni a varianza 1, D-19/D-26); R/S è già adimensionale.
- **Prova preliminare (GBT, quick, prima di toccare i notebook; test visto solo per scegliere il disegno, non i
  suoi iperparametri):** solo pendenze: 0.043 a N = 512 e 0.026 a N = 2048, robuste ma meno precise. Pendenze +
  livelli assoluti + grafo: 0.030 e 0.018. Con F non diviso per σ̂: un po' meglio (0.032 contro 0.036 sul solo
  gruppo dei livelli), ma è l'artefatto dell'ampiezza: scartato.
- **Scartate:** scale assolute oltre 32 (non disponibili per N = 128 senza valori mancanti); togliere dal grafo
  le feature estensive (log k_max, ⟨k⟩), che in prova peggiorano (0.034 contro 0.033) senza migliorare la robustezza.

### D-31 — GNN: Graph Isomorphism Network con lettura a ogni strato · 2026-11-06 · notebook 04
- **Scelta:** strati GINConv (aggregazione a somma + MLP, ε addestrabile), BatchNorm, ReLU; lettura per grafo con media e massimo dei nodi dopo ogni strato, concatenate; testa densa 64→1.
- **Perché:** l'informazione su H nei grafi di visibilità sta nella distribuzione dei gradi e nelle loro correlazioni; l'aggregazione a somma permette di contarli (GIN è espressiva quanto il test di Weisfeiler–Lehman), mentre la media normalizzata della GCN li cancella in parte. Leggere tutti gli strati combina informazione locale e di vicinato più ampio. Il pooling globale rende la rete indipendente dal numero di nodi, come la CNN.
- **Scartate:** GCN (aggregazione normalizzata, meno espressiva sui gradi); GAT (attenzione: più parametri e più lenta, nessun vantaggio atteso su grafi senza feature ricche); pooling gerarchico (DiffPool, ecc.: costoso su grafi da 1000 nodi).

### D-32 — Grafi in forma compatta, convertiti a tensori solo per il batch · 2026-11-06 · notebook 04
- **Scelta:** archi in int32 e in una sola direzione; il batch PyG (grafo disconnesso + vettore `batch`) si forma al volo.
- **Perché:** la NVG di 40 000 fBm da 1024 nodi ha milioni di archi; in int64 e nei due versi (formato standard di PyG) occuperebbe 4× la memoria, oltre la RAM di Colab gratuito.

### D-33 — Feature dei nodi: log del grado, con e senza valori della serie · 2026-11-06 · notebook 04
- **Scelta:** variante principale con il solo log(k) standardizzato sul train; ablazione con in più il valore della serie nel nodo, normalizzato per serie.
- **Perché:** con il solo grado la rete vede solo la topologia, quindi eredita le invarianze del grafo (verificate sulle previsioni nel notebook 04): è il test diretto di "cosa conserva e cosa perde la rappresentazione". I valori normalizzati per serie aggiungono la forma locale della serie senza reintrodurre l'ampiezza (coerente con D-26).
- **Due grafi:** NVG del fBm (la scelta di Lacasa et al.) e HVG del fGn (il grafo il cui esponente λ era più informativo nel notebook 02).

### D-34 — GNN: iperparametri scelti sulla variante principale e riusati per le altre · 2026-11-06 · notebook 04
- **Scelta:** due configurazioni provate su NVG(fBm) con il solo grado; la migliore sul validation è usata per tutte e quattro le varianti.
- **Perché:** la GNN è il modello più costoso (≈ 20 s per epoca su CPU in quick); ripetere la ricerca per ogni variante quadruplicherebbe il costo. Lo sforzo di tuning resta paragonabile a quello della CNN (due configurazioni). Le varianti senza ricerca propria sono, se mai, svantaggiate: un loro risultato positivo è conservativo.

### D-36 — Modello ibrido: due insiemi di feature, variante a residuo, GNN ibrida · 2026-11-07 · notebook 05
- **Scelta (concordata):** ibrido-min (le 4 stime fisiche del piano: H_DFA, H_R/S, γ della NVG, λ della HVG) e ibrido-full (tutte le 42 feature); variante a residuo (target H − H_DFA); GNN ibrida solo per la variante migliore sul validation. Fusione tardiva: feature concatenate prima della testa densa. Architettura, iperparametri, seed e addestramento identici alla rete senza feature, quindi nessuna nuova ricerca di iperparametri.
- **Perché:** l'unica differenza tra rete e ibrido è l'input aggiuntivo, quindi la differenza di prestazioni misura il contributo della fisica. Min vs full dice se alla rete servono le stime o le curve complete. Il residuo verifica l'idea "impara una correzione". Un test di permutazione misura quanto l'ibrido usa davvero ciascuna feature.
- **Scartate:** fusione precoce (feature come canali aggiuntivi della serie: le feature sono scalari per serie, non segnali nel tempo); ibrido per tutte e quattro le GNN (≈ 4× il costo).

### D-35 — GNN con LayerNorm invece di BatchNorm · 2026-11-07 · notebook 04
- **Problema osservato:** con BatchNorm l'RMSE di validazione della GNN oscillava tra 0.09 e 0.27 da un'epoca all'altra mentre quello di train era stabile (curve nel primo run del notebook 04). Le statistiche di BatchNorm sono calcolate su tutti i nodi del batch, dominate dai grafi con più nodi e dagli hub di grado alto: in valutazione (statistiche medie) la rete vede una distribuzione diversa da quella di training. Con un validation così rumoroso l'early stopping sceglie un'epoca a caso.
- **Scelta:** LayerNorm, che normalizza ogni nodo separatamente e si comporta allo stesso modo in training e valutazione.
- **Verifica:** HVG(fGn), solo grado, 10 epoche: validation con BatchNorm 0.38→0.19 con salti fino a ×2; con LayerNorm decrescente in modo regolare fino a 0.083. Il livello finale è simile: la correzione stabilizza l'addestramento, non "regala" prestazioni.
- **Nota di onestà:** la correzione è stata fatta dopo il primo run completo del notebook 04, i cui numeri di test erano quindi già stati visti (GNN migliore: RMSE 0.041). È motivata da un difetto di addestramento evidente nelle curve di validation e il confronto BatchNorm/LayerNorm è stato fatto solo sul validation; non è una ricerca di iperparametri guidata dal test. I numeri del primo run sono riportati qui per trasparenza: NVG(fBm) grado 0.063, NVG(fBm) grado+valori 0.056, HVG(fGn) grado 0.118, HVG(fGn) grado+valori 0.041.
