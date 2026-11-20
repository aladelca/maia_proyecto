# Sleep staging beyond a single cohort

**Research and implementation plan · Universidad de los Andes · 18 November 2026**

Use both cohorts of Sleep-EDF Expanded to complete the activity. For the paper, test whether learning from expert disagreement improves N1 classification and helps identify epochs worth reviewing. A temporal convolutional network (TCN) and LightGBM already provide a strong comparison for both questions.

## Why we have used one database

Section 4 of the activity names **Sleep-EDF Expanded (Sleep-EDFx)** and explicitly includes **Sleep Cassette** and **Sleep Telemetry**. It cites direct public access as an advantage. Section 5 asks for comparison with studies using the same database. Section 6 lists the National Sleep Research Resource as further reading. These provisions anchor the course benchmark in Sleep-EDFx and leave room for separately reported research extensions. See *Tema 2. Propuesta_Proyecto IBIO (1)*, pp. 2-4. [PhysioNet dataset](https://physionet.org/content/sleep-edfx/1.0.0/)

The October implementation plan explicitly limited the original training campaign to real Sleep-EDFx. The team then froze subject folds in Cassette to compare methods on identical nights and deferred Telemetry for a separate analysis. This choice preserves a test of transfer between acquisition settings. See [original implementation plan](20261020-1943-infraestructura-local-sleep-edfx-mlflow.md) and [experiment protocol](../docs/experiment-protocol.md).

| Data already available | Subjects | Nights | Use so far |
| --- | --- | --- | --- |
| Cassette development | 62 | 122 | Five subject-grouped folds |
| Cassette reserved test | 16 | 31 | Reserved for final evaluation |
| Telemetry | 22 | 44 | Downloaded. Paired placebo and temazepam nights |
| Total in the local inventory | 100 | 197 | Each PSG has a matching hypnogram |

## What the results justify

The equal-weight waveform CNN-TCN and three-channel LightGBM reached **0.8009 mean fold macro-F1** (SD 0.0183), with **0.5401 pooled N1 F1**. Of 7,377 missed N1 epochs, 3,838 were called N2 and 2,165 wake. Accuracy was 61.7% near stage changes and 89.1% in stable intervals. These are Cassette development results. [Verified campaign](../output/evidencia/resumen/tcn/leaderboard-all.csv) · [Confusion counts](../output/evidencia/resumen/tcn/wave_tcn_lightgbm_50_50/metrics.json)

**Order of work.** Complete the Sleep-EDFx requirements and freeze a deployable evaluation window. Then run the N1 study with ISRUC and independent Dreem evaluation. The montage, event-supervision and pretraining studies answer separate questions and can be added as resources allow.

---

## Complete the activity on Sleep-EDFx

The proposal distinguishes the five sleep stages from an additional output for movement. It also describes a platform where the user chooses which of the available signals to supply. The three-channel research reference uses two EEGs and EOG. The serving interface accepts a prepared single-night Parquet with a fixed feature schema and five outputs. See activity §5, pp. 2-3. [Data contract](../docs/data-contract.md) · [Serving contract](../src/sleep_staging/serving/contract.py)

| Requirement | Planned experiment and acceptance evidence |
| --- | --- |
| 30-second epochs and movement | Keep W, N1, N2, N3 and REM for the five-stage literature benchmark. Train a second classifier with M as a sixth output. Merge original stages 3 and 4 into N3. Exclude unknown `?` from supervision and report exclusions. Publish separate five- and six-class matrices with denominators. |
| Flexible signal inputs | Cross the three EEG choices (Fpz-Cz, Pz-Oz, both) with four accessories (neither, EOG, chin EMG, both). Score all **12 profiles** on identical Cassette nights. Verify each profile through the upload-to-hypnogram path. A recording requires at least one EEG. |
| Traditional and deep methods | Compare feature-based LightGBM with a waveform CNN-TCN on the same subjects, signals and window. Report each branch and the fixed 50/50 probability blend. Refit a channel-matched tree or use an identified single branch when a profile requires it. |
| Reproducibility and comparison | Group both nights of each person together. Report overall accuracy, macro-F1 and stage confusion matrices. Compare published Sleep-EDF results by release/subset, montage, window, class policy and split unit. Include a matched single-EEG run for comparison with DeepSleepNet. [Original paper](https://arxiv.org/abs/1703.04046) |

**Movement is rare.** The current cropped development cache contains 115 M epochs among 152,167 rows, from 28 people and 37 nights. The five folds contain 35, 14, 21, 5 and 40 M epochs. Compare a six-class softmax with a training-only class-weighted variant. Report M precision, recall, F1, false positives per night, and support by subject and fold. Recount support after changing the window. [Movement counts by recording](../output/evidencia/resumen/ibio/tables/movement-by-night.csv)

**Why match the EMG comparison?** Blending a historical four-channel tree with the waveform model gave 0.8018 mean fold macro-F1. The paired 95% interval for its pooled macro-F1 difference from the three-channel reference spanned -0.0011 to +0.0027. The tree configuration also differed. Refitting with the same settings isolates the contribution of EMG. [Channel comparison](../notebook/09_transiciones_y_revision.ipynb)

**First transfer experiment.** Freeze the three-channel model on Cassette development, then evaluate the 44 Telemetry nights once. Report the two treatment conditions separately and resample each person's two nights together. For EMG experiments, Cassette provides a 1 Hz processed envelope and Telemetry provides a 100 Hz trace. Define and test a common envelope representation or evaluate distinct EMG encoders. [Acquisition details](https://physionet.org/content/sleep-edfx/1.0.0/)

---

## Which additional data answer which question?

Keep a Sleep-EDFx-only baseline for the activity. In the paper, give each additional source a specific role before opening outcome labels. Report evaluation cohorts separately so that larger datasets cannot conceal failures in smaller ones.

| Source and access | Role in the paper | What to check at intake |
| --- | --- | --- |
| **ISRUC-Sleep subgroup I** · Public release, 100 people with sleep disorders, one night each | Additional training and grouped inner validation for learning from two independent scorers | Preserve both hypnograms, align epoch start times and record which rater is missing. Restrict the primary factorial comparison to a common set of two-rated ISRUC epochs. [University of Coimbra](https://sleeptight.isr.uc.pt/) |
| **Dreem DOD-H and DOD-O** · Public, 25 healthy participants and 55 with obstructive sleep apnea in the scored benchmark | Independent evaluation of N1, expert agreement and review priority. Report results for each cohort. | Join the five individual scorers by recording and epoch. Use the published consensus. One additional DOD-O signal file lacks consensus. Exclude it from scored evaluation and log it. [Data and release note](https://github.com/Dreem-Organization/dreem-learning-open) · [Individual scorers](https://github.com/Dreem-Organization/dreem-learning-evaluation) |
| **HMC v1.1** · Public, 151 clinical PSGs | External test of robustness to different EEG derivations and missing sensors | Four EEG and two EOG leads at 256 Hz. Preserve electrodes, references and polarity. The two EOGs share a reference and can form horizontal E1-E2. [PhysioNet HMC](https://physionet.org/content/hmc-sleep-staging/1.1/) |
| **MASS cohort 1, SS2** · 19 people, biosignals require access approval | Auxiliary training on expert-marked spindles and K-complexes. Reserve participant-disjoint MASS folds for event evaluation. | Event annotations are downloadable separately. Use the actual scored channels and intervals. SS2 stage labels use 20-second R&K pages. [MASS access](https://ceams-carsm.ca/MASS/) · [SS2 annotations](https://doi.org/10.5683/SP3/Y889CS) |
| **Sleep Heart Health Study (SHHS)** · Controlled NSRR access | Unlabelled PSG pretraining for label-efficiency experiments | The repository lists 5,793 visit-1 and 2,651 visit-2 PSGs. Link visits to the same person and audit overlap with any training checkpoints. [PSG inventory](https://sleepdata.org/datasets/shhs/pages/05-polysomnography-introduction.md) · [NSRR access](https://sleepdata.org/showcase/where-to-start) |

**Start with data already on disk.** Telemetry supplies the first shift in acquisition setting. ISRUC supplies disagreement during learning, and Dreem supplies an independent way to evaluate it. HMC can be added for the montage hypothesis. MASS and SHHS can proceed independently when waveform access and compute are available.

---

## Main paper experiment · Learning from disagreement in N1

**Hypothesis.** Expert vote fractions can teach a model when an epoch is ambiguous. A contrastive loss can teach it to distinguish N1 from wake and N2 when raters agree on the stage. Test whether combining these signals improves N1 F1 on a new cohort and preserves confidence calibration. Soft targets and contrastive N1 learning both have published precedents. The contribution under test is their interaction and its transfer to an independently scored population. [Multi-rater targets](https://pmc.ncbi.nlm.nih.gov/articles/PMC10171642/) · [SSC-SleepNet](https://pubmed.ncbi.nlm.nih.gov/40408218/)

Train all four arms on the same Cassette-development and ISRUC training subjects, available inputs, batches and update budget. Use a common montage-aware input adapter, the same 15-epoch TCN, seeds and inner selection budget. Compare the hard-label pooled arm with an additional Cassette-only control to estimate the effect of adding the cohort.

| Arm | Stage target | Contrastive training |
| --- | --- | --- |
| A · Reference | Fixed primary ISRUC rater. Cassette uses its available scorer. | Off |
| B · Contrastive only | Same hard target as A | On |
| C · Votes only | Empirical distribution of the two ISRUC raters. Cassette target stays one-hot. | Off |
| D · Combined | Same target as C | On, with the same pair-selection rule as B |

For B and D, draw pairs from different training subjects. Use agreed ISRUC labels to form positive and negative pairs, emphasizing N1 versus W and N2. Use identical agreement masks in B and D. Keep disputed epochs in the stage loss. Fix rater 1 in A and B before fitting and repeat with rater 2 as a sensitivity check. A two-rater split gives a 50/50 target. It has no unique majority. Report each neural arm alone and with the same fold-trained, channel-matched LightGBM at 50/50.

**Primary contrast.** Compare the neural-only D and A arms on pooled N1 F1 within DOD-O against published consensus. Report all four arms, their paired differences and the factorial interaction on the same subjects. Treat the fixed blends as secondary. DOD-H provides replication. Stratify by at least four of five scorers agreeing and by lower agreement. Report W-to-N1 and N2-to-N1 errors so that overcalling N1 is visible.

**Review at a fixed budget.** In Cassette, the highest-entropy 10% per night captured 7,190 of 23,750 model/scorer discrepancies (30.3%). On Dreem compare entropy, disagreement between model branches and random selection at the same per-night quota. Score capture and precision separately for model/consensus errors and human disagreement (at most three of five agree). A blinded reader study with timed review and a random control would measure actual corrections and time saved. [Local review study](../notebook/09_transiciones_y_revision.ipynb)

---

## Three further experiments and one low-cost blend

### Flexible montages · Cassette, ISRUC and HMC

Share the EEG encoder across derivations, identify both electrodes and the reference, and pool only available channels. Give EOG and the EMG envelope their own representations. On the same development signals compare a fixed-slot CNN with a channel-set encoder, each with and without channel dropout. Keep at least one EEG during dropout. Compare all 12 Cassette profiles and report the worst profile as well as the full profile. For HMC, freeze a physically defined input map for each arm, including a common single-EEG comparison and an EEG-plus-EOG comparison. A central-to-mastoid lead is a distinct input from Fpz-Cz. [AnySleep](https://pmc.ncbi.nlm.nih.gov/articles/PMC13722680/) · [HMC montage](https://physionet.org/content/hmc-sleep-staging/1.1/)

### Event-supervised N2 boundaries · MASS SS2

Add a time-local auxiliary head for sleep spindles and K-complexes. Compare stage loss alone, stage plus spindle, stage plus K-complex, and both events. All arms see the same waveform batches and have the same stage objective and update budget. Use event labels only within annotated channels and intervals. For matched stage supervision use native 20-second SS2 pages with epoch-length-aware pooling. Learn event timing in seconds and keep the target Sleep-EDFx task at 30 seconds. Evaluate event F1 on held-out MASS people with a prespecified event-matching tolerance and the change in N1/N2 confusion on Cassette. [SS2 biosignals](https://doi.org/10.5683/SP3/K26LXJ) · [Expert events](https://doi.org/10.5683/SP3/Y889CS)

### Masked signal pretraining · SHHS

Pretrain an encoder to reconstruct masked waveform or time-frequency patches, then fine-tune for stages. Compare random initialization, in-domain masked pretraining on the outer-training Cassette subjects, and SHHS pretraining with the same backbone. Use the same nested 10%, 25%, 50% and 100% subsets of labelled training people in every arm. Match downstream update budgets, document pretraining hours, and keep both visits together. Plot macro-F1 and N1 F1 versus labelled subjects. Pilot on MPS and scale on CUDA after measuring memory and throughput. [Masked sleep-EEG pretraining](https://pubmed.ncbi.nlm.nih.gov/38293234/) · [SHHS](https://sleepdata.org/datasets/shhs/pages/05-polysomnography-introduction.md)

### Spectrogram fusion · Existing Cassette data

Train a 2D CNN on log-spectrograms with the same channels and 15-epoch TCN as the waveform branch. Compare waveforms, spectrograms, the existing two-way blend, and a fixed blend of **50% LightGBM, 25% waveform CNN-TCN, 25% spectrogram CNN-TCN**. Average aligned class probabilities. Report each branch and the paired change over the two-way blend. This is the fastest architecture experiment that needs no new recordings. [XSleepNet](https://arxiv.org/abs/2007.05492)

---

## How the results will be judged

### Evaluation protocol

1. **Freeze the input window.** Keep the existing benchmark with its annotation-derived 30-minute wake margin for historical comparison. For deployment and external tests, use all complete acquired epochs as the primary window. Where independently logged lights-off/on times exist, include that interval as a sensitivity analysis. Predict before reading reference labels. Report coverage, wake fraction and scored denominators for each window. Whole-night normalization and centered context support retrospective scoring.
2. **Keep participants independent.** Preserve the five outer Cassette development folds and keep every night from one person together. Assign external training subjects to grouped folds for internal selection. Fit normalization parameters requiring population data, losses, models, calibration and any blend selection inside training data. Reset context at each night or gap. Keep evaluation cohorts out of pretraining, validation and weight selection.
3. **Compare identical scored epochs.** For every arm save epoch IDs, class-ordered probabilities, montage, window, seed and checkpoint. Report coverage losses. Keep five- and six-class outcomes separate. On cohorts with only five reference stages, use the five-stage model for the primary comparison. In a secondary six-output analysis, count every M prediction on a scored five-stage epoch as an error and show its rate.
4. **Report uncertainty by person.** Give mean and SD across the five development folds, pooled per-stage F1, accuracy, kappa, class supports and raw and row-normalized confusion matrices. Use paired subject bootstraps for model differences and 95% intervals, retaining all nights per person. Report the N1 F1 contrast, N1 precision/recall, N1-W and N1-N2 confusions, transition accuracy, Brier score against vote fractions, and average precision for review flags.

### Decision and final release

Select architecture, preprocessing and the optional blend on development data. Prespecify the research primary contrast and the cohort before accessing external outcome labels. Run the frozen Sleep-EDFx comparison once on the 16 Cassette test participants and both Telemetry conditions. Run the prespecified external arms together and report all results, including negative findings. Treat additional comparisons as exploratory.

Evidence for an N1 improvement would be a positive D-minus-A difference with a paired 95% interval above zero on DOD-O. Use DOD-H to assess replication. Estimate change in macro-F1 and calibration alongside it. If that evidence is inconclusive, report the estimate and interval and retain the simpler development reference. A benchmark of epochs flagged for review supports a claim about prioritization. An observed time saving requires timed readers.

---

## Implementation handoff

### Repository changes in execution order

1. [x] **Freeze the baseline.** Keep `configs/splits/sc_subjects_v1.csv` and the verified campaign. Register the two window definitions, five- and six-class tasks, and publication endpoints under a new research configuration and output root. Store winning and reference probabilities.
2. [ ] **Extend intake.** Build on `src/sleep_staging/data/edf.py`, `data/labels.py` and `research_data.py`. Namespace participant and recording IDs by source. In the manifests record file version/hash, role, actual electrodes/reference, sampling rate, units, EMG type, annotation standard, epoch origin, rater ID, vote counts and event coverage. Anti-alias before downsampling EEG/EOG. Version any changed caches.
3. [x] **Implement the course model and upload path.** In `tcn_data.py`, `models/tcn.py` and `tcn_training.py`, carry channel presence and target vocabulary with each batch. Preserve time gaps. Add a versioned EDF-to-prediction path in `serving/contract.py` and its caller, alongside the current Parquet contract. Return the applied profile, eligibility, predicted class and class probabilities. Validate the 12 inputs, both EEG types alone and missing/unsupported channels.
4. [ ] **Add the paper experiments incrementally.** Reuse the fold handling, device selection and checkpoint logic behind `scripts/run_tcn_research.py`. Implement the four N1 arms first. Run the development-only spectrogram blend as a parallel scientific comparison. Add the HMC, MASS and SHHS experiments in separate configurations. Keep raw data and per-source rights in versioned manifests and record runs in MLflow.
5. [ ] **Release evidence.** Deliver two distinct confusion matrices for the activity, the channel-profile table, the literature comparison, and a reproducible model bundle. For the paper, save paired estimates, per-cohort tables, the label-efficiency curve when available, and the review-budget analysis. Keep new runs under a fresh output prefix so the reference bundle can be restored.

### Implementación de la revisión IBIO

La revisión de la propuesta priorizó tres experimentos con la base del curso. Los puntos 2, 4 y 5 de la hoja de ruta más amplia conservan trabajo para el estudio con fuentes externas y lectores.

- [x] Entrenar y evaluar LightGBM por perfil y CNN-TCN con canales opcionales. Verificar los 12 perfiles desde una carga de EDF real.
- [x] Comparar seis clases directas y detección jerárquica de movimiento. Publicar sensibilidad, precisión y falsas alarmas por registro.
- [x] Evaluar la fusión de ondas, espectrogramas y árboles. Desglosar los errores de N1 por contexto y reentrenar U-Sleep con Fpz-Cz y EOG.
- [x] Congelar la selección en Cassette desarrollo y puntuar los 16 participantes reservados y las dos condiciones de Telemetry. Guardar probabilidades, matrices, intervalos por participante y las corridas principales en MLflow.

Las [instrucciones de reproducción](../docs/ibio-registro-completo.md) indican los archivos implementados en `fullnight/` y `serving/edf.py`. El [informe de resultados](../output/pdf/resultados-ibio-registro-completo.pdf) presenta las tres comparaciones y sus límites.

### Verification and practical limits

Check synthetic fixtures for ingestion alignment, 20/30-second handling, missing rater/event masks, subject overlap, EEG presence, probability order and normalization. Use real development nights for model smoke runs. Check save/reload and prediction agreement within a declared tolerance on MPS and CUDA where available. Run three paired seeds for shortlisted neural arms and track wall time, peak memory and GPU-hours.

Existing entry points for regression checks and a short run are below. Run the second command on Apple hardware or substitute `cuda` on a CUDA machine.

```bash
.venv/bin/python -m pytest -q tests/unit/test_labels_epochs.py \
  tests/unit/test_tcn_research.py tests/unit/test_serving.py
.venv/bin/python scripts/run_tcn_research.py \
  --combination all --device mps --smoke
```

**Dependencies.** The core study can start from the verified local Sleep-EDFx files. Confirm access before budgeting MASS or SHHS processing. Independent reader availability determines whether the paper can measure time savings. Each optional study has its own outputs and can be deferred without changing the course evaluation.
