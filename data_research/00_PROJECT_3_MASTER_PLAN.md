# Project 3 — Master Plan

**Project name:** Comprehensive Data Acquisition + Feature Engineering + Systematic RL Evaluation
**Predecessor:** Project 2 (closed with marginal RL signal on BTC 1h technical features)
**Core hypothesis:** Project 2's null/marginal results were caused by insufficient data diversity, not by RL algorithms being unsuitable. By systematically acquiring ALL plausibly relevant data and engineering ALL state-of-the-art feature representations, we will identify which data combinations enable RL trading agents to find genuine signal.

---

## 1. Project Philosophy

This project is **data-centric, perfectionist, and exhaustive**. Three operational principles override everything else:

1. **Data first.** Models are not the bottleneck. Data is. We acquire EVERYTHING plausibly relevant before fitting any new model.
2. **Comprehensive over selective.** When in doubt, include the data source. Filtering happens during experiments (Phase 3), not during acquisition (Phase 1).
3. **Documented and organized.** Every folder has a README. Every dataset has a data dictionary. Future-Harvey or future-agent must be able to understand any folder without context.

---

## 2. Three-Phase Structure

| Phase | Focus | Output |
|-------|-------|--------|
| **Phase 1: Data Acquisition** | Acquire ALL plausibly relevant data sources | Organized data lake with READMEs |
| **Phase 2: Feature Engineering** | Convert raw data into all reasonable feature representations | Standardized feature library |
| **Phase 3: Systematic Experiments** | Use Project 2's best RL configs to test which data subsets improve performance | Evidence-based ranking of data sources |

Phases are sequential. Phase 2 cannot start until Phase 1 produces validated organized data. Phase 3 cannot start until Phase 2 produces feature library.

---

## 3. Document Index

This project is organized as multiple short documents (idiot-proof for inferior agent models). Read only the document for the stage being executed.

### Master
- `00_PROJECT_3_MASTER_PLAN.md` — This document

### Phase 1: Data Acquisition
- `10_PHASE_1_OVERVIEW.md` — Phase 1 goals and dependencies
- `11_STAGE_1.1_STORAGE_ARCHITECTURE.md` — Folder structure, naming conventions, README templates
- `12_STAGE_1.2_DATA_CATALOG.md` — Exhaustive list of every data source to acquire
- `13_STAGE_1.3_FREE_DATA_ACQUISITION.md` — Procedures for free data sources
- `14_STAGE_1.4_REGISTRATIONS_AND_KEYS.md` — Manual user actions (subscriptions, accounts)
- `15_STAGE_1.5_PAID_DATA_ACQUISITION.md` — Procedures for subscription-based sources
- `16_STAGE_1.6_VALIDATION_AND_DOCUMENTATION.md` — Per-folder validation and READMEs

### Phase 2: Feature Engineering
- `20_PHASE_2_OVERVIEW.md` — Phase 2 goals and dependencies
- `21_STAGE_2.1_DOWNSAMPLING_AND_RESAMPLING.md` — Multi-timeframe generation
- `22_STAGE_2.2_TECHNICAL_AND_STATISTICAL_FEATURES.md` — Standard + advanced features
- `23_STAGE_2.3_SIGNAL_DECOMPOSITION_FEATURES.md` — Wavelet, Hilbert, multitaper, EMD
- `24_STAGE_2.4_LEARNED_REPRESENTATIONS.md` — Autoencoders + embeddings

### Phase 3: Systematic Experiments
- `30_PHASE_3_OVERVIEW.md` — Phase 3 goals and dependencies
- `31_STAGE_3.1_EXPERIMENT_FRAMEWORK.md` — How to test data subsets systematically
- `32_STAGE_3.2_RESULTS_SYNTHESIS.md` — How findings get aggregated

Total: 14 documents (master + 13 stage docs).

---

## 4. Standing Rules (Apply to ALL Stages)

### Rule M.1: Read only the relevant document

Agent reads the master plan + the specific stage document being executed. Does not attempt to read all documents at once. Does not skip ahead.

### Rule M.2: Each stage has a user gate

After completing a stage, agent produces stage deliverable, HALTS, and waits for user approval before next stage.

### Rule M.3: Held-out data discipline preserved from Project 2


# TODO 1: CORREGIR, porque usa solo hasta 2024 si ya tenemos algunos datos de sde todo 2025, de hecho los datos ya descargados de histdata de eurusd y usdjpy  están creo de 11 hasta fin del año 2025
```
HELD_OUT_BOUNDARY = "2024-01-01"
```

Data acquired covers full history (no cutoff during acquisition). But during Phase 3 experiments, data from 2024-01-01 onward is held-out and touched exactly once per final candidate.

(Note: HO boundary changed from 2020-01-01 in Project 2 to 2024-01-01 in Project 3 because new data goes back to 2024 anyway and we need substantial training data. Pre-registered for transparency.)

### Rule M.4: Organization is non-negotiable

Every folder MUST have:
- `README.md` describing contents
- `data_dictionary.md` if folder contains datasets (column descriptions, units, source, date ranges)
- `provenance.json` (machine-readable: source URL, acquisition date, license, version)

Folders without these three files are considered incomplete and trigger ESCALATION.

### Rule M.5: Quality over speed

This project explicitly rejects "fast and mediocre." If a stage takes 3× longer to do correctly, do it correctly. Time budget non-binding (per user direction).

### Rule M.6: No invented techniques

For Phase 2 feature engineering, every technique used MUST cite published source (Jansen, Lopez de Prado, Chan, Tsay, or peer-reviewed paper). No "I think this might work" features. State of the art only.

### Rule M.7: Idiot-proof execution

Each stage document is written assuming agent has zero project context except master plan + stage document. Cross-references explicit. Procedures step-by-step. No assumptions about implicit knowledge.

### Rule M.8: SSH + conda activation pattern

# TODO 2:  recordarle que la activación del env de conda llamado tensorflow, se hace desde el .bashrc en todas las máquinas, también se agregan los diversos paths requeridos apra que se pueda usar la GPU correctamente, entonces si se conecta por ssh en una forma que se lea el bashrc automáticamente, nos e debe volver a activar el conda env, ya que produce errores al activarse dos veres, pero si se conecta por ssh no interactivamente, no estoy seguro de que se active el bashrc entonces si debería activarse con cada conexion.


All remote commands use:
```bash
source /home/harveybc/anaconda3/etc/profile.d/conda.sh && conda activate tensorflow && <command>
```

### Rule M.9: Storage assumed unlimited

User has confirmed sufficient disk space. Do not optimize for storage. Acquire everything that has plausible signal value.

### Rule M.10: Cost ceiling

Maximum subscription cost: **$500/month combined across all paid sources**. Within this limit, choose subscriptions that:
- Provide data not available free
- Have clearly documented signal value (cited in Jansen/Lopez de Prado/etc.)
- Cover gaps in free sources

Reject Bloomberg Terminal ($24K/yr), Refinitiv ($22K/yr), institutional-only feeds. Use Polygon.io, Alpha Vantage Premium, Glassnode Advanced, CryptoQuant Pro, Quandl/Nasdaq Data Link selects.

---

## 5. Reference Books and Sources

The data catalog (Stage 1.2) and feature engineering (Phase 2) draw from:

| Reference | Focus |
|-----------|-------|
| Jansen, "Machine Learning for Algorithmic Trading" 2nd ed (2020) | Primary reference. Comprehensive data sources + ML techniques |
| Lopez de Prado, "Advances in Financial Machine Learning" (2018) | Microstructure, meta-labeling, fractional differentiation |
| Chan, "Machine Trading" (2017) + "Algorithmic Trading" (2013) | Practical strategy construction, order types |
| Tsay, "Analysis of Financial Time Series" 3rd ed (2010) | Statistical methods, GARCH, regime models |
| Selected papers per technique (cited in stage documents) | State-of-the-art for each Phase 2 method |

---

## 6. Connection to Project 2 Outputs

Project 3 USES (does not redo):

- Project 2 RL infrastructure (gym-fx + agent-multi if built; else infrastructure/rl/trading_env.py from Stage II-7.4)
- Project 2 best-performing model configurations (PPO BTC 1h, etc.) as starting points for Phase 3 experiments
- Project 2 evaluation framework (F-10 kill criteria, DSR with multiple-testing correction, IS/HO discipline)
- Project 2 lessons (no synthetic data, sanity checks mandatory, etc.)

Project 3 DOES NOT redo Project 2's algorithm research. Models from Project 2 are tools for testing data, not subjects of further investigation in this project.

---

## 7. Project 3 Final Deliverable

After Phase 3 completes:

`PROJECT_3_FINAL_REPORT.md` answers:

# TODO 3: porque se habl a de esa tal signal? recuerda que no estamos tratande de hacer predicctiones sino usar los datos como entrada de nuestros modelos de rl

1. Which data sources, in combination, produced highest RL trading performance?
2. Which feature engineering techniques provided most signal?
3. Which timeframes are most signal-rich?
4. What is the realistic Sharpe achievable with optimal data + Project 2 best models?
5. What data sources turned out to be noise / non-contributory?
6. What gaps remain (data not acquired, techniques not tested) for future projects?

This report becomes the foundation for any future Project 4 (e.g., NEAT capstone, productization, etc.).

---

## 8. Risk Acknowledgments

# TODO 4: si un provedor pagado es mediocre, pues no usarlo.

1. **Probability that Project 3 still produces null result: estimated 30-40%.** Even with comprehensive data, retail-accessible markets may not have systematic exploitable edge for RL at the scales we're working. If null, the documented data + features library remains valuable for any future paradigm.

2. **Some data sources may not deliver promised value despite cost.** Phase 3 experiments will identify these; Phase 1 acquires defensively.

3. **Subscription value depends on data quality, not provider name.** Cheap providers can be excellent (e.g., FRED is free and gold-standard). Expensive providers can be mediocre.

4. **Time investment is substantial.** Even with unlimited compute, manual user tasks (registrations, subscription decisions) require user time.

---

## 9. Approval to Begin

User approves master plan. Agent begins by reading `10_PHASE_1_OVERVIEW.md`.
