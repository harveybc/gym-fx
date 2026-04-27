# Phase 1 Overview — Data Acquisition

**Phase goal:** Acquire ALL plausibly relevant data sources for RL trading research, organized into a documented data lake.

**Phase output:** Complete data lake at `~/Documents/financial_data/` with:
- All raw data acquired
- Per-folder READMEs describing contents
- Per-dataset data dictionaries
- Provenance tracking (source URLs, acquisition dates, licenses)
- Validation reports per dataset

---

## Phase 1 Stages

| Stage | Document | Purpose |
|-------|----------|---------|
| 1.1 | `11_STAGE_1.1_STORAGE_ARCHITECTURE.md` | Build folder structure + templates |
| 1.2 | `12_STAGE_1.2_DATA_CATALOG.md` | Document EVERY data source we will acquire |
| 1.3 | `13_STAGE_1.3_FREE_DATA_ACQUISITION.md` | Acquire free sources |
| 1.4 | `14_STAGE_1.4_REGISTRATIONS_AND_KEYS.md` | User performs manual subscriptions/registrations |
| 1.5 | `15_STAGE_1.5_PAID_DATA_ACQUISITION.md` | Acquire paid sources |
| 1.6 | `16_STAGE_1.6_VALIDATION_AND_DOCUMENTATION.md` | Validate, document, audit |

---

## Stage Dependencies

```
1.1 Storage Architecture (foundation, no dependencies)
    │
    ▼
1.2 Data Catalog (decides what to acquire)
    │
    ├──▶ 1.3 Free Data Acquisition (parallel with 1.4)
    │
    └──▶ 1.4 Registrations and Keys (user manual work)
              │
              ▼
         1.5 Paid Data Acquisition (depends on 1.4)
              │
              ▼ (joins with 1.3)
         1.6 Validation and Documentation (final stage)
```

Stages 1.3 and 1.4 can run in parallel — agent acquires free data while user handles registrations.

---

## Phase 1 User Gates

After each stage, agent produces deliverable, halts, awaits user approval:

- After 1.1: User reviews folder structure
- After 1.2: User reviews catalog and approves subscription decisions
- After 1.3: User reviews free data acquisition results
- After 1.4: User confirms all registrations complete and credentials provided
- After 1.5: User reviews paid data acquisition results
- After 1.6: User approves Phase 1 complete, Phase 2 may start

---

## Phase 1 Deliverables Summary

By end of Phase 1, the following must exist:

1. Complete data lake at `~/Documents/financial_data/` with all subfolders populated
2. README.md in every folder
3. data_dictionary.md for every dataset folder
4. provenance.json for every dataset folder
5. `PHASE_1_COMPLETION_REPORT.md` documenting:
   - Inventory of all acquired data
   - Total data volume
   - Source breakdown (free vs paid)
   - Total monthly subscription cost
   - Any sources that failed to acquire (with reasons)
   - Coverage gaps known
   - Validation summary

---

## Critical Rules for Phase 1

### Rule P1.1: Acquire defensively

When in doubt about a source's value, acquire it. Filtering happens in Phase 3 experiments. Storage is unlimited.

### Rule P1.2: Document at acquisition time

Do NOT defer documentation. As each dataset is acquired, immediately write its README, data_dictionary, and provenance. "Documentation later" means "documentation never."

### Rule P1.3: Validate at acquisition time

Each dataset acquired runs through Stage II-0b validation tests (bar count, weekend gaps where applicable, fat tails, etc.). Failed validation triggers ESCALATION.

### Rule P1.4: Original data preserved

Acquired raw data never modified. If transformations needed (resampling, cleaning), they happen in Phase 2 and produce new files. Raw is sacred.

### Rule P1.5: Source credentials never committed to git

API keys, subscription credentials in `.env` files. All `.env` files in `.gitignore`. Commit credential references, not credential values.

### Rule P1.6: Acknowledge what we are NOT acquiring

Some data sources are explicitly excluded. Stage 1.2 documents exclusions with reasons. Examples:
- Bloomberg Terminal (cost-prohibitive, not adding value over alternatives)
- Real-time tick data archives (not needed for non-HFT research)
- Specific exotic alt-data with no demonstrated signal value

---

## Approval to Begin Phase 1

User approves Phase 1 overview. Agent reads `11_STAGE_1.1_STORAGE_ARCHITECTURE.md` and begins Stage 1.1.
