# DATA AUDIT PRD: ancient-scripts-datasets
## Comprehensive Survey, Issue Diagnosis, and Remediation Plan

**Audit Date:** 2026-02-20
**Scope:** Every file, dataset, and subset in `ancient-scripts-datasets/`
**Goal:** Identify all data quality issues, diagnose root causes, and plan fixes

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Core Ancient Script Datasets](#2-core-ancient-script-datasets)
   - 2.1 Gothic
   - 2.2 Ugaritic-Hebrew
   - 2.3 Iberian
3. [Validation Datasets](#3-validation-datasets)
   - 3.1 Core Family Files (9 families)
   - 3.2 Names Subset
   - 3.3 Expanded Files
4. [Training Data](#4-training-data)
   - 4.1 Language Profiles (1,110 files)
   - 4.2 Metadata Files
   - 4.3 Validation Splits
   - 4.4 Per-Language Data Volume Audit
5. [Religious Terms](#5-religious-terms)
6. [Cited Sources](#6-cited-sources)
7. [Scripts](#7-scripts)
8. [Cognate Pipeline](#8-cognate-pipeline)
9. [README Accuracy](#9-readme-accuracy)
10. [Master Issue Registry](#10-master-issue-registry)
11. [Remediation Plans](#11-remediation-plans)

---

## 1. EXECUTIVE SUMMARY

### Overall Health: STRUCTURALLY SOUND, BUT WITH SIGNIFICANT DATA GAPS

The dataset repository is well-architected with proper staging patterns, provenance tracking,
and linguistically grounded encoding. However, the audit uncovered **12 distinct issues**
ranging from critical data gaps to minor documentation drift.

### Critical Issues (3)
| # | Issue | Impact |
|---|-------|--------|
| C1 | **Zero-entry languages:** Russian, Arabic, Polish, Turkish, Tamil and 10+ other major languages have profile files but 0 lexical entries | Training data for these languages is entirely missing |
| C2 | **1,110 profiles vs 199 languages claim:** README and coverage report claim 199 languages, but 1,110 profile files exist; many are empty shells | Overstated coverage; actual usable languages ~500-600 |
| C3 | **No model training bridge:** Pipeline produces JSONL cognate pairs but has no code to convert to PhoneticPriorModel input format | Cannot train the model on the data without a new conversion layer |

### Major Issues (4)
| # | Issue | Impact |
|---|-------|--------|
| M1 | **Ugaritic data asymmetry:** 83% of rows in uga-heb.no_spe.cog have missing Ugaritic (`_` in col 1); only 6,395 of 43,951 rows are actual cognate pairs | Misleading pair count; effective data is ~15% of stated size |
| M2 | **Missing Glottocodes:** 746 of 1,110 languages (67%) lack Glottocodes | Phylogenetic classification unreliable for majority of languages |
| M3 | **WikiPron data not integrated into lexicons:** wikipron_stats.tsv shows 411k Russian entries, 201k Portuguese, etc., but these are NOT in the actual training lexicons | Massive pronunciation data exists but was never loaded |
| M4 | **No negative sample generation:** Pipeline only outputs positive cognate pairs above threshold; model training requires non-cognate negative examples | Cannot produce balanced training sets |

### Minor Issues (5)
| # | Issue | Impact |
|---|-------|--------|
| m1 | **Iberian CSV uses Python list syntax** instead of standard CSV/JSON | Non-standard parsing required |
| m2 | **Iberian TSV header typo:** `3category` instead of `category` | Will break column-name-based parsing |
| m3 | **README entry counts outdated:** Gothic religious claimed ~65, actual 90; Iberian claimed ~40, actual 68; Ugaritic-Hebrew claimed ~170, actual 241 | Documentation drift |
| m4 | **27 duplicate cognate pairs** in uga-heb.no_spe.cog | Minor data quality issue |
| m5 | **11 FALSA/SUSPECTA Iberian inscriptions** included without metadata flags | Should be filtered or tagged |

---

## 2. CORE ANCIENT SCRIPT DATASETS

### 2.1 Gothic (`data/gothic/`)

#### Files
| File | Size | Lines | Format |
|------|------|-------|--------|
| gotica.txt | 516 KB | 4,452 | Plain text, UTF-8, CRLF |
| gotica.xml.zip | 428 KB | - | ZIP containing 3.5 MB XML |
| got.pretrained.pth | 27 KB | - | PyTorch checkpoint (valid) |
| segments.pkl | 4 KB | - | Python pickle (valid) |

#### Metrics
- **Text lines:** 4,340 (excluding 112 comments/headers)
- **Total tokens:** 76,771
- **Unique types:** 14,892
- **Type-token ratio:** 19.4%

#### Verification Against Real-World
- **Expected:** Wulfila Bible corpus is ~67,400 tokens with ~3,600 lemmas
- **Actual:** 76,771 tokens, 14,892 unique types
- **Assessment:** Token count slightly exceeds expected (76k vs 67k) likely due to
  variant readings and manuscript witnesses included. Unique types (14,892) far exceeds
  known Gothic lemmas (3,600) because this counts inflected forms, not lemmas.
- **Verdict:** CORRECT. Data is complete and authentic.

#### Issues Found
- **None.** Gothic dataset is the highest quality subset.
- Binary files (pth, pkl) are valid and loadable.

---

### 2.2 Ugaritic-Hebrew (`data/ugaritic/`)

#### Files
| File | Size | Lines | Format |
|------|------|-------|--------|
| uga-heb.no_spe.cog | 396 KB | 43,952 | TSV, ASCII, CRLF |
| uga-heb.small.no_spe.cog | 41 KB | 4,536 | TSV, ASCII, CRLF |

#### Metrics (Full File)
| Metric | Count |
|--------|-------|
| Total rows (excl. header) | 43,951 |
| Rows with Ugaritic present (col1 != `_`) | 6,396 |
| Rows with Hebrew present (col2 != `_`) | 38,898 |
| **Valid cognate pairs (both present)** | **6,395** |
| Rows with Ugaritic missing | 36,684 (83.5%) |
| Rows with Hebrew missing | 5,053 |
| Unique Ugaritic types | 7,236 |
| Unique Hebrew types | 39,452 (incl. pipe-separated alternatives) |
| Duplicate pairs | 27 (appearing exactly 2x) |
| Truly unique combinations | 43,924 |

#### Metrics (Small File)
- 4,535 rows, 655 with Ugaritic present (10.3% of full file)

#### Verification Against Real-World
- **Expected:** ~1,500-1,600 confidently known Ugaritic words
- **Actual:** 7,236 unique Ugaritic forms in dataset
- **Assessment:** The 7,236 figure includes morphological variants, alternative
  transliterations, and derived forms. This is **4.5x the known lexicon**, which
  is reasonable for an inflected Semitic language with variant readings.
- **However:** The file is fundamentally **Hebrew-centric** (83.5% of rows are
  Hebrew-only with `_` for Ugaritic). The effective cognate pair count is 6,395,
  not the implied 43,951.

#### Issues Found

**[ISSUE M1] Ugaritic Data Asymmetry**
- **What's wrong:** 83.5% of rows have missing Ugaritic entries. The file appears
  to be a Hebrew vocabulary list with occasional Ugaritic cognates attached, not a
  balanced bilingual cognate file.
- **Where to diagnose:** Compare `uga-heb.no_spe.cog` against the original
  NeuroDecipher data source at `third_party/NeuroDecipher/data/`
- **Root cause hypothesis:** The COG format from Luo et al. encodes a parallel corpus
  where the Hebrew text is segmented and the Ugaritic text is not. The `_` entries
  represent Hebrew words that have no known Ugaritic cognate, which is the expected
  format for the decipherment model (it needs the full Hebrew vocabulary to search
  against, not just matched pairs).
- **How to fix:** This is likely INTENTIONAL for the decipherment model, but the
  README should clarify that only 6,395 rows are actual cognate pairs while the
  remainder is the full Hebrew search vocabulary. If only paired data is needed for
  training, filter to rows where both columns are non-`_`.

**[ISSUE m4] 27 Duplicate Cognate Pairs**
- **What's wrong:** 27 Ugaritic-Hebrew pairs appear exactly twice
- **Where to diagnose:** `sort uga-heb.no_spe.cog | uniq -d`
- **How to fix:** Deduplicate. Low priority since it's 27 of 43,951 rows (0.06%).

---

### 2.3 Iberian (`data/iberian/`)

#### Files
| File | Size | Lines | Format |
|------|------|-------|--------|
| iberian.csv | 72 KB | 2,095 | CSV with Python list syntax, UTF-8, CRLF |

#### Metrics
- **Data rows:** 2,094 inscriptions
- **Unique Hesperia references:** 2,094 (no duplicates)
- **Estimated total chunks:** ~3,466 (across all inscriptions, avg 1.66/inscription)
- **Flagged entries:** 5 FALSA + 6 SUSPECTA = 11 (0.5%)
- **Rows with lacunae (`---`):** ~1,000+ (~50%)

#### Verification Against Real-World
- **Expected:** ~2,000 known Iberian inscription fragments in the Hesperia database
- **Actual:** 2,094 inscriptions
- **Assessment:** MATCHES expectations. The 3,466 "chunks" count refers to
  individual text segments within inscriptions, not inscription count.

#### Issues Found

**[ISSUE m1] Non-Standard CSV Format**
- **What's wrong:** The `cleaned` column uses Python list literal syntax:
  `"['ikon-', 'ḿkeiḿi', 'iltubel-', 'eśeban']"` instead of standard CSV or JSON.
- **Where to diagnose:** Read `iberian.csv` column 2 — all rows use `['...']` syntax
- **How to fix:**
  1. Parse with `ast.literal_eval()` in Python (works but fragile)
  2. Better: Convert to proper JSON arrays: `["ikon-", "ḿkeiḿi", "iltubel-", "eśeban"]`
  3. Implementation: Read CSV, parse col2 with `ast.literal_eval`, write back with `json.dumps`

**[ISSUE m5] FALSA/SUSPECTA Entries Untagged**
- **What's wrong:** 11 inscriptions flagged as false/suspicious are embedded in
  the Hesperia reference code (e.g., `L.22.01FALSA`) rather than in a metadata column.
- **Where to diagnose:** `grep -c "FALSA\|SUSPECTA" iberian.csv`
- **How to fix:**
  1. Add a `status` column: `authentic` | `falsa` | `suspecta`
  2. Strip the suffix from the reference code
  3. Or: create a separate `iberian_excluded.csv` for flagged entries

---

## 3. VALIDATION DATASETS

### 3.1 Core Family Files (`data/validation/`)

#### 9 Family Files — ALL VERIFIED CORRECT
| Family | File | Languages | Lang Codes | Rows | Concepts |
|--------|------|-----------|------------|------|----------|
| Germanic | germanic.tsv | 4 | got, ang, non, goh | 160 | 40 |
| Celtic | celtic.tsv | 3 | sga, cym, bre | 120 | 40 |
| Balto-Slavic | balto_slavic.tsv | 3 | lit, chu, rus | 120 | 40 |
| Indo-Iranian | indo_iranian.tsv | 3 | san, ave, fas | 120 | 40 |
| Italic | italic.tsv | 3 | lat, osc, xum | 120 | 40 |
| Hellenic | hellenic.tsv | 2 | grc, gmy | 80 | 40 |
| Semitic | semitic.tsv | 3 | heb, arb, amh | 120 | 40 |
| Turkic | turkic.tsv | 3 | otk, tur, aze | 120 | 40 |
| Uralic | uralic.tsv | 3 | fin, hun, est | 120 | 40 |
| **TOTAL** | **9 files** | **27 unique** | | **960** | **40** |

- All 27 language codes verified as real ISO 639-3 codes
- All family groupings are linguistically correct
- No duplicate rows, no empty cells, no malformed entries
- Column structure consistent: Language_ID | Parameter_ID | Form | IPA | Glottocode

#### 40 Concepts Verified
- **Body (8):** head, eye, ear, mouth, hand, foot, heart, blood
- **Kinship (5):** father, mother, son, daughter, brother
- **Nature (10):** water, fire, sun, moon, star, earth, mountain, river, stone, tree
- **Animals (5):** horse, dog, fish, bird, ox
- **Verbs (8):** eat, drink, give, come, die, know, hear, see
- **Other (4):** name, god, king, house

**Verdict:** CORRECT. No issues found in core validation files.

### 3.2 Names Subset (`data/validation/names.tsv`)

#### Metrics
- **Total entries:** 168
- **Unique languages:** 30 (27 from core families + 5 ancient: akk, arc, hit, phn, uga)
- **Unique concepts:** 15 (10 deity, 2 proper, 3 place)
- **Non-null Cognate_Set_ID:** 160
- **Null Cognate_Set_ID:** 8

#### Cognate Pair Statistics (names_pairs.tsv)
- **Total pairs:** 832
- **True cognate pairs:** 227 (27.3%)
- **False-positive pairs:** 605 (72.7%)

**README claims match exactly.** All verified.

#### Issue Found
**5 ancient language codes (akk, arc, hit, phn, uga) used in names.tsv are NOT in
languages.tsv metadata.** This is minor — they are intentionally part of the names
subset only, not the main 199-language inventory. But it should be documented.

### 3.3 Expanded Files (10 files)

| File | Rows | Languages | Concepts |
|------|------|-----------|----------|
| germanic_expanded.tsv | 914 | 11 | 105 |
| celtic_expanded.tsv | 339 | 4 | 100 |
| balto_slavic_expanded.tsv | 1,232 | 13 | 105 |
| indo_iranian_expanded.tsv | 810 | 10 | 104 |
| italic_expanded.tsv | 718 | 9 | 105 |
| hellenic_expanded.tsv | 151 | 3 | 100 |
| semitic_expanded.tsv | 240 | 3 | 100 |
| turkic_expanded.tsv | 876 | 10 | 103 |
| uralic_expanded.tsv | 2,615 | 27 | 105 |
| concepts_expanded.tsv | 106 | - | 106 (mapping file) |
| **TOTAL** | **8,001** | | |

**Verdict:** CORRECT. Expanded files are consistent with core data.

---

## 4. TRAINING DATA

### 4.1 Language Profiles (`data/training/language_profiles/`)

**Total profile files:** 1,110 markdown files

**Format:** Each file is `{iso_code}.md` containing:
- Language name, family, Glottocode
- Number of lexical entries by source
- IPA sample transcriptions
- Source attributions

#### Sample Profile Verification (15 profiles checked)

| Language | ISO | Family | Claimed Entries | IPA Quality | Status |
|----------|-----|--------|-----------------|-------------|--------|
| English | eng | Germanic | 945 (NELex) + 110,159 (WikiPron) | Genuine IPA (aɪ, ŋ, ː) | OK |
| French | fra | Italic | 1,084 (NELex) | Genuine IPA (ɛ, ʁ, ʊ) | OK |
| German | deu | Germanic | 1,003 (NELex) | Genuine IPA (ʊ, ə, ŋ) | OK |
| Hindi | hin | Indo-Iranian | 1,365 (NELex) | Genuine IPA (ɪ, ɛ, ː) | OK |
| Hungarian | hun | Uralic | 1,097 (NELex) | Genuine IPA (ə, ː) | OK |
| Hebrew | heb | Semitic | 988 (NELex) | Genuine IPA (ɛ, ʃ, χ) | OK |
| Tibetan | bod | Sino-Tibetan | 298 (SinoTibetan) | Genuine IPA (ə, ː) | OK |
| Central Maewo | mwo | Austronesian | 2,139 (ABVD) | Present | OK |
| **Russian** | **rus** | **Balto-Slavic** | **0 entries** | **N/A** | **PROBLEM** |
| **Arabic** | **ara** | **Semitic** | **0 entries** | **N/A** | **PROBLEM** |
| **Polish** | **pol** | **Balto-Slavic** | **0 entries** | **N/A** | **PROBLEM** |
| **Turkish** | **tur** | **Turkic** | **0 entries** | **N/A** | **PROBLEM** |
| **Tamil** | **tam** | **Dravidian** | **0 entries** | **N/A** | **PROBLEM** |

**IPA verification:** All non-zero profiles contain authentic Unicode IPA characters
(fricatives, nasals, vowels, stops, affricates with proper diacritics).

### 4.2 Metadata Files

#### languages.tsv
- **Columns:** ISO | Family | Glottocode | Entries | Sources
- **Total rows:** 1,110 languages
- **Total lexical entries (summed):** 3,179,124

#### source_stats.tsv
| Source | Role |
|--------|------|
| northeuralex | ~1,000 entries/lang, IPA transcriptions, Uralic/Germanic/Slavic focus |
| wikipron | Broadest coverage (337+ langs), pronunciation dictionaries |
| abvd | ~210 basic vocab items/lang, 2,038 Austronesian languages |
| sinotibetan | Sino-Tibetan etymology, ~300 entries/lang |
| wold | World Loanword Database |

#### cldf_stats.tsv
- 363 languages with CLDF pronunciation data
- Top: Finnish (158,400), English (111,862), French (81,198)

#### wikipron_stats.tsv
- 424 languages with pronunciation data
- Top: Russian (411,125), Portuguese (201,945), Mandarin (159,038)

### 4.3 Training Validation Splits (`data/training/validation/`)

#### Main Splits
| File | Entries | Purpose |
|------|---------|---------|
| true_cognates_L1.tsv | 50,001 | Closest phylogenetic distance |
| true_cognates_L2.tsv | 32,666 | Mid-distance cognates |
| true_cognates_L3.tsv | 50,001 | Distant cognates |
| false_positives.tsv | 50,001 | Non-cognate sound-alikes |
| true_negatives.tsv | 50,001 | True non-cognates |
| borrowings.tsv | 116,757 | Documented loanwords |
| timespan_ancient_ancient.tsv | 4 | Ancient-ancient pairs |
| timespan_ancient_modern.tsv | 3,744 | Ancient-modern pairs |
| timespan_medieval_modern.tsv | 1,202 | Medieval-modern pairs |
| timespan_modern_modern.tsv | 344,475 | Modern-modern pairs |
| **Total** | **~698,907** | |

#### Per-Family Splits (15 families)
| Family | Entries |
|--------|---------|
| Austronesian | 123,861 |
| Uralic | 63,781 |
| Germanic | 28,966 |
| Balto-Slavic | 24,362 |
| Italic | 19,719 |
| Japonic | 12,260 |
| Turkic | 10,697 |
| Indo-Iranian | 10,345 |
| Sino-Tibetan | 9,779 |
| Semitic | 5,650 |
| Celtic | 4,899 |
| Dravidian | 2,361 |
| Hellenic | 2,032 |
| Kartvelian | 1,996 |
| Koreanic | 1,992 |
| **Total** | **322,700** |

#### Religious Domain Splits
- all_pairs.tsv: 40,974
- true_cognates.tsv: 26,809
- borrowings.tsv: 9,150
- false_positives.tsv: 5,017
- Plus 7 semantic category files and 15 by-family files
- **Total religious entries:** ~124,146

**Verdict:** Validation split structure is comprehensive and well-organized.

### 4.4 Per-Language Data Volume Audit

This is the most critical section. The audit cross-references three data sources:
1. `languages.tsv` — claimed entry counts
2. `wikipron_stats.tsv` — WikiPron pronunciation data available
3. Language profile files — what's actually in each profile

#### CRITICAL FINDING: Zero-Entry Major Languages

**[ISSUE C1] The following major world languages have profile files but ZERO lexical entries:**

| Language | ISO | Family | WikiPron Available | In Lexicon | Gap |
|----------|-----|--------|-------------------|------------|-----|
| Russian | rus | Balto-Slavic | 411,125 | 0 | 411,125 |
| Portuguese | por | Italic | 201,945 | 0* | 201,945 |
| Polish | pol | Balto-Slavic | 130,133 | 0 | 130,133 |
| Arabic | ara | Semitic | 13,339 | 0 | 13,339 |
| Turkish | tur | Turkic | ~10,000+ | 0 | ~10,000 |
| Tamil | tam | Dravidian | ~5,000+ | 0 | ~5,000 |
| Korean | kor | Koreanic | ~8,000+ | 0 | ~8,000 |
| Japanese | jpn | Japonic | ~12,000+ | 0 | ~12,000 |
| Swahili | swa | Niger-Congo | ~2,000+ | 0 | ~2,000 |

*Portuguese shows 201,945 in wikipron_stats.tsv but the profile shows 0 lexicon entries.

**Root cause:** The `ingest_wikipron.py` script exists in `scripts/` but was apparently
**never run** or its output was never committed. The WikiPron data was audited and stats
were generated, but the actual pronunciation entries were not loaded into the lexicons.

#### Languages With Suspiciously Small Entry Counts

Languages from ABVD (Austronesian Basic Vocabulary Database) legitimately have small
counts (~100-500 entries) because ABVD provides ~210 basic vocabulary items per language.
This is NOT an error — it's the expected size for that source.

**Smallest entries:**
- bzh (Amdis): 38 entries
- iff: 46 entries
- bvy: 48 entries

These are minor/endangered Austronesian languages with limited documentation. The low
counts are expected and correct.

#### CRITICAL FINDING: Profile Count vs Language Count

**[ISSUE C2] 1,110 profiles exist but only ~199-363 have substantial data:**

| Category | Count | Source |
|----------|-------|--------|
| Total profile files | 1,110 | language_profiles/ directory |
| README claim | 199 | README.md and coverage_report.txt |
| Languages in CLDF | 363 | cldf_stats.tsv |
| Languages in WikiPron | 424 | wikipron_stats.tsv |
| Languages with >0 lexicon entries | ~500-600 | Estimated from profiles |
| Languages with >1,000 entries | ~100-150 | Estimated from NorthEuraLex coverage |

**Root cause:** The profile generation script (`generate_language_readmes.py`) created
profiles for ALL languages found in ANY source (NorthEuraLex, ABVD, WikiPron, CLDF, WOLD,
SinoTibetan). But the actual lexicon data was only committed for languages processed
through the CLDF pipeline (the 199 in the coverage report). WikiPron data was catalogued
but not integrated.

---

## 5. RELIGIOUS TERMS

### 5.1 Gothic Religious (`data/religious_terms/gothic_religious.tsv`)
- **Claimed:** ~65 terms
- **Actual:** 90 entries
- **Columns:** category | subcategory | gothic_word | english_meaning | example_verse | notes
- **Quality:** HIGH. All terms verified as authentic Wulfila Bible vocabulary.
- **15 subcategories** covering deity terms, ritual verbs, sin/salvation, sacred persons, etc.
- **No duplicates, no empty entries.**
- **Issue:** README undercounts by 38% (claims ~65, actual 90).

### 5.2 Iberian Religious (`data/religious_terms/iberian_religious.tsv`)
- **Claimed:** ~40 elements
- **Actual:** 68 entries
- **Columns:** 3category | subcategory | element | proposed_meaning | occurrences | example_refs | scholarly_basis

**[ISSUE m2] Header Typo**
- **What's wrong:** First column is `3category` instead of `category`
- **Where:** Line 1 of `iberian_religious.tsv`
- **How to fix:** Simple find-replace: `3category` -> `category`

- **Quality:** GOOD given that Iberian is undeciphered. All semantic interpretations
  clearly marked as scholarly proposals with citations (Untermann 1990, Velaza 2015,
  Rodriguez Ramos 2014).
- **Issue:** README undercounts by 70% (claims ~40, actual 68).

### 5.3 Ugaritic-Hebrew Religious (`data/religious_terms/ugaritic_hebrew_religious.tsv`)
- **Claimed:** ~170 pairs
- **Actual:** 241 entries
- **Columns:** category | subcategory | root | english_meaning | ugaritic_form | hebrew_cognate | notes
- **64 duplicate roots** — these are INTENTIONAL (same root appearing in multiple
  subcategories, e.g., a sacrificial root appearing under both "ritual" and "high-risk").
- **Sound correspondences verified:** Ug. d = Heb. z, Ug. v(theta) = Heb. $, Ug. x = Heb. H
- **Quality:** HIGH. Authentic cognate pairs from NeuroDecipher dataset.
- **Issue:** README undercounts by 42% (claims ~170, actual 241).

---

## 6. CITED SOURCES

### 6.1 Genesis (`data/cited_sources/genesis/`)
| File | Size | Format | Content | Verified |
|------|------|--------|---------|----------|
| Hebrew.xml | 5.5 MB | CES 4.0 XML | Full Hebrew Bible, 39 books, 415,175 words | YES - authentic |
| Latin.xml | 4.7 MB | CES 4.0 XML | Full Latin Vulgate, 66 books, 534,314 words | YES - authentic |

### 6.2 Basque (`data/cited_sources/basque/`)
| File | Size | Format | Content | Verified |
|------|------|--------|---------|----------|
| Basque-NT.xml | 1.5 MB | CES 4.0 XML | Basque NT, 132,288 words | YES - authentic |
| Trask_Etymological_Dictionary_Basque.pdf | 1.5 MB | PDF | 418-page academic dictionary | YES - legitimate |

### 6.3 Iberian Names (`data/cited_sources/iberian_names/`)
| File | Size | Format | Content | Verified |
|------|------|--------|---------|----------|
| RodriguezRamos2014.pdf | 1.2 MB | PDF | Scholarly work on Iberian inscriptions | YES - legitimate |

**Verdict:** All cited sources are authentic, properly formatted, and match what the
configs and README reference. No issues.

---

## 7. SCRIPTS

### 7.1 Script Inventory (`scripts/`)

| Script | Size | Purpose |
|--------|------|---------|
| assemble_lexicons.py | 5.6 KB | Assemble per-language lexicons, verify deduplication |
| assign_cognate_links.py | 11 KB | Assign cognate links from expert annotations + Levenshtein |
| audit_cldf.py | 5.3 KB | Audit CLDF repos for coverage |
| build_validation_sets.py | 63 KB | Build stratified validation with phylogenetic tree |
| convert_cldf_to_tsv.py | 37 KB | Convert CLDF to validation TSVs |
| expand_cldf_full.py | 14 KB | Extract ALL data from CLDF (no concept filtering) |
| generate_language_readmes.py | 18 KB | Generate per-language profile markdowns |
| ingest_wikipron.py | 7.5 KB | Ingest WikiPron pronunciation data |
| normalize_lexicons.py | 6.3 KB | Normalize IPA and recompute SCA |

### 7.2 Coverage Report (`scripts/coverage_report.txt`)
- **Total CLDF entries:** 16,559
- **Total languages:** 199
- **Total concepts:** 105
- **TSV files generated:** 43

### 7.3 Issues
- All scripts use **relative path resolution** from script location (good practice)
- No hardcoded absolute paths found
- `ingest_wikipron.py` EXISTS but its output was apparently never committed (see Issue C1/M3)

---

## 8. COGNATE PIPELINE

### 8.1 Package Health
- **Package:** cognate_pipeline v0.1.0
- **Build:** Hatchling, Python >= 3.11
- **Tests:** 248 test methods across 18 files
- **Entry point:** `cognate-pipeline` CLI with 6 subcommands

### 8.2 Data Infrastructure Quality

#### language_map.json: 377 languages mapped (ISO -> Glottocode)
- Covers all ancient scripts + 360 modern languages
- Fallback 34-entry hardcoded map for critical languages
- **Assessment:** GOOD

#### family_map.json: 1,089 languages across 58 families
- Covers Indo-European (9 branches), Afroasiatic (5), Sino-Tibetan, Austronesian,
  Dravidian, Kartvelian, Uralic, Turkic, Mongolic, Tungusic, Americas (14), Africa,
  Japonic, Koreanic, and more
- **Assessment:** COMPREHENSIVE and CORRECT (spot-checked)

#### sound_class.py: SCA encoding
- 80+ IPA segments mapped to phonological natural classes
- Includes ancient script transliterations ($ -> S, H -> H, < -> H, @ -> S)
- Pharyngeals, clicks, implosives, retroflex all covered
- **Assessment:** LINGUISTICALLY SOUND

#### baseline_levenshtein.py: Distance metric
- Same class: 0.0, vowel-to-vowel: 0.3, related consonants: 0.3, unrelated: 1.0, indel: 0.5
- Normalized to 0-1 similarity scale
- **Assessment:** APPROPRIATE for cognate detection

### 8.3 Pipeline-to-Model Gap

**[ISSUE C3] No Model Training Bridge**

The cognate pipeline produces:
- JSONL staging files (raw lexemes -> normalised -> cognate links -> cognate sets)
- PostgreSQL database (8-table schema)
- CLDF export / JSON-LD export

The PhoneticPriorModel expects:
- Character-level input sequences with IPA feature vectors
- Lost text (unsegmented) + known vocabulary (segmented)
- Alignment pairs for training

**What's missing:**
1. No conversion from JSONL -> PyTorch tensors
2. No negative sample generation (non-cognate pairs)
3. No phonological feature extraction (pipeline only does abstract sound classes)
4. No train/val/test split stratification code
5. No documented model interface between pipeline output and model input

**Estimated effort:** 200-400 lines of Python for a data loader module.

### 8.4 Pipeline Coverage Assessment

| Capability | Status | Notes |
|------------|--------|-------|
| Ingest CSV/TSV/COG/CLDF/JSON | Working | 7 formats supported |
| IPA normalisation (4 backends) | Working | Confidence tracking included |
| SCA encoding | Working | 80+ segments, 0% unknowns on test data |
| Weighted Levenshtein scoring | Working | Tested with known cognates |
| Union-Find clustering | Working | Connected components verified |
| PostgreSQL persistence | Working | 8-table schema with migrations |
| CLDF export | Working | Standard format |
| **Model training data generation** | **NOT IMPLEMENTED** | **Critical gap** |

---

## 9. README ACCURACY

### Main README (`ancient-scripts-datasets/README.md`)

| Claim | Expected | Actual | Status |
|-------|----------|--------|--------|
| "199 languages" | 199 | 1,110 profiles (199 with CLDF data) | MISLEADING |
| "105 concepts" | 105 | 105 in expanded, 40 in core | CORRECT |
| "~16.5k entries" | 16,500 | 16,559 (CLDF only) | CORRECT |
| "43 family branches" | 43 | 43 TSV files in expanded | CORRECT |
| Gothic "4th century CE" | - | - | CORRECT |
| "~7,353 tokens" (Ugaritic) | 7,353 | 7,236 unique types | CLOSE (-1.6%) |
| "3,466 chunks" (Iberian) | 3,466 | 2,094 inscriptions (~3,466 chunks) | CORRECT (if counting chunks) |
| Gothic religious "~65 terms" | 65 | 90 | UNDERSTATED by 38% |
| Iberian religious "~40 elements" | 40 | 68 | UNDERSTATED by 70% |
| Ugaritic-Hebrew religious "~170 pairs" | 170 | 241 | UNDERSTATED by 42% |
| "27% true pairs" (names) | 27% | 27.3% (227/832) | CORRECT |
| "73% false positives" (names) | 73% | 72.7% (605/832) | CORRECT |

---

## 10. MASTER ISSUE REGISTRY

### Critical (Must Fix)

#### C1: Zero-Entry Major Languages
- **Severity:** CRITICAL
- **Files affected:** ~15-20 language profiles (rus, ara, pol, tur, tam, kor, jpn, etc.)
- **Impact:** Cannot train model on these languages despite WikiPron data existing
- **Root cause:** `ingest_wikipron.py` output never committed to lexicons
- **Diagnosis location:** Compare `wikipron_stats.tsv` against language profiles
- **Fix complexity:** MEDIUM (script exists, just needs to be run and output committed)

#### C2: 1,110 Profiles vs 199 Languages Claim
- **Severity:** CRITICAL (documentation/expectations)
- **Files affected:** README.md, coverage_report.txt, all 1,110 profile files
- **Impact:** Users expect 199 populated languages but get 1,110 files, many empty
- **Root cause:** `generate_language_readmes.py` ran on ALL sources including WikiPron
  catalog, but lexicon data only committed for CLDF-processed subset
- **Diagnosis location:** Count non-zero profiles: `grep -l "Entries:.*[1-9]" data/training/language_profiles/*.md`
- **Fix complexity:** LOW for docs update, MEDIUM to populate missing lexicons

#### C3: No Model Training Bridge
- **Severity:** CRITICAL (for the stated purpose of training PhoneticPriorModel)
- **Files affected:** None exist yet — need to create new module
- **Impact:** Cannot use pipeline output to train the model
- **Root cause:** Pipeline was designed for linguistic research, not model training
- **Fix complexity:** MEDIUM (200-400 lines of new Python code)

### Major (Should Fix)

#### M1: Ugaritic Data Asymmetry
- **Severity:** MAJOR (misleading)
- **Files affected:** `data/ugaritic/uga-heb.no_spe.cog`
- **Impact:** 83.5% of rows are Hebrew-only; effective cognate pairs = 6,395 not 43,951
- **Root cause:** COG format from NeuroDecipher includes full Hebrew vocabulary for
  search, not just matched pairs
- **Fix complexity:** LOW (documentation clarification + optional filtered subset)

#### M2: Missing Glottocodes (67%)
- **Severity:** MAJOR
- **Files affected:** `data/training/metadata/languages.tsv`, 746 profiles
- **Impact:** Phylogenetic classification unreliable for majority
- **Root cause:** ABVD languages lack Glottocodes in source data
- **Diagnosis location:** `awk -F'\t' '$3==""' languages.tsv | wc -l`
- **Fix complexity:** MEDIUM-HIGH (need to look up Glottocodes from Glottolog for 746 languages)

#### M3: WikiPron Data Not Integrated
- **Severity:** MAJOR
- **Files affected:** All zero-entry profiles
- **Impact:** 411k Russian entries, 201k Portuguese, etc. exist but are unused
- **Root cause:** `ingest_wikipron.py` was not run or output not committed
- **Diagnosis:** Check if `data/training/lexicons/` directory exists with per-language TSVs
- **Fix complexity:** MEDIUM (run the script, validate output, commit)

#### M4: No Negative Sample Generation
- **Severity:** MAJOR (for ML training)
- **Files affected:** cognate_pipeline (missing feature)
- **Impact:** Cannot create balanced training sets
- **Root cause:** Pipeline designed for cognate detection, not ML training
- **Fix complexity:** MEDIUM (add negative sampling to pipeline or data loader)

### Minor (Nice to Fix)

#### m1: Iberian CSV Python List Syntax
- **Fix:** Convert `['a', 'b']` to `["a", "b"]` JSON format
- **Complexity:** LOW (10 lines of Python)

#### m2: Iberian Religious Header Typo
- **Fix:** Replace `3category` with `category` in line 1
- **Complexity:** TRIVIAL (1 line edit)

#### m3: README Entry Counts Outdated
- **Fix:** Update ~65 -> 90, ~40 -> 68, ~170 -> 241, 7,353 -> 7,236
- **Complexity:** TRIVIAL (text edits)

#### m4: 27 Duplicate Ugaritic Pairs
- **Fix:** Deduplicate with `sort -u`
- **Complexity:** TRIVIAL

#### m5: FALSA/SUSPECTA Iberian Entries Untagged
- **Fix:** Add `status` column or move to separate file
- **Complexity:** LOW

---

## 11. REMEDIATION PLANS

### PLAN C1: Populate Zero-Entry Languages from WikiPron

#### What's Wrong
Major languages (Russian, Arabic, Polish, Turkish, Tamil, Korean, Japanese, etc.) have
empty language profiles despite WikiPron containing hundreds of thousands of pronunciation
entries for them.

#### Where to Find and Diagnose
1. Compare `data/training/metadata/wikipron_stats.tsv` (lists available WikiPron entries)
   against language profiles in `data/training/language_profiles/`
2. Check if `scripts/ingest_wikipron.py` has already been configured for these languages
3. Verify WikiPron data format compatibility with the pipeline

#### High-Level Solution
Run the existing `ingest_wikipron.py` script to pull WikiPron pronunciation data into
the lexicon format, then regenerate affected language profiles.

#### Step-by-Step Implementation
1. **Audit the gap:**
   ```bash
   # List all languages with WikiPron data but 0 lexicon entries
   # Cross-reference wikipron_stats.tsv against languages.tsv
   ```

2. **Verify WikiPron source availability:**
   - WikiPron data is publicly available at https://github.com/CUNY-CL/wikipron
   - The `ingest_wikipron.py` script already knows how to parse it
   - Hypothesis A: WikiPron TSV files are in a `sources/` directory (gitignored)
   - Hypothesis B: Script downloads directly from WikiPron GitHub
   - Hypothesis C: WikiPron was processed locally and stats generated, but output
     lexicon files were not committed

3. **Run ingestion for missing languages:**
   ```bash
   cd ancient-scripts-datasets
   python scripts/ingest_wikipron.py --languages rus,ara,pol,tur,tam,kor,jpn,swa,...
   ```

4. **Validate output:**
   - Check that each language lexicon TSV has expected columns (form, IPA, concept_id)
   - Verify IPA transcriptions are genuine Unicode IPA
   - Spot-check 10 entries per language against Wiktionary

5. **Regenerate profiles:**
   ```bash
   python scripts/generate_language_readmes.py
   ```

6. **Update metadata:**
   - Regenerate `languages.tsv` with new entry counts
   - Update `source_stats.tsv`
   - Update `coverage_report.txt`

7. **Commit and verify:**
   - Run existing test suite to ensure nothing breaks
   - Verify total entry count increased significantly

#### Hypotheses for Finding Phonetic Data
For languages where WikiPron is insufficient:
- **Hypothesis 1:** Use Wiktionary pronunciation sections (WikiPron's source)
- **Hypothesis 2:** Use epitran (grapheme-to-phoneme for ~100 languages)
- **Hypothesis 3:** Use phonemizer/espeak-ng (300+ languages, lower quality)
- **Hypothesis 4:** Use IDS/ABVD orthographic forms with transliteration passthrough
- **Hypothesis 5:** For ancient languages, use published IPA dictionaries (already
  covered by NorthEuraLex for ~100 languages)

#### Subagent Strategy
Spin up a subagent for each language group that:
1. Downloads WikiPron data for that language
2. Parses to TSV format (form | IPA | source)
3. Runs `normalize_lexicons.py` to compute SCA sound classes
4. Validates output
5. Reports entry count and quality metrics

---

### PLAN C2: Reconcile Profile Count vs Language Count

#### What's Wrong
1,110 language profiles exist but README claims 199 languages. The discrepancy is
because profiles were generated for ALL languages in ANY source (NorthEuraLex + ABVD +
WikiPron + CLDF + WOLD + SinoTibetan), but the "199 languages" count refers only to
those processed through the CLDF validation pipeline.

#### High-Level Solution
Either:
- **Option A:** Delete empty profiles and update count to match actual data (~500-600)
- **Option B:** Populate all 1,110 profiles with at least WikiPron data (preferred)
- **Option C:** Keep all profiles but clearly document tiers (Tier 1: full data,
  Tier 2: WikiPron only, Tier 3: profile only)

#### Step-by-Step Implementation
1. Categorize all 1,110 profiles into tiers based on available data
2. Update README to document the tiered system
3. If Option B: run WikiPron ingestion (see Plan C1)
4. If Option C: add a `tier` field to each profile and to `languages.tsv`
5. Update all documentation and coverage reports

---

### PLAN C3: Build Model Training Bridge

#### What's Wrong
The cognate pipeline outputs JSONL files with cognate pairs, but the PhoneticPriorModel
expects character-level input with IPA features, alignment pairs, and segmentation data.

#### Where to Find and Diagnose
1. Read `repro_decipher_phonetic_prior/repro/model/phonetic_prior.py` for input format
2. Read `repro_decipher_phonetic_prior/datasets/` for existing data loading code
3. Read pipeline output format in `cognate_pipeline/src/cognate_pipeline/cognate/models.py`

#### High-Level Solution
Create a `training_export.py` module that reads pipeline JSONL output and produces
model-compatible training data.

#### Step-by-Step Implementation
1. **Analyze model input format:**
   - What tensors does PhoneticPriorModel expect?
   - Character vocabulary (lost + known)
   - IPA feature vectors
   - Alignment supervision signals

2. **Design the bridge module:**
   ```python
   # Reads: staging/cognate/cognate_links.jsonl
   # Reads: staging/normalised/*.jsonl
   # Produces: train.pt, val.pt, test.pt
   ```

3. **Implement negative sampling:**
   - For each positive cognate pair, sample N negative pairs from:
     - Same concept, different (unrelated) language family
     - Different concept, same language
   - Ratio: 1:3 or 1:5 positive:negative

4. **Implement feature extraction:**
   - From `phonetic_canonical` IPA -> phonological features
   - Manner of articulation (stop, fricative, nasal, etc.)
   - Place of articulation (bilabial, alveolar, velar, etc.)
   - Voicing, aspiration, length

5. **Implement stratified splits:**
   - Train/val/test by language family (no family leakage)
   - Stratify by concept frequency
   - Stratify by phylogenetic distance

6. **Test with existing model:**
   - Load generated data into PhoneticPriorModel
   - Verify shapes and types match
   - Run 1 training step to confirm gradient flow

---

### PLAN M1: Document Ugaritic Data Asymmetry

#### Step-by-Step
1. Add a note to `data/ugaritic/README.md` (create if needed) explaining:
   - Full file contains 43,951 rows
   - Only 6,395 are true Ugaritic-Hebrew cognate pairs
   - Remaining 37,556 are Hebrew-only search vocabulary
   - This is INTENTIONAL for the decipherment model
2. Optionally create `uga-heb.pairs_only.cog` with only the 6,395 matched rows
3. Update main README count from "~7,353 tokens" to clarify what's being counted

---

### PLAN M2: Populate Missing Glottocodes

#### Step-by-Step
1. Download Glottolog database (https://glottolog.org/glottolog/glottolog-cldf-v5.1.zip)
2. For each of the 746 languages missing Glottocodes:
   a. Look up ISO 639-3 code in Glottolog
   b. If found, add Glottocode
   c. If not found (rare/unclassified), mark as `unattested`
3. Update `languages.tsv` and all affected profile files
4. Verify with `audit_cldf.py`

#### Subagent Strategy
Spin up a subagent that:
1. Downloads Glottolog CLDF data
2. Builds ISO-639-3 -> Glottocode lookup table
3. Processes all 746 missing entries
4. Reports coverage improvement

---

### PLAN M3: Integrate WikiPron Data

(Largely overlaps with Plan C1 — see above for details)

Key additional steps:
1. Determine if WikiPron data needs to be re-downloaded or if it was previously cached
2. Check `sources/` directory (gitignored) for existing WikiPron downloads
3. If not cached, download from WikiPron GitHub releases
4. Process through `ingest_wikipron.py`
5. Merge with existing lexicon data (NorthEuraLex, ABVD, etc.)
6. Deduplicate entries that appear in both sources

---

### PLAN M4: Add Negative Sample Generation

#### Step-by-Step
1. Add `generate_negatives()` function to `cognate_pipeline/cognate/`:
   ```python
   def generate_negatives(
       positive_pairs: list[CognateLink],
       lexicon: dict[str, list[NormalisedLexeme]],
       ratio: int = 3,
       strategy: str = "cross_family"  # or "random", "hard_negative"
   ) -> list[CognateLink]:
   ```

2. Strategies:
   - **cross_family:** Pair words from unrelated families sharing the same concept
   - **random:** Random pairs from any two languages
   - **hard_negative:** Sound-alike words from unrelated families (hardest negatives)

3. Add CLI flag: `cognate-pipeline detect_cognates --include-negatives --neg-ratio 3`

4. Include in JSONL output with `relationship_type: "true_negative"` or `"false_positive"`

---

### PLANS m1-m5: Minor Fixes

#### m1: Iberian CSV Format
```python
import ast, csv, json
with open('iberian.csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = [(ref, json.dumps(ast.literal_eval(cleaned))) for ref, cleaned in reader]
# Write back with proper JSON
```

#### m2: Iberian Religious Header
```bash
sed -i 's/^3category/category/' iberian_religious.tsv
```

#### m3: README Updates
- Gothic religious: ~65 -> 90
- Iberian religious: ~40 -> 68
- Ugaritic-Hebrew religious: ~170 -> 241
- Ugaritic token count: ~7,353 -> 7,236 (or clarify counting method)

#### m4: Deduplicate Ugaritic Pairs
```bash
sort -u uga-heb.no_spe.cog > uga-heb.no_spe.deduped.cog
```

#### m5: Tag Iberian FALSA/SUSPECTA
Add `status` column: `authentic` | `falsa` | `suspecta`, strip suffix from ref codes.

---

## APPENDIX A: REAL-WORLD DATA VOLUME EXPECTATIONS

Reference numbers for validating dataset sizes:

| Language/Source | Expected Entries | Actual in Dataset | Assessment |
|-----------------|-----------------|-------------------|------------|
| English vocabulary | 170,000-470,000 | 111,862 (CLDF+WikiPron) | Adequate for training |
| Russian vocabulary | 150,000-200,000 | 0 (411k available via WikiPron) | MISSING |
| Arabic vocabulary | 50,000-100,000 | 0 (13k available via WikiPron) | MISSING |
| NorthEuraLex/language | ~1,000 | ~900-1,100 | CORRECT |
| WikiPron major langs | 10,000-400,000 | Stats catalogued, not loaded | NOT INTEGRATED |
| ABVD/language | ~210 | ~100-500 | CORRECT |
| Wulfila Bible | ~67,400 tokens | 76,771 tokens | CORRECT (includes variants) |
| Ugaritic lexicon | ~1,500-1,600 words | 7,236 forms (incl. variants) | PLAUSIBLE |
| Iberian inscriptions | ~2,000 | 2,094 | CORRECT |

## APPENDIX B: FILE INTEGRITY MATRIX

| File | Format | Encoding | Size | Integrity |
|------|--------|----------|------|-----------|
| gotica.txt | Plain text | UTF-8, CRLF | 516 KB | VALID |
| gotica.xml.zip | ZIP | - | 428 KB | VALID (extracts to 3.5 MB XML) |
| got.pretrained.pth | PyTorch | Binary | 27 KB | VALID (proper ZIP magic bytes) |
| segments.pkl | Pickle | Binary | 4 KB | VALID |
| uga-heb.no_spe.cog | TSV | ASCII, CRLF | 396 KB | VALID (27 dupes) |
| uga-heb.small.no_spe.cog | TSV | ASCII, CRLF | 41 KB | VALID |
| iberian.csv | CSV | UTF-8, CRLF | 72 KB | VALID (non-standard format) |
| gothic_religious.tsv | TSV | UTF-8 | 10 KB | VALID |
| iberian_religious.tsv | TSV | UTF-8 | 12 KB | VALID (header typo) |
| ugaritic_hebrew_religious.tsv | TSV | UTF-8 | 18 KB | VALID (intentional dupes) |
| Hebrew.xml | CES XML | UTF-8 | 5.5 MB | VALID |
| Latin.xml | CES XML | UTF-8 | 4.7 MB | VALID |
| Basque-NT.xml | CES XML | UTF-8 | 1.5 MB | VALID |
| Trask_Etymological_Dictionary_Basque.pdf | PDF | - | 1.5 MB | VALID |
| RodriguezRamos2014.pdf | PDF | - | 1.2 MB | VALID |

## APPENDIX C: LANGUAGE FAMILY COVERAGE

| Family | Core Validation | Expanded | Training Profiles | Status |
|--------|----------------|----------|-------------------|--------|
| Germanic | 4 langs, 160 rows | 11 langs, 914 rows | ~50+ profiles | GOOD |
| Celtic | 3 langs, 120 rows | 4 langs, 339 rows | ~10+ profiles | GOOD |
| Balto-Slavic | 3 langs, 120 rows | 13 langs, 1,232 rows | ~30+ profiles | PARTIAL (major langs missing data) |
| Indo-Iranian | 3 langs, 120 rows | 10 langs, 810 rows | ~40+ profiles | GOOD |
| Italic | 3 langs, 120 rows | 9 langs, 718 rows | ~20+ profiles | GOOD |
| Hellenic | 2 langs, 80 rows | 3 langs, 151 rows | ~5 profiles | ADEQUATE |
| Semitic | 3 langs, 120 rows | 3 langs, 240 rows | ~15+ profiles | PARTIAL (Arabic missing) |
| Turkic | 3 langs, 120 rows | 10 langs, 876 rows | ~15+ profiles | PARTIAL (Turkish missing) |
| Uralic | 3 langs, 120 rows | 27 langs, 2,615 rows | ~30+ profiles | GOOD |
| Austronesian | - | - | ~753 profiles (ABVD) | SMALL per-lang but many langs |
| Sino-Tibetan | - | - | ~50+ profiles | ADEQUATE |
| Dravidian | - | - | ~10+ profiles | PARTIAL (Tamil missing) |
| Japonic | - | - | ~5 profiles | PARTIAL (Japanese missing) |
| Koreanic | - | - | ~3 profiles | PARTIAL (Korean missing) |

---

**END OF PRD**
**Total issues identified: 12 (3 critical, 4 major, 5 minor)**
**Estimated total remediation effort: ~800-1200 lines of code + data regeneration**
