# Training Data

Phonetic lexicons and language profiles for cross-linguistic cognate detection and decipherment model training.

## Summary

| Metric | Value |
|--------|-------|
| **Languages** | 1,097 |
| **Total entries** | 3,204,193 |
| **Language families** | 56 |
| **With IPA transcriptions** | 369 languages (WikiPron, NorthEuraLex, WOLD) |
| **Orthographic only** | 728 languages (ABVD Austronesian) |
| **Glottocodes assigned** | 1,096 / 1,097 (only `uun` missing — retired ISO code) |

## Data Quality Tiers

| Tier | Entry Range | Languages | Description |
|------|-------------|-----------|-------------|
| **T1** | 50,000+ | 15 | Major world languages (rus, eng, zho, fin, por, etc.) |
| **T2** | 10,000–49,999 | 23 | Well-documented languages (deu, nld, kor, hin, etc.) |
| **T3** | 1,000–9,999 | 162 | Moderate coverage (most WikiPron languages) |
| **T4** | 100–999 | 876 | Basic vocabulary lists (mostly ABVD Austronesian) |
| **T5** | <100 | 21 | Minimal data (use with caution) |

## Sources

| Source | Languages | Entries | Transcription Type |
|--------|-----------|---------|-------------------|
| **WikiPron** | 307 | 2,823,641 | IPA (broad preferred) |
| **NorthEuraLex** | 107 | 121,611 | IPA segments |
| **WOLD** | 38 | 59,711 | IPA segments |
| **ABVD** | 751 | 278,774 | Orthographic (pseudo-IPA) |

Note: Many languages appear in multiple sources. The assembly script deduplicates by (word, IPA) pairs.

## Top Language Families

| Family | Languages | Total Entries |
|--------|-----------|---------------|
| Austronesian | 753 | 269,930 |
| Balto-Slavic | 20 | 811,232 |
| Germanic | 32 | 291,612 |
| Italic | 28 | 576,176 |
| Uralic | 29 | 265,775 |
| Sino-Tibetan | 14 | 276,290 |
| Indo-Iranian | 28 | 80,893 |
| Turkic | 17 | 57,396 (approx) |
| Semitic | 16 | 57,396 (approx) |
| Celtic | 9 | 44,891 |

## Directory Structure

```
training/
├── lexicons/              # Per-language TSV files (1,097 files)
│   ├── eng.tsv            # Word → IPA → SCA → Source → Concept_ID → Cognate_Set_ID
│   ├── rus.tsv
│   └── ...
├── language_profiles/     # Per-language markdown profiles (1,097 files)
│   ├── eng.md             # Overview, cognate pairs, religious domain, partners
│   └── ...
└── metadata/
    ├── languages.tsv      # Master language index (ISO, Family, Glottocode, Entries, Sources)
    ├── source_stats.tsv   # Per-language per-source entry counts
    ├── wikipron_stats.tsv # WikiPron-only statistics
    └── cldf_stats.tsv     # CLDF-only statistics
```

## Lexicon TSV Format

Each `lexicons/{iso}.tsv` file has these columns:

| Column | Description |
|--------|-------------|
| `Word` | Surface form (orthographic or romanized) |
| `IPA` | IPA transcription (or pseudo-IPA for orthographic sources) |
| `SCA` | SCA sound class encoding (List 2012) |
| `Source` | Data source (`wikipron`, `northeuralex`, `wold`, `abvd`) |
| `Concept_ID` | Concepticon gloss or concept identifier (`-` if unavailable) |
| `Cognate_Set_ID` | Cognate set identifier (`-` if unavailable) |

## Regeneration

To regenerate from sources:

```bash
# 1. Ingest WikiPron (requires sources/wikipron/)
python scripts/ingest_wikipron.py

# 2. Extract CLDF sources (requires sources/northeuralex/, wold/, abvd/)
python scripts/expand_cldf_full.py

# 3. Assemble and deduplicate all lexicons
python scripts/assemble_lexicons.py
```
