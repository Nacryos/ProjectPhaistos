# Ancient Scripts Decipherment Datasets

Collated datasets for the paper:

> **Deciphering Undersegmented Ancient Scripts Using Phonetic Prior**
> Jiaming Luo, Frederik Hartmann, Enrico Santus, Regina Barzilay, Yuan Cao
> *Transactions of the Association for Computational Linguistics*, 2021
> [arXiv:2010.11054](https://arxiv.org/abs/2010.11054)

This repository gathers the training datasets used in the paper — both those hosted in the authors' GitHub repos and the external cited sources.

---

## Repository Structure

```
data/
├── gothic/                          # Gothic language data
│   ├── got.pretrained.pth           # Pretrained phonological embeddings (PyTorch)
│   ├── segments.pkl                 # Phonetic segment data (Python pickle)
│   ├── gotica.txt                   # Gothic Bible plain text (Wulfila project)
│   └── gotica.xml.zip               # Gothic Bible TEI XML (Wulfila project)
│
├── ugaritic/                        # Ugaritic-Hebrew cognate data
│   ├── uga-heb.no_spe.cog          # Full cognate pairs (TSV, ~7,353 tokens)
│   └── uga-heb.small.no_spe.cog    # Small training subset (~10% of full)
│
├── iberian/                         # Iberian inscription data
│   └── iberian.csv                  # Cleaned Hesperia epigraphy (3,466 chunks)
│
├── religious_terms/                 # ** CURATED SUBSET: Religious vocabulary **
│   ├── README.md                    # Methodology and category definitions
│   ├── ugaritic_hebrew_religious.tsv  # ~170 Ug-Heb cognate pairs (deity, ritual, sacred)
│   ├── gothic_religious.tsv         # ~65 Gothic Bible religious terms
│   └── iberian_religious.tsv        # ~40 Iberian votive/religious elements
│
├── validation/                      # Phylogenetic validation dataset (9 branches)
│   ├── README.md                    # Format, sources, concept list
│   ├── concepts.tsv                 # 40 shared concept IDs
│   ├── germanic.tsv                 # got, ang, non, goh (~160 entries)
│   ├── celtic.tsv                   # sga, cym, bre (~120 entries)
│   ├── balto_slavic.tsv             # lit, chu, rus (~120 entries)
│   ├── indo_iranian.tsv             # san, ave, fas (~120 entries)
│   ├── italic.tsv                   # lat, osc, xum (~120 entries)
│   ├── hellenic.tsv                 # grc, gmy (~80 entries)
│   ├── semitic.tsv                  # heb, arb, amh (~120 entries)
│   ├── turkic.tsv                   # otk, tur, aze (~120 entries)
│   └── uralic.tsv                   # fin, hun, est (~120 entries)
│
└── cited_sources/                   # External datasets cited in the paper
    ├── genesis/
    │   ├── Hebrew.xml               # Hebrew Bible (Christodouloupoulos & Steedman 2015)
    │   └── Latin.xml                # Latin Bible (same corpus)
    ├── basque/
    │   ├── Basque-NT.xml            # Basque New Testament (same corpus)
    │   └── Trask_Etymological_Dictionary_Basque.pdf  # Trask's Basque etymological dictionary
    └── iberian_names/
        └── RodriguezRamos2014.pdf   # Iberian onomastic index (personal names)
```

---

## Dataset Details

### Gothic (`data/gothic/`)

| File | Source | Description |
|---|---|---|
| `got.pretrained.pth` | [DecipherUnsegmented](https://github.com/j-luo93/DecipherUnsegmented) | Pretrained phonological embeddings trained on Gothic IPA data |
| `segments.pkl` | [DecipherUnsegmented](https://github.com/j-luo93/DecipherUnsegmented) | Serialized phonetic segment inventory |
| `gotica.txt` | [Wulfila Project](https://www.wulfila.be/gothic/download/) | Plain text of the Gothic Bible (4th century CE translation by Bishop Wulfila) |
| `gotica.xml.zip` | [Wulfila Project](https://www.wulfila.be/gothic/download/) | TEI P5 XML encoding with linguistic annotations |

The Gothic Bible is the primary source of Gothic text. The paper uses unsegmented Gothic inscriptions from the 3rd-10th century AD period.

### Ugaritic (`data/ugaritic/`)

| File | Source | Description |
|---|---|---|
| `uga-heb.no_spe.cog` | [NeuroDecipher](https://github.com/j-luo93/NeuroDecipher) | Full Ugaritic-Hebrew cognate pairs |
| `uga-heb.small.no_spe.cog` | [NeuroDecipher](https://github.com/j-luo93/NeuroDecipher) | ~10% training subset |

**Format:** Tab-separated values. Each row is a cognate pair. Column 1 = Ugaritic transliteration, Column 2 = Hebrew transliteration. `|` separates multiple cognates; `_` marks missing entries. Originally from Snyder et al. (2010), covering 7,353 segmented tokens from the 14th-12th century BC.

### Iberian (`data/iberian/`)

| File | Source | Description |
|---|---|---|
| `iberian.csv` | [DecipherUnsegmented](https://github.com/j-luo93/DecipherUnsegmented) | Cleaned epigraphic inscriptions |

**Format:** CSV with columns `REF. HESPERIA` (inscription reference code) and `cleaned` (transcribed text). Contains 3,466 undersegmented character chunks from the 6th-1st century BC. Sourced from the [Hesperia database](http://hesperia.ucm.es/en/proyecto_hesperia.php) and cleaned via the authors' Jupyter notebook.

### Cited Sources (`data/cited_sources/`)

These are external datasets referenced in the paper for known-language vocabularies and comparison:

| File | Citation | Usage in Paper |
|---|---|---|
| `genesis/Hebrew.xml` | Christodouloupoulos & Steedman (2015) | Hebrew vocabulary for Ugaritic comparison |
| `genesis/Latin.xml` | Christodouloupoulos & Steedman (2015) | Latin vocabulary for cross-linguistic comparison |
| `basque/Basque-NT.xml` | Christodouloupoulos & Steedman (2015) | Basque vocabulary for Iberian comparison |
| `basque/Trask_Etymological_Dictionary_Basque.pdf` | Trask (2008) | Basque etymological data |
| `iberian_names/RodriguezRamos2014.pdf` | Rodriguez Ramos (2014) | Iberian personal name lists with Latin correspondences |

The Bible texts are from the [Massively Parallel Bible Corpus](https://github.com/christos-c/bible-corpus) (CC0 licensed).

---

## Additional Data Sources (Not Included)

The following sources were cited in the paper but are not machine-readable or freely downloadable:

- **Wiktionary descendant trees** for Proto-Germanic, Old Norse, and Old English vocabularies — extracted by the authors from Wiktionary's structured data
- **Original Hesperia epigraphy** (`hesperia_epigraphy.csv`) — referenced in the DecipherUnsegmented README but not present in the repository

---

## Cognate Detection Pipeline

The `cognate_pipeline/` directory contains a full Python package for cross-linguistic cognate detection, built on the datasets in this repository. It provides:

- **Ingestion** of CSV/TSV/COG, CLDF, Wiktionary JSONL, and generic JSON sources
- **Phonetic normalisation** with transcription type tracking (IPA, transliteration, orthographic)
- **SCA sound class encoding** (List 2012) for phonological comparison
- **Family-aware cognate candidate generation** (tags `cognate_inherited` vs `similarity_only`)
- **Weighted Levenshtein scoring** with SCA-class-aware substitution costs
- **Clustering** via connected components or UPGMA
- **PostgreSQL/PostGIS database** with 8 normalised tables and Alembic migrations
- **Export** to CLDF Wordlist and JSON-LD formats
- **Full provenance tracking** through every pipeline stage

Supports 36 languages across 9 phylogenetic branches (Germanic, Celtic, Balto-Slavic, Indo-Iranian, Italic, Hellenic, Semitic, Turkic, Uralic) plus isolates, with Glottocode resolution and IPA transcriptions.

See `data/validation/README.md` for the phylogenetic validation dataset.

See [`cognate_pipeline/README.md`](cognate_pipeline/README.md) for installation and usage.

---

## Original Repositories

- [j-luo93/DecipherUnsegmented](https://github.com/j-luo93/DecipherUnsegmented) — main code for the paper
- [j-luo93/NeuroDecipher](https://github.com/j-luo93/NeuroDecipher) — predecessor (Ugaritic/Linear B decipherment)
- [j-luo93/xib](https://github.com/j-luo93/xib) — earlier Iberian codebase

## Paper Citation

```bibtex
@article{luo2021deciphering,
  title={Deciphering Undersegmented Ancient Scripts Using Phonetic Prior},
  author={Luo, Jiaming and Hartmann, Frederik and Santus, Enrico and Barzilay, Regina and Cao, Yuan},
  journal={Transactions of the Association for Computational Linguistics},
  volume={9},
  pages={69--81},
  year={2021},
  doi={10.1162/tacl_a_00354}
}
```
