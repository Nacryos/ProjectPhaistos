# Reproduction: Deciphering Undersegmented Ancient Scripts Using Phonetic Prior

This directory contains a reproducible scaffold for the TACL 2021 paper:

- Paper PDF used locally: `/Users/aaronbao/Downloads/DecipherUnsegmented (3).pdf`
- Upstream paper repo: `third_party/DecipherUnsegmented`
- Baseline repo: `third_party/NeuroDecipher`
- Dataset repo (including religious subset): `third_party/ancient-scripts-datasets`

Pinned repository commits are recorded in `third_party/LOCK.json`.

## Folder layout

```text
repro_decipher_phonetic_prior/
├── README.md
├── Makefile
├── environment.yml
├── requirements.txt
├── artifacts/
├── data/
│   └── prepared/
├── data_external/
├── datasets/
│   ├── __init__.py
│   ├── filters.py
│   └── registry.py
├── repro/
│   ├── __init__.py
│   ├── paths.py
│   ├── utils.py
│   ├── third_party.py
│   ├── report.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── prepare.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── smoke.py
│   │   ├── gothic.py
│   │   ├── ugaritic.py
│   │   ├── iberian.py
│   │   └── validation.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── phonetic_prior.py
│   ├── reference/
│   │   ├── __init__.py
│   │   └── paper_metrics.py
│   └── wrappers/
│       ├── __init__.py
│       ├── neurodecipher_wrapper.py
│       └── xib_wrapper.py
├── scripts/
│   ├── setup.sh
│   ├── fetch_data.sh
│   ├── prepare_datasets.sh
│   ├── smoke_test.sh
│   ├── train_gothic.sh
│   ├── eval_gothic.sh
│   ├── run_neurocipher.sh
│   ├── fig4.sh
│   ├── table2.sh
│   ├── table3.sh
│   ├── table4.sh
│   ├── compare_graphs.sh
│   ├── run_validation_pair.sh
│   └── reproduce_all.sh
├── patches/
└── third_party/
    ├── LOCK.json
    ├── DecipherUnsegmented/
    ├── NeuroDecipher/
    ├── ancient-scripts-datasets/
    ├── xib/
    ├── dev_misc/
    └── arglib/
```

## Environment setup

### Conda

```bash
conda env create -f environment.yml
conda activate repro_decipher_phonetic_prior
bash scripts/setup.sh
```

### Pip only

```bash
python3 -m venv .venv
source .venv/bin/activate
bash scripts/setup.sh
```

## Data preparation

### Prepare all corpora + religious variants

```bash
bash scripts/prepare_datasets.sh
```

This writes deterministic prepared datasets under:

- `data/prepared/<corpus>/<variant>/<hash>/`

and provenance to:

- `artifacts/data_provenance.json`

### External appendix assets

```bash
bash scripts/fetch_data.sh
```

`fetch_data.sh` caches downloads in `data_external/` and writes SHA256 checksums to `data_external/checksums.sha256`.
For non-machine-readable appendix assets (Wiktionary descendant-tree extracts, Iberian personal-name TSV), the script fails with explicit required filenames unless `ALLOW_INCOMPLETE=1` is set.

## Single-command pipeline

```bash
bash scripts/reproduce_all.sh
```

Default behavior:

- Runs deterministic data prep.
- Runs a smoke test of DP code paths.
- Runs Gothic/Ugaritic/Iberian experiment wrappers.
- Generates CSV tables and Figure 4 panel images.
- Writes `artifacts/REPRO_SUMMARY.md`.

## Make targets

```bash
make setup
make data
make smoke
make train_gothic
make eval_gothic
make fig4
make table2
make table3
make table4
make compare_graphs
make run_validation
make reproduce_all
```

## Updated languages from `ancient-scripts-datasets`

The vendored dataset repo includes a new phylogenetic validation suite under:

- `third_party/ancient-scripts-datasets/data/validation/*.tsv`

Supported validation branches:

- `validation_germanic`
- `validation_celtic`
- `validation_balto_slavic`
- `validation_indo_iranian`
- `validation_italic`
- `validation_hellenic`
- `validation_semitic`
- `validation_turkic`
- `validation_uralic`
- `validation_names`

Each branch exposes multiple language codes. The phylogenetic branches cover 23 languages; `validation_names` adds a cross-family names subset spanning 30 languages.

- `variant="lang:<lost_lang>"` (lost language vs all others)
- `variant="pair:<lost_lang>:<known_lang>"` (explicit pair run)

## Paper component mapping to code modules

Algorithm 1 mapping:

- `ComputeCharDistr` -> `repro/model/phonetic_prior.py:ComputeCharDistr`
- `EditDistDP` -> `repro/model/phonetic_prior.py:EditDistDP`
- `WordBoundaryDP` -> `repro/model/phonetic_prior.py:WordBoundaryDP`
- SGD step -> `repro/model/phonetic_prior.py:train_one_step`

Model pieces from paper:

- Eq.3 softmax temperature mapping -> `PhoneticPriorModel.compute_char_distr`
- Eq.4 lost embedding composition from known IPA features -> `PhoneticPriorModel.compute_char_distr`
- Monotonic edit-distance DP with insertion/deletion/substitution and adjacent-index insertion -> `PhoneticPriorModel.edit_distance_dp`
- Objective with quality, coverage, sound-loss regularization -> `PhoneticPriorModel.objective`

Upstream wrappers (minimal upstream edits):

- NeuroCipher baseline execution -> `repro/wrappers/neurodecipher_wrapper.py`
- xib extraction wrapper -> `repro/wrappers/xib_wrapper.py`

## Dataset registry API

Implemented in `datasets/registry.py`:

- `list_corpora() -> List[str]`
- `get_corpus(name: str, variant: Optional[str] = None) -> Corpus`
- `list_validation_languages(corpus_name: str) -> List[str]`
- `export_for_decipherunsegmented(corpus, out_dir, config) -> Dict[str, str]`

`Corpus` exposes:

- `lost_text`
- `known_text`
- `metadata`
- `splits`
- `ground_truth` (when available)

Religious subset variants are available but not used by default targets:

- `get_corpus("gothic", variant="religious")`
- `get_corpus("ugaritic", variant="religious")`
- `get_corpus("iberian", variant="religious")`

Validation examples:

- `get_corpus("validation_germanic", variant="lang:got")`
- `get_corpus("validation_germanic", variant="pair:got:ang")`
- `get_corpus("validation_semitic", variant="pair:heb:arb")`
- `get_corpus("validation_names", variant="pair:lat:got")`

## Run updated language pairs through model

Run the phonetic-prior model on any validation pair:

```bash
BRANCH=germanic LOST=got KNOWN=ang STEPS=30 bash scripts/run_validation_pair.sh
```

or:

```bash
python3 -m repro.experiments.validation --branch semitic --lost heb --known arb --steps 30
```

Outputs are written to:

- `artifacts/runs/validation_<branch>_<lost>_to_<known>.json`

## Future: secondary filters

Placeholder filters exist in `datasets/filters.py`:

- `filter_by_religious_subset(corpus: Corpus) -> Corpus`
- `filter_by_metadata(corpus: Corpus, predicate) -> Corpus`

These intentionally raise `NotImplementedError` for now.

To use religious data in future runs, wire a config flag such as `use_religious_variant=true` into the experiment entrypoints and call `get_corpus(..., variant="religious")` during data preparation. `scripts/reproduce_all.sh` does not enable this flag.

## Generated outputs

After a successful run:

- `artifacts/table2.csv`
- `artifacts/table3.csv`
- `artifacts/table4.csv`
- `artifacts/figures/fig4a.png`
- `artifacts/figures/fig4b.png`
- `artifacts/figures/fig4c.png`
- `artifacts/figures/fig4d.png`
- `artifacts/REPRO_SUMMARY.md`
- `artifacts/data_provenance.json`
- `artifacts/runs/smoke_test.json`

## Runtime expectations

- Smoke test: ~1-3 minutes on CPU.
- Reference-mode table/figure generation: <5 minutes.
- Full run-based reproduction with all external assets and upstream training: multi-hour to day-scale CPU runtime (GPU optional).
