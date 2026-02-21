# Run Log — PhoneticPriorModel

Reverse-chronological notes, reports, and observations from model investigation, training runs, and data preparation. Each entry is timestamped for tracking improvements over time.

---

## 2026-02-20 — All Pre-Run Items Complete (commit d256fb0)

**Phase:** Pre-run fixes (final)
**Status:** All 10 items resolved — full pipeline ready for training runs
**Commit:** `d256fb0`

### Low-Priority Items Resolved

| # | Action | Status | Detail |
|---|---|---|---|
| 8 | Cognate annotation for validation branches | DONE | Pure-Python SCA-based scoring (cognate_scoring.py); auto-annotates ground truth with cognate_score |
| 9 | Results visualization dashboard | DONE | visualize.py with 5 plot types: char heatmaps, P@K curves, training loss, branch comparison, confusion tables |
| 10 | Clean ABVD/WOLD data quality | DONE | _strip_wold_slashes() removes morphological `/` from IPA fields |

### New Modules

**`datasets/cognate_scoring.py`** — Cognate confidence scoring:
- SCA (Sound-Class-based) encoding: maps IPA to 14 Dolgopolsky-style classes
- Weighted edit distance on SCA-encoded strings (similar sounds = lower cost)
- Combined score: 30% raw Levenshtein + 70% SCA distance
- `annotate_ground_truth()` adds cognate_score to each GT row
- Validation: hund~hound=0.94, wasser~water=0.90, pater~father=0.73, hund~table=0.14

**`repro/eval/visualize.py`** — Visualization dashboard:
- `save_char_distr()` persists P(known|lost) to npz after training
- `plot_char_heatmap()` — character distribution heatmap with IPA labels
- `plot_p_at_k()` — P@K comparison curves across variants
- `plot_training_loss()` — multi-panel training curves (objective, quality, omega_cov, omega_loss)
- `plot_branch_comparison()` — grouped bar chart across validation branches
- `plot_confusion_matrix()` — top-N predicted known chars per lost char
- `plot_all_for_experiment()` — auto-detects output type and generates all applicable plots
- CLI: `python -m repro.eval.visualize all outputs/gothic`
- Also wired as `python -m repro.run_experiment visualize outputs/gothic`

### Pipeline Integration

- `save_char_distr()` wired into all 4 experiment runners (gothic, ugaritic, iberian_names, validation)
- Validation corpus builder automatically annotates GT with cognate_score via cognate_scoring module
- `visualize` subcommand added to run_experiment.py CLI

---

## 2026-02-20 — Priority Fixes Complete: Data, Model, and Pipeline (commits 6bfe8c4, 07b33e7)

**Phase:** Pre-run fixes
**Status:** All 7 priority items resolved — model ready for training runs
**Commits:** `6bfe8c4` (data/model fixes), `07b33e7` (corpus variants + validation runner)

### Priority Items Resolved

| # | Action | Status | Commit |
|---|---|---|---|
| 1 | Fix data file paths in YAML configs | DONE | `6bfe8c4` |
| 2 | Fix `_` padding filtering | DONE | `6bfe8c4` |
| 3 | Fix normalize_token for transliteration chars | DONE | `6bfe8c4` |
| 4 | Wire religious/name subsets into experiment runner | DONE | `07b33e7` |
| 5 | Create per-branch validation configs | DONE | `07b33e7` |
| 6 | Replace pseudo-random IPA features | DONE | `419eef7` (prev) |
| 7 | Fix omega_cov gradient death | DONE | `6bfe8c4` |

### Data Files Created/Downloaded

- `data_external/rodriguez_ramos_2014_personal_names.tsv` — 62 Iberian queries, 64 Latin stems (Bronze of Ascoli + onomastic formants)
- `data_external/wiktionary_descendants_{pg,on,oe}.tsv` — 37 Gothic cognate pairs each (generated from germanic_expanded.tsv)
- `third_party/NeuroDecipher/data/uga-heb.{small.,}no_spe.cog` — downloaded from original repo
- `third_party/DecipherUnsegmented/data/{iberian.csv,got.pretrained.pth,segments.pkl}` — downloaded

All 30 required YAML paths now resolve correctly. 5 optional closeness vocab files (Arabic, Spanish, Hungarian, Turkish, Aramaic) skipped (`required: false`).

### Model Fixes (commit 6bfe8c4)

**omega_cov gradient fix** — `word_boundary_dp()` now returns differentiable soft coverage:
```python
p_unmatched = (c_o - dp[i]).exp().clamp(0.0, 1.0)
soft_match.append(1.0 - p_unmatched)
```
Before: coverage was plain float with zero gradient. After: differentiable tensor with `MeanBackward0` grad_fn.

**`_` padding fix** — Filters `_` placeholder rows in `load_bilingual_dataset()` (.cog convention).

**normalize_token fix** — Added `` ` `` and `@` to regex whitelist for Ugaritic transliteration.

### Pipeline Additions (commit 07b33e7)

**`--corpus-variant` flag** — All experiment subcommands accept `--corpus-variant religious` to train on religious-text subsets via `get_corpus(name, variant="religious")`.

**`validation` subcommand** — Run against any of 55 validation branches:
```bash
python -m repro.run_experiment validation --branch germanic_expanded --smoke
python -m repro.run_experiment validation --branch semitic --lost-lang pair:arabic:hebrew
```
Uses registry's `_build_validation_corpus()` with concept-aligned cognate pairs.

**`configs/validation.yaml`** — Generic config for validation branches (embedding_dim=50, 3000 steps, base+full variants).

### Smoke Test Results

End-to-end model test (Gothic data, panphon features, GroupedIPAProjector):
- Feature matrix: 17 chars × 21 features (panphon)
- GroupedIPAProjector active with 5 groups
- 3 training steps complete, omega_cov differentiable
- All data loading tests pass (Ugaritic: 203q/275kv, Gothic: 34q/34kv, Iberian: 62q/64kv)

### Remaining Items (Lower Priority) — Now Complete

| # | Action | Status |
|---|---|---|
| 8 | Add cognate annotation to validation branches | DONE (`d256fb0`) |
| 9 | Build results visualization dashboard | DONE (`d256fb0`) |
| 10 | Clean ABVD/WOLD data quality | DONE (`d256fb0`) |

---

## 2026-02-20 — Implementation: Panphon + Grouped Projections (commit 419eef7)

**Phase:** Implementation
**Status:** Complete — pushed to main
**Commit:** `419eef7`

### Changes Made

**`repro/eval/common.py` — Feature extraction rewrite**
- Replaced `build_char_feature_matrix()`: was 8-dim pseudo-random (`ord(char) % N`), now 21-dim panphon articulatory features
- Added `PANPHON_FEATURE_GROUPS` constant defining 5 phonological categories:
  - `major_class` (3): syl, son, cons
  - `manner` (5): cont, delrel, lat, nas, strid
  - `laryngeal` (3): voi, sg, cg
  - `place` (4): ant, cor, distr, lab
  - `vowel_body` (6): hi, lo, back, round, tense, long
- Added `_IPA_CHAR_NORM` mapping for non-IPA characters in our data: Gothic þ->θ, ƕ->x, ȝ->ɣ, macron vowels->IPA long, ASCII g->IPA ɡ
- Added diacritic fallback: strips combining marks and retries base character
- Lazy-loaded panphon with `PYTHONUTF8=1` to handle Windows cp1252 encoding issue

**`repro/model/phonetic_prior.py` — Grouped projector architecture**
- Added `GroupedIPAProjector(nn.Module)`: 5 independent `nn.Linear(group_size, group_dim, bias=False)` modules
- Embedding dim distributed evenly: e.g., dim=35 -> 5 groups of 7 dims each
- `PhoneticPriorModel.__init__` auto-selects `GroupedIPAProjector` when feature dim matches panphon layout (sum of group sizes == 21), falls back to single `nn.Linear` for one-hot features
- Added `ipa_group_sizes` to `PhoneticPriorConfig` dataclass

**`requirements.txt`**
- Added `panphon>=0.22`

### Verification Results

| Test | Result |
|---|---|
| [b] vs [p] feature diff | 1/21 (voicing only) — correct |
| [d] vs [t] feature diff | 1/21 (voicing only) — correct |
| [s] vs [z] feature diff | 1/21 (voicing only) — correct |
| [a] vs [i] feature diff | 3/21 (height, backness, tense) — correct |
| b-p similarity > b-k similarity | Yes, even at random init |
| Gradients through all 5 groups | Yes (verified steps 1-5) |
| One-hot fallback (base model) | Works, uses single nn.Linear |
| IPA coverage | 29/31 chars direct, 2/2 via normalisation |

### Architecture Before vs After

```
BEFORE (pseudo-random):
  char -> [is_vowel, is_cons, ord%2, ord%3/2, ord%5/4, ord%7/6, ord%11/10, len%3/2]
       -> nn.Linear(8, 35)    # single mixed projection
       -> 35-dim embedding     # phonologically meaningless

AFTER (panphon + grouped):
  char -> panphon -> [major_class(3) | manner(5) | laryngeal(3) | place(4) | vowel_body(6)]
       -> [nn.Linear(3,7) | nn.Linear(5,7) | nn.Linear(3,7) | nn.Linear(4,7) | nn.Linear(6,7)]
       -> concat -> 35-dim embedding   # 5 independent phonological subspaces
```

---

## 2026-02-20 — IPA Feature Bottleneck: Root Cause & Solution Selection

**Phase:** Pre-implementation analysis
**Status:** Solution D (panphon + grouped embeddings) selected for implementation
**Scope:** Paper vs. reproduction gap analysis, feature encoding options, architectural decision

---

### Root Cause: Paper vs. Reproduction Mismatch

The paper (Luo et al. 2021, Section 3.2.1 + Figure 3 + Appendix A.2-A.3) describes a **compositional IPA embedding** system:

> "Each IPA character is represented by a vector of phonological features. The model learns to embed these features into a new space... the phone [b] can be represented as the concatenation of the **voiced** embedding, the **stop** embedding and the **labial** embedding."

The **original code** (in `j-luo93/xib` and `j-luo93/DecipherUnsegmented`) implements this correctly:
- Uses `ipapy` to extract **14 phonological categories** (112 total feature values)
- Selects **7 feature groups**: ptype, c_voicing, c_place, c_manner, v_height, v_backness, v_roundness
- Each group gets a learned `nn.Embedding(num_features_in_group, 100)`
- Concatenates all 7 group embeddings -> **700-dimensional** character embedding
- Result: [b] and [p] share 2/3 of their embedding (both are stop + labial)

The **reproduction code** (`repro/eval/common.py:253-282`) replaces all of this with:
```python
# Pseudo-random features -- NOT phonologically grounded
float(code % 2), float(code % 3)/2.0, float(code % 5)/4.0, ...
```
This produces an **8-dimensional** vector where 6 of 8 dimensions are Unicode codepoint modular arithmetic. Under this scheme, [b] (U+0062) and [p] (U+0070) get completely unrelated feature vectors -- destroying the geometric structure that is the entire point of the phonetic prior.

The paper's ablation (Table 4) shows IPA embeddings provide a **+12.8% improvement** on Gothic-Old Norse. Without real phonological features, the "phonetic prior" variant operates on noise.

---

### Solutions Evaluated

| Solution | Description | Feature Dim | Grouped? | Phonological Geometry |
|---|---|---|---|---|
| **A: panphon flat** | 21 ternary features -> single nn.Linear | 21 | No | Encoded in input, but mixed by projection |
| **B: soundvectors** | 39 ternary features (incl. tone, diphthongs) | 39 | No | Richest input, but 15 dims wasted on non-tonal languages |
| **C: Original ipapy** | Port authors' 112-feature system from xib | 700 | Yes (7 groups) | Exact paper reproduction |
| **D: panphon grouped** | 21 features in 5 groups, per-group nn.Linear | 5 x group_d | Yes (5 groups) | Best: real features + group independence |

### Why Not B (soundvectors)?

Soundvectors has 39 features: 24 core + 6 tone + 7 diphthong trajectory + 2 additional place. For our target languages (Ugaritic, Gothic, Iberian): none are tonal (6 dims = 0), diphthongs are transcribed as monophthong sequences at the character level (7 dims = 0). So 15/39 dimensions carry no signal. The 24 core features are only marginally richer than panphon's 21 (adding explicit pharyngeal/laryngeal which panphon encodes via feature combinations). The CLTS data dependency adds setup fragility for no gain.

### Why D Over A: Group Independence Matters for Sound Shifts

Sound changes operate along **specific phonological dimensions**:
- Grimm's Law: voiced stops -> voiceless stops (voicing change only)
- Palatalization: velars -> palatals before front vowels (place change only)
- Lenition: stops -> fricatives (manner change only)
- Great Vowel Shift: systematic height changes (vowel height only)

**Solution A** (flat projection) passes 21 features through a single `nn.Linear(21, d)`. This freely mixes all features in one 35-dimensional space. The model must discover from limited data that voicing and place are independent dimensions -- a structural fact it should get for free.

**Solution D** (grouped projections) gives each phonological category its own `nn.Linear(group_size, group_d)`, then concatenates. The character embedding naturally decomposes into independent subspaces:
```
emb([b]) = [manner_emb | laryngeal_emb | place_emb | vowel_emb | class_emb]
```
When the model learns that Gothic [p] maps to Old Norse [f] (Grimm's Law), it learns a transformation in the manner subspace without disturbing place or voicing. This is the paper's key inductive bias.

### Selected: Solution D — panphon Features with Grouped Embeddings

**Feature groups (matching standard phonological categories):**

| Group | panphon Features | Dim |
|---|---|---|
| major_class | syl, son, cons | 3 |
| manner | cont, delrel, lat, nas, strid | 5 |
| laryngeal | voi, sg, cg | 3 |
| place | ant, cor, distr, lab | 4 |
| vowel_body | hi, lo, back, round, tense, long | 6 |

5 groups, 21 total features. Each group projected to `embedding_dim // 5` dimensions.

**Architecture change:** Replace `self.ipa_projector = nn.Linear(feat_dim, d)` with per-group `nn.Linear` modules that concatenate to produce the same output dimension. The rest of the model (`compute_char_distr`, `edit_distance_dp`, `word_boundary_dp`) remains unchanged.

---

## 2026-02-20 — Pre-Run Investigation: Model Architecture Deep Dive

**Phase:** Pre-training investigation
**Status:** No model run yet — blocking issues identified
**Scope:** Architecture analysis, data compatibility, subset strategy, pipeline readiness

---

### 1. What Exactly Does the Model Do?

The PhoneticPriorModel is a **probabilistic unsupervised decipherment model** with ~1,412 learnable parameters. It learns a character-level mapping between an unknown ("lost") script and a known language, using differentiable dynamic programming.

**Core mechanism:**
- **`mapping_logits`** [K x L matrix]: Learnable logits representing the probability that each lost-script character maps to each known-language character. Trained via softmax -> dot-product scoring -> log-softmax.
- **`ipa_projector`** (nn.Linear): Projects IPA feature vectors into an embedding space where phonetically similar characters are close together. This is the "phonetic prior" — it biases the mapping toward phonologically plausible correspondences.
- **Two-level differentiable DP:**
  1. **Edit Distance DP** (`edit_distance_dp`): Computes soft edit distance between a lost-script token and a known-language word using softmin/logsumexp (differentiable approximation of min). Substitution cost = -log(char_distr[k, l]); gap cost = alpha.
  2. **Word Boundary DP** (`word_boundary_dp`): For unsegmented inscriptions (Gothic, Iberian), segments the inscription into spans and scores each span against the known vocabulary.
- **Alpha annealing**: Gap penalty linearly decreases from 10.0 -> 3.5 over 2,000 steps (curriculum learning — starts strict, gradually allows more insertions/deletions).
- **Omega loss**: Entropy regularizer preventing the mapping from collapsing to a trivial solution.

**Training is fully unsupervised** — cognate labels are used only for evaluation (P@1, P@10, MRR), never during training.

---

### 2. Is the Architecture Optimal for This Purpose?

**Strengths:**
- Minimal parameter count (~1,412) is appropriate — the problem is fundamentally about character-level correspondences, not high-dimensional representation learning.
- Differentiable DP is elegant and correct for sequence alignment.
- Alpha annealing provides effective curriculum learning.
- The model successfully reproduced paper results for Ugaritic->Hebrew decipherment.

**Identified Weaknesses:**

| Issue | Severity | Detail |
|---|---|---|
| **IPA features are pseudo-random** | HIGH | `build_char_feature_matrix()` uses `ord(char) % dim` — Unicode codepoint modular arithmetic, not actual phonological features. The "phonetic prior" is barely phonetic. |
| **omega_cov is gradient-dead** | MEDIUM | Coverage term is a plain Python float, not a tensor. It contributes to the objective value but produces zero gradient signal — the model cannot learn from it. |
| **Context-independent mappings** | MEDIUM | Each character maps independently regardless of position or neighbors. Real sound correspondences are often context-dependent (e.g., initial vs. medial position). |
| **O(n x m) ranking** | LOW | Evaluation ranks each query against the entire known vocabulary via pairwise edit distance. Scales poorly to large vocabularies (100K+), but fine for our datasets. |

**Verdict:** The architecture is sound for its intended purpose but has two clear improvement opportunities: (1) replacing pseudo-random IPA features with real phonological features, and (2) making omega_cov a proper tensor so it contributes gradients.

---

### 3. Will Our Prepared Data Be Correctly Formatted?

**BLOCKING issues that must be fixed before running:**

| Issue | Detail | Fix Required |
|---|---|---|
| **Missing third_party files** | YAML configs point to `third_party/NeuroDecipher/data/uga-heb.small.no_spe.cog` and `third_party/DecipherUnsegmented/` — these directories are empty | Symlink or copy our `ancient-scripts-datasets/data/` files into the expected paths, OR update YAML configs to point to new locations |
| **Column name mismatch** | `export_training.py` writes ISO codes as column headers; `load_bilingual_dataset()` uses flexible matching via `_pick_column()` but expects recognizable language names | Verify column names match or update `_pick_column()` canonicalization |
| **`_` padding rows** | Original `.cog` files use `_` for missing entries. `load_bilingual_dataset()` doesn't filter these, polluting the vocabulary | Add `_` filtering in data loading |
| **normalize_token corruption** | Regex `\W+` strips backtick and `@` from Ugaritic transliteration tokens, corrupting them | Adjust regex to preserve transliteration-specific characters |

**Non-blocking but important:**
- ABVD data (57.7% orthographic, not IPA) — won't affect the core Ugaritic/Gothic/Iberian experiments but will degrade any training that uses ABVD languages
- WOLD slash contamination (10,029 rows with `/` allophone notation) — needs cleanup before use

---

### 4. Can Religious/Name Subsets Improve Accuracy?

**Yes, but with caveats.**

The model is unsupervised — it doesn't use labels during training. However, subsets can help in two ways:

1. **As focused evaluation sets**: Religious terms (241 Ugaritic-Hebrew pairs) and name pairs (`names_pairs.tsv` with explicit Is_Cognate flags) provide high-quality evaluation data. Running the model and evaluating on these subsets tells you how well the mapping works for semantically coherent domains.

2. **As constrained search spaces**: If you feed the model only religious terms as the "known vocabulary," the smaller search space (fewer distractors) should mechanically improve P@1/P@10 metrics. This is a valid experimental design — it tests whether domain-focused vocabularies yield better decipherment.

**Current status:**
- Religious subsets ARE registered in `registry.py` (`_build_ugaritic_religious`, `_build_gothic_religious`, `_build_iberian_religious`)
- But the experiment runner does NOT wire `variant="religious"` — this needs to be connected
- `names_pairs.tsv` is ideal for evaluation (has explicit cognate/non-cognate labels)

**Recommendation:** Add a `--variant religious` flag to the experiment runner and create YAML configs that use religious subset vocabularies. This is straightforward — maybe 20 lines of code.

---

### 5. Do Cognate Pairing Datasets Need Reformatting?

**Yes — there's a critical flaw.**

| Dataset | Status | Issue |
|---|---|---|
| **Original .cog files** (Ugaritic-Hebrew) | Compatible | Format matches model expectations. Just need correct file paths. |
| **Validation branch TSVs** | BROKEN | Have NO `Cognate_Set_ID`. `export_training.py` treats shared concept as cognacy — this is wrong. Shared concept != cognate (3-16% false positive rate depending on family). |
| **Religious subsets** | Good quality | Hand-curated by scholars, verified against sources. Need to be wired into the pipeline. |
| **names_pairs.tsv** | Best quality | Has explicit `Is_Cognate` boolean flags. Ideal for evaluation. |

**Required fixes:**
1. Validation branch files need proper cognate annotation (either manual or via automated cognate detection algorithms like LexStat)
2. `export_training.py` should flag concept-matched pairs as "candidate" not "confirmed cognate"
3. For immediate use, stick with the original `.cog` files (Ugaritic-Hebrew) which have verified cognate pairs

---

### 6. What Conditions Need To Be Set for a New Run?

**Environment requirements:**
- Python 3.10 (pinned in the repo — our 3.13.2 may cause issues)
- PyTorch 2.2.2 (CPU is fine, no GPU needed)
- Dependencies: numpy, matplotlib, pyyaml, scipy, pandas

**Pre-run checklist:**

1. **Fix data paths** — Either:
   - Update YAML configs to point to `ancient-scripts-datasets/data/`
   - Or symlink files into `third_party/` directories
2. **Fix `_` filtering** in `load_bilingual_dataset()`
3. **Fix normalize_token** regex for transliteration characters
4. **Create new YAML configs** for:
   - Religious subset experiments
   - Name-based evaluation experiments
   - Each validation branch (Germanic, Celtic, etc.)
5. **Optionally fix IPA features** — Replace pseudo-random `ord() % dim` with real PHOIBLE/panphon features
6. **Optionally fix omega_cov** — Wrap in `torch.tensor()` so it contributes gradients

**Training time estimate:** Minutes per single run, hours for full sweep across all branches and variants.

---

### 7. Do We Need a New Results Interface?

**The existing interface is adequate but could be improved.**

**What exists:**
- Per-query CSV output with P@1, P@10, MRR columns
- Summary statistics printed to stdout
- Comparison tools for base vs. full (with phonetic prior) variants
- Output organized in `results/<experiment>/<variant>/` directories

**What's missing:**
- No mapping visualization (which lost character -> which known character)
- No side-by-side comparison across branches/subsets
- No aggregate dashboard across all experiments
- No confusion matrix for character mappings

**Recommendation:** The existing CSV output is sufficient for initial runs. A visualization dashboard would be valuable but is not blocking — we can build it after getting initial results.

---

### 8. Summary: Action Items for New Model Run

**Priority order:**

| # | Action | Effort | Blocking? |
|---|---|---|---|
| 1 | Fix data file paths in YAML configs | Small | YES |
| 2 | Fix `_` padding filtering | Small | YES |
| 3 | Fix normalize_token for transliteration chars | Small | YES |
| 4 | Wire religious/name subsets into experiment runner | Medium | No (but high value) |
| 5 | Create per-branch YAML configs | Medium | No |
| 6 | Replace pseudo-random IPA features with real phonological features | Medium | No (but significant accuracy improvement expected) |
| 7 | Fix omega_cov gradient death | Small | No |
| 8 | Add cognate annotation to validation branches | Large | No (use original .cog files first) |
| 9 | Build results visualization dashboard | Medium | No |
| 10 | Clean ABVD/WOLD data quality issues | Large | No (not used in core experiments) |

Items 1-3 are required before any model run. Items 4-6 are the highest-value improvements. The rest can follow incrementally.

---

*Entry added by automated investigation (4 parallel agents: architecture, compatibility, subset/eval, pipeline). All source files read directly from the repository.*
