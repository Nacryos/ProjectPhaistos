# Run Log — PhoneticPriorModel

Reverse-chronological notes, reports, and observations from model investigation, training runs, and data preparation. Each entry is timestamped for tracking improvements over time.

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
