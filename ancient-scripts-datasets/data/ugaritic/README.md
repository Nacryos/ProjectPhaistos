# Ugaritic-Hebrew Cognate Data

## Format

The `.cog` files use the COG format from NeuroDecipher (Snyder et al. 2010).

**Tab-separated, 2 columns:**
- Column 1 (`uga-no_spe`): Ugaritic consonantal transliteration
- Column 2 (`heb-no_spe`): Hebrew consonantal transliteration

**Special conventions:**
- `_` marks a missing entry (no known cognate in that language)
- `|` separates alternative cognate forms (e.g., `bhm|bm`)

## Files

| File | Rows | Description |
|------|------|-------------|
| `uga-heb.no_spe.cog` | 43,924 | Full file including Hebrew search vocabulary |
| `uga-heb.pairs_only.cog` | 2,187 | Cognate pairs only (both languages present) |
| `uga-heb.small.no_spe.cog` | 4,535 | ~10% stratified sample of the full file |

## Data Structure

The full file (`uga-heb.no_spe.cog`) contains:

| Category | Rows | % |
|----------|------|---|
| Both Ugaritic AND Hebrew present (true cognate pairs) | 2,187 | 5.0% |
| Hebrew only (Ugaritic = `_`) | 36,684 | 83.5% |
| Ugaritic only (Hebrew = `_`) | 5,053 | 11.5% |

**The asymmetry is intentional.** The decipherment model needs the complete Hebrew
vocabulary as its search space, not just the matched cognate pairs. The `_` entries
represent the full Hebrew lexicon that the model searches against when deciphering
Ugaritic text.

For training on **cognate pairs only**, use `uga-heb.pairs_only.cog` which contains
only the 2,187 rows where both languages are present.

## Source

Originally from Snyder et al. (2010), provided via the
[NeuroDecipher](https://github.com/j-luo93/NeuroDecipher) repository.
Covers 7,236 unique Ugaritic types from the 14th-12th century BC.
