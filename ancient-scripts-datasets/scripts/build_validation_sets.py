#!/usr/bin/env python3
"""Build stratified validation datasets for cognate detection ML training.

Reads lexicon TSVs and cognate-pair TSVs from data/training/, builds a
phylogenetic tree of language relationships, and generates stratified
validation sets split by phylogenetic distance, timespan, family, and
concept domain (religious terms).

Output goes to data/training/validation/.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure cognate_pipeline package is importable
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "cognate_pipeline" / "src"),
)
from cognate_pipeline.normalise.sound_class import ipa_to_sound_class

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO_ROOT / "data" / "training"
LEXICONS_DIR = TRAINING_DIR / "lexicons"
COGNATE_DIR = TRAINING_DIR / "cognate_pairs"
OUTPUT_DIR = TRAINING_DIR / "validation"
FAMILY_MAP_PATH = (
    REPO_ROOT
    / "cognate_pipeline"
    / "src"
    / "cognate_pipeline"
    / "cognate"
    / "family_map.json"
)

PAIR_CAP = 50_000
SEED = 42
MAX_PAIRS_PER_CONCEPT_PER_LEVEL = 100
MAX_CROSS_FAMILY_PAIRS_PER_CONCEPT = 50
TRUE_NEG_SAMPLE_ATTEMPTS = 2_000_000

# ---------------------------------------------------------------------------
# TSV field names for output
# ---------------------------------------------------------------------------

OUTPUT_FIELDS = [
    "Lang_A",
    "Word_A",
    "IPA_A",
    "SCA_A",
    "Lang_B",
    "Word_B",
    "IPA_B",
    "SCA_B",
    "Concept_ID",
    "Label",
    "Phylo_Dist",
    "Timespan",
    "Score",
    "Source",
]

# ---------------------------------------------------------------------------
# Era classification
# ---------------------------------------------------------------------------

ANCIENT: set[str] = {
    "grc", "lat", "san", "ave", "got", "akk", "egy", "phn", "uga",
    "sux", "hit", "osc", "xum", "gmy", "sga", "chu", "och", "obr",
    "cop", "arc", "syc", "ett",
}

MEDIEVAL: set[str] = {
    "ang", "enm", "fro", "osp", "non", "goh", "dum", "mga", "wlm",
    "orv", "otk", "ota", "okm", "kaw", "mnc", "bod",
}


def classify_era(iso: str) -> str:
    """Return 'ancient', 'medieval', or 'modern'."""
    if iso in ANCIENT:
        return "ancient"
    if iso in MEDIEVAL:
        return "medieval"
    return "modern"


def get_timespan(iso_a: str, iso_b: str) -> str:
    """Return one of the four canonical timespan buckets."""
    era_a = classify_era(iso_a)
    era_b = classify_era(iso_b)
    eras = frozenset((era_a, era_b))
    if eras == {"ancient"}:
        return "ancient_ancient"
    if eras == {"modern"}:
        return "modern_modern"
    if eras == {"medieval"}:
        return "medieval_modern"
    if "ancient" in eras and "medieval" in eras:
        return "ancient_modern"
    if "ancient" in eras and "modern" in eras:
        return "ancient_modern"
    # medieval + modern
    return "medieval_modern"


# ---------------------------------------------------------------------------
# Religious concepts — organised by sub-domain (generous classification)
# ---------------------------------------------------------------------------

RELIGIOUS_CORE: set[str] = {
    "DEITY", "DEITY/GOD", "GOD", "SPIRIT", "TEMPLE", "ALTAR", "SACRIFICE",
    "WORSHIP", "PRAY", "PRIEST", "HOLY", "PREACH", "BLESS", "CHURCH",
    "MOSQUE", "SOUL", "RELIGION", "IDOL", "MINISTER",
}

RELIGIOUS_SUPERNATURAL: set[str] = {
    "GHOST", "DEMON", "MAGIC", "SORCERER", "MAGICIAN", "OMEN",
    "ELF OR FAIRY", "FAIRY TALE", "DREAM (SOMETHING)", "DREAM",
}

RELIGIOUS_MORAL: set[str] = {
    "SIN", "BAD OR EVIL", "EVIL", "BELIEVE", "TRUTH", "SHAME",
    "GUILTY", "CRIME", "ADULTERY", "PITY", "FORGIVE", "INNOCENT",
    "ACCUSE", "CONDEMN", "JUDGE", "JUDGMENT", "LAW", "PUNISHMENT",
    "WITNESS", "POOR", "RICH", "FAITHFUL", "GOOD",
}

RELIGIOUS_RITUAL: set[str] = {
    "CURSE", "FAST", "CIRCUMCISION", "INITIATION CEREMONY", "WEDDING",
    "OATH", "SWEAR", "CUSTOM", "BURY", "GRAVE", "CORPSE", "DANCE",
    "DRUM", "SONG",
}

RELIGIOUS_VERBS: set[str] = {
    "GIVE", "DONATE", "KNEEL", "BURN (SOMETHING)", "KILL", "POUR",
    "FEED", "SHARE", "INVITE", "COMMAND", "PROMISE", "OBEY",
    "HELP", "PROTECT", "DEFEND", "RESCUE", "HOPE (SOMETHING)",
    "FEAR (BE AFRAID)", "FEAR (FRIGHT)", "LOVE",
}

RELIGIOUS_COSMIC: set[str] = {
    "HEAVEN", "HELL", "LIGHTNING", "THUNDER", "FIRE", "SUN", "MOON",
    "STAR", "SKY", "RAINBOW", "EARTHQUAKE", "WORLD", "LIFE",
    "DEATH", "BE DEAD OR DIE", "BE ALIVE", "BE BORN",
    "ANCESTORS", "DESCENDANTS",
}

RELIGIOUS_PLACES: set[str] = {
    "TEMPLE", "CHURCH", "MOSQUE", "CAVE", "MOUNTAIN", "MOUNTAIN OR HILL",
    "SPRING OR WELL", "GARDEN", "FOREST", "STONE", "STONE OR ROCK",
    "VILLAGE", "TOWN", "COUNTRY", "ISLAND", "RIVER", "SEA",
    "NATIVE COUNTRY",
}

# Additional numeric IDs from cross-lingual datasets that map to religious concepts
_RELIGIOUS_NUMERIC_IDS: set[str] = {
    "3231", "53", "911", "853", "1103", "257", "24", "852", "1702", "304",
    "391", "8", "303", "1565", "878", "1973", "1945", "392", "2137", "1175",
    "107", "1349", "1603", "811", "2971", "661", "1944",
}

# Union of all sub-domains — the generous set
RELIGIOUS_ALL: set[str] = (
    RELIGIOUS_CORE
    | RELIGIOUS_SUPERNATURAL
    | RELIGIOUS_MORAL
    | RELIGIOUS_RITUAL
    | RELIGIOUS_VERBS
    | RELIGIOUS_COSMIC
    | RELIGIOUS_PLACES
    | _RELIGIOUS_NUMERIC_IDS
)

# Sub-domain name → concept set mapping (for generating sub-domain files)
RELIGIOUS_SUBDOMAINS: dict[str, set[str]] = {
    "core_religious": RELIGIOUS_CORE,
    "supernatural": RELIGIOUS_SUPERNATURAL,
    "moral_ethical": RELIGIOUS_MORAL,
    "ritual_ceremony": RELIGIOUS_RITUAL,
    "religious_verbs": RELIGIOUS_VERBS,
    "cosmic_spiritual": RELIGIOUS_COSMIC,
    "sacred_places": RELIGIOUS_PLACES,
}

# Pre-compute uppercase set for fast case-insensitive matching
_RELIGIOUS_ALL_UPPER: set[str] = {c.upper() for c in RELIGIOUS_ALL}


def is_religious(concept_id: str) -> bool:
    """Return True if *concept_id* refers to a religious concept."""
    if concept_id in RELIGIOUS_ALL:
        return True
    return concept_id.upper() in _RELIGIOUS_ALL_UPPER


def _in_subdomain(concept_id: str, subdomain_set: set[str]) -> bool:
    """Return True if *concept_id* belongs to the given sub-domain set."""
    if concept_id in subdomain_set:
        return True
    return concept_id.upper() in {c.upper() for c in subdomain_set}


# ---------------------------------------------------------------------------
# Top families
# ---------------------------------------------------------------------------

TOP_FAMILIES = [
    "germanic", "italic", "balto_slavic", "indo_iranian", "hellenic",
    "celtic", "uralic", "turkic", "sino_tibetan", "austronesian",
    "semitic", "dravidian", "japonic", "koreanic", "kartvelian",
]

# Map family_map values that differ from the tree's branch names
FAMILY_ALIAS = {
    "slavic": "balto_slavic",
    "baltic": "balto_slavic",
}

# ---------------------------------------------------------------------------
# SCA similarity (standalone, mirrors baseline_levenshtein.py)
# ---------------------------------------------------------------------------

_VOWELS = set("AEIOU")
_LABIALS = {"P", "B", "M"}
_CORONALS = {"T", "D", "N", "S", "L", "R"}
_VELARS = {"K", "G"}
_LARYNGEALS = {"H"}
_GLIDES = {"W", "Y"}
_NATURAL_CLASSES = [_VOWELS, _LABIALS, _CORONALS, _VELARS, _LARYNGEALS, _GLIDES]


def _substitution_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    for cls in _NATURAL_CLASSES:
        if a in cls and b in cls:
            return 0.3
    return 1.0


def weighted_levenshtein(s1: str, s2: str) -> float:
    n, m = len(s1), len(s2)
    if n == 0:
        return m * 0.5
    if m == 0:
        return n * 0.5
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * 0.5
    for j in range(m + 1):
        dp[0][j] = j * 0.5
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = _substitution_cost(s1[i - 1], s2[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + 0.5,
                dp[i][j - 1] + 0.5,
                dp[i - 1][j - 1] + sub,
            )
    return dp[n][m]


def normalised_similarity(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    dist = weighted_levenshtein(s1, s2)
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0


# ---------------------------------------------------------------------------
# Phylogenetic tree definition
# ---------------------------------------------------------------------------

def build_raw_tree() -> dict[str, Any]:
    """Return the hard-coded phylogenetic tree.

    Leaf-group values are either lists of ISO codes or the sentinel
    ``"__from_family_map__"`` which is resolved later.
    """
    return {
        "indo_european": {
            "germanic": {
                "west_germanic": {
                    "anglo_frisian": ["eng", "ang", "enm", "fry", "frr", "ofs"],
                    "franconian": ["nld", "dum", "lim", "afr"],
                    "high_german": ["deu", "goh", "gsw", "bar", "ltz", "yid"],
                },
                "north_germanic": [
                    "swe", "dan", "nor", "nno", "nob", "isl", "fao", "non",
                ],
                "east_germanic": ["got"],
            },
            "italic": {
                "romance": {
                    "ibero_romance": ["spa", "por", "cat", "glg", "osp"],
                    "gallo_romance": ["fra", "oci", "fro"],
                    "italo_dalmatian": ["ita", "nap", "scn", "dlm", "cos"],
                    "eastern_romance": ["ron", "rup"],
                },
                "latino_faliscan": ["lat", "osc", "xum"],
            },
            "celtic": {
                "goidelic": ["gle", "gla", "sga", "mga"],
                "brythonic": ["cym", "bre", "cor", "wlm"],
            },
            "balto_slavic": {
                "baltic": ["lit", "lav", "ltg"],
                "east_slavic": ["rus", "ukr", "bel", "orv"],
                "west_slavic": ["pol", "ces", "slk", "dsb", "hsb", "csb", "pox"],
                "south_slavic": ["bul", "mkd", "hrv", "slv", "hbs", "chu"],
            },
            "hellenic": ["ell", "grc", "gmy"],
            "indo_iranian": {
                "iranian": [
                    "fas", "pes", "oss", "kmr", "ckb", "pbu", "tgk", "ave",
                    "zza",
                ],
                "indic": [
                    "hin", "ben", "san", "guj", "mar", "pan", "sin", "urd",
                    "asm", "nep", "rom", "rmn",
                ],
            },
            "armenian": ["hye"],
            "albanian": ["sqi"],
            "anatolian": ["hit"],
        },
        "uralic": {
            "finnic": [
                "fin", "est", "ekk", "krl", "olo", "vep", "vot", "izh", "liv",
            ],
            "ugric": ["hun", "mns", "kca"],
            "samic": ["sme", "sma", "smj", "smn", "sms", "sjd"],
            "mordvinic": ["myv", "mdf"],
            "permic": ["kpv", "koi", "udm"],
            "mari": ["mhr", "mrj"],
            "samoyedic": ["yrk", "enf", "sel", "nio"],
        },
        "turkic": {
            "oghuz": ["tur", "aze", "azj", "ota", "otk"],
            "kipchak": ["kaz", "kir", "tat", "bak"],
            "siberian": ["sah", "tyv"],
            "karluk": ["uzb", "uzn"],
            "oghur": ["chv"],
        },
        "sino_tibetan": {
            "sinitic": ["zho", "cmn", "yue", "och"],
            "tibeto_burman": ["bod", "mya", "obr", "new", "lif"],
        },
        "austronesian": {
            "malayo_polynesian": "__from_family_map__",
        },
        "semitic": [
            "heb", "arb", "ara", "amh", "mlt", "syc", "arc", "akk", "phn",
            "uga",
        ],
        "dravidian": ["tam", "tel", "kan", "mal"],
        "japonic": ["jpn"],
        "koreanic": ["kor", "jje", "okm"],
        "kartvelian": ["kat", "lzz"],
    }


# ---------------------------------------------------------------------------
# Tree resolution helpers
# ---------------------------------------------------------------------------

def _collect_isos_from_tree(node: Any) -> set[str]:
    """Recursively collect all ISO codes that already appear in *node*."""
    if isinstance(node, list):
        return set(node)
    if isinstance(node, str):
        if node == "__from_family_map__":
            return set()
        return {node}
    isos: set[str] = set()
    for v in node.values():
        isos |= _collect_isos_from_tree(v)
    return isos


def resolve_tree(tree: dict[str, Any], family_map: dict[str, str]) -> dict[str, Any]:
    """Replace ``"__from_family_map__"`` sentinels and add catch-all groups.

    Returns a new tree (original is not mutated).
    """
    tree = _deep_copy_tree(tree)

    # Phase 1: resolve sentinels -----------------------------------------
    _resolve_sentinels(tree, family_map)

    # Phase 2: add catch-all for languages in family_map but not in tree --
    present = _collect_isos_from_tree(tree)
    extras: dict[str, list[str]] = defaultdict(list)
    for iso, fam in family_map.items():
        if iso in present:
            continue
        canonical = FAMILY_ALIAS.get(fam, fam)
        extras[canonical].append(iso)

    for fam, isos in extras.items():
        if fam not in tree:
            tree[fam] = sorted(isos)
        else:
            # Family exists as a top-level node — add under an
            # "other_{fam}" subgroup so we don't clobber existing structure.
            node = tree[fam]
            if isinstance(node, dict):
                existing = _collect_isos_from_tree(node)
                new_isos = [i for i in isos if i not in existing]
                if new_isos:
                    node[f"other_{fam}"] = sorted(new_isos)
            elif isinstance(node, list):
                existing = set(node)
                for iso in isos:
                    if iso not in existing:
                        node.append(iso)
            # If the node is a single string, wrap it
            elif isinstance(node, str) and node != "__from_family_map__":
                tree[fam] = [node] + sorted(isos)

    return tree


def _deep_copy_tree(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _deep_copy_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return list(node)
    return node


def _resolve_sentinels(node: Any, family_map: dict[str, str]) -> None:
    """In-place replacement of ``"__from_family_map__"`` values."""
    if not isinstance(node, dict):
        return
    for key, val in list(node.items()):
        if val == "__from_family_map__":
            # key is the family name that should match family_map values
            # For "malayo_polynesian" under "austronesian", pull all
            # family_map entries mapped to "austronesian".
            parent_family = _find_parent_family(node, key)
            if parent_family is None:
                parent_family = key
            isos = sorted(
                iso for iso, fam in family_map.items() if fam == parent_family
            )
            node[key] = isos if isos else []
        elif isinstance(val, dict):
            _resolve_sentinels(val, family_map)


def _find_parent_family(node: dict, child_key: str) -> str | None:  # noqa: ARG001
    """Heuristic: the sentinel is typically placed one level below the
    actual family name.  Walk the raw tree keys for a match.  For our
    tree, ``malayo_polynesian`` is under ``austronesian``, so we return
    ``austronesian``."""
    # We rely on the caller context; this is called from _resolve_sentinels
    # which walks the tree recursively.  At the point we find the sentinel
    # the *node* dict is ``{"malayo_polynesian": "__from_family_map__"}``,
    # and we need the grandparent key.  Since we don't track the parent key
    # inside the recursive walk, we use a simpler approach: just look up in
    # a mapping.
    _SENTINEL_PARENT: dict[str, str] = {
        "malayo_polynesian": "austronesian",
    }
    return _SENTINEL_PARENT.get(child_key)


# ---------------------------------------------------------------------------
# Language path index & phylo distance
# ---------------------------------------------------------------------------

def build_lang_paths(
    tree: dict[str, Any],
) -> dict[str, list[str]]:
    """Map each ISO code to its full path from root to its leaf group.

    For ``eng`` inside ``indo_european > germanic > west_germanic >
    anglo_frisian`` the path is
    ``["indo_european", "germanic", "west_germanic", "anglo_frisian"]``.
    """
    paths: dict[str, list[str]] = {}

    def _walk(node: Any, prefix: list[str]) -> None:
        if isinstance(node, list):
            for iso in node:
                paths[iso] = list(prefix)
        elif isinstance(node, dict):
            for key, child in node.items():
                _walk(child, prefix + [key])
        elif isinstance(node, str):
            # Single ISO as a leaf
            paths[node] = list(prefix)

    _walk(tree, [])
    return paths


def compute_distance(
    lang_a: str,
    lang_b: str,
    lang_paths: dict[str, list[str]],
) -> tuple[int, str]:
    """Return (edge_count, level_label) between two languages.

    Level mapping:
        L1 — same leaf group (same last path element)
        L2 — LCA is two levels above leaf
        L3 — LCA is deeper within same top-level family
        L4 — different top-level families or not in tree
    """
    pa = lang_paths.get(lang_a)
    pb = lang_paths.get(lang_b)
    if pa is None or pb is None:
        return (99, "L4")
    if not pa or not pb:
        return (99, "L4")

    # Find LCA depth
    lca_depth = 0
    for i, (a, b) in enumerate(zip(pa, pb)):
        if a != b:
            break
        lca_depth = i + 1
    else:
        # Exhausted the shorter path without mismatch
        lca_depth = min(len(pa), len(pb))

    if lca_depth == 0:
        # Different top-level families
        return (len(pa) + len(pb), "L4")

    depth_a = len(pa)
    depth_b = len(pb)
    edge_count = (depth_a - lca_depth) + (depth_b - lca_depth)

    # Determine level from the relationship between paths.
    #
    # L1 — same leaf group: paths are identical (both languages listed under
    #       the same terminal node, e.g. both in "anglo_frisian").
    # L2 — same branch, different leaf group: LCA is the *parent* of the
    #       leaf groups (e.g. eng in anglo_frisian, deu in high_german; LCA
    #       is west_germanic) OR LCA is one more level up within the same
    #       major branch.  Concretely: the paths diverge at the last or
    #       second-to-last position relative to the *shorter* path.
    # L3 — same top-level family, deeper divergence (e.g. eng vs fra both in
    #       indo_european but in different major branches).
    # L4 — different top-level families (already handled above).

    if pa == pb:
        # Same leaf group
        return (edge_count, "L1")

    # Classification based on how deep the shared ancestry is:
    #   L2 — same branch: they share at least 2 path elements
    #         e.g. both under [indo_european, germanic, ...]
    #         or both under [uralic, finnic, ...]
    #   L3 — same top-level family but different branches:
    #         they share only 1 path element (the super-family)
    #         e.g. [indo_european, germanic, ...] vs [indo_european, italic, ...]
    if lca_depth >= 2:
        return (edge_count, "L2")
    # lca_depth == 1: same top-level family, different branches
    return (edge_count, "L3")


def get_top_family(iso: str, lang_paths: dict[str, list[str]], family_map: dict[str, str]) -> str:
    """Return the top-level family name for *iso*.

    Tries the tree path first; falls back to family_map.
    """
    path = lang_paths.get(iso)
    if path:
        # The first element is the top-level family node; but we want the
        # *branch* name that corresponds to TOP_FAMILIES.  The branch is
        # typically the second element (e.g. "germanic" under "indo_european").
        # For top-level families like "uralic", the first element IS the family.
        if len(path) >= 2 and path[1] in _TOP_FAMILY_SET:
            return path[1]
        if path[0] in _TOP_FAMILY_SET:
            return path[0]
        # Search path for any matching family name
        for segment in path:
            if segment in _TOP_FAMILY_SET:
                return segment
        # Return the first path element as family
        return path[0] if path else "unknown"

    # Fallback: family_map
    fam = family_map.get(iso, "unknown")
    return FAMILY_ALIAS.get(fam, fam)


_TOP_FAMILY_SET = set(TOP_FAMILIES)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

LexEntry = tuple[str, str, str]  # (word, ipa, sca)


def load_lexicons(
    lexicons_dir: Path,
) -> dict[tuple[str, str], list[LexEntry]]:
    """Load all lexicon TSVs into ``{(iso, concept_id): [(word, ipa, sca), ...]}``.

    Entries with ``Concept_ID`` of ``"-"`` or empty are skipped.
    """
    lexicon: dict[tuple[str, str], list[LexEntry]] = defaultdict(list)
    files = sorted(lexicons_dir.glob("*.tsv"))
    total_entries = 0
    skipped_no_concept = 0

    for i, fp in enumerate(files, 1):
        iso = fp.stem
        if i % 100 == 0 or i == len(files):
            print(f"  Loading lexicon {i}/{len(files)} ({iso}) ...")
        try:
            with fp.open(encoding="utf-8") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    cid = row.get("Concept_ID", "").strip()
                    if cid in ("", "-"):
                        skipped_no_concept += 1
                        continue
                    word = row.get("Word", "").strip()
                    ipa = row.get("IPA", "").strip()
                    sca = row.get("SCA", "").strip()
                    if not word and not ipa and not sca:
                        continue
                    lexicon[(iso, cid)].append((word, ipa, sca))
                    total_entries += 1
        except Exception as exc:
            print(f"  WARNING: failed to read {fp.name}: {exc}", file=sys.stderr)

    print(f"  Loaded {total_entries:,} entries across {len(files)} lexicons "
          f"(skipped {skipped_no_concept:,} without concept).")
    return lexicon


def build_concept_index(
    lexicon: dict[tuple[str, str], list[LexEntry]],
) -> dict[str, list[str]]:
    """Map each concept_id to the list of ISO codes that have entries for it."""
    concept_langs: dict[str, set[str]] = defaultdict(set)
    for (iso, cid) in lexicon:
        concept_langs[cid].add(iso)
    return {cid: sorted(isos) for cid, isos in concept_langs.items()}


def load_cognate_pairs(path: Path) -> list[dict[str, str]]:
    """Load a cognate-pairs TSV file into a list of dicts."""
    rows: list[dict[str, str]] = []
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping.")
        return rows
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(dict(row))
    print(f"  Loaded {len(rows):,} pairs from {path.name}")
    return rows


# ---------------------------------------------------------------------------
# Pair record builder
# ---------------------------------------------------------------------------

def make_pair_record(
    lang_a: str,
    word_a: str,
    ipa_a: str,
    sca_a: str,
    lang_b: str,
    word_b: str,
    ipa_b: str,
    sca_b: str,
    concept_id: str,
    label: str,
    lang_paths: dict[str, list[str]],
    source: str = "lexicon",
    score_override: float | None = None,
) -> dict[str, str]:
    """Build a single output-row dict."""
    _, level = compute_distance(lang_a, lang_b, lang_paths)
    ts = get_timespan(lang_a, lang_b)
    if score_override is not None:
        score = score_override
    else:
        score = normalised_similarity(sca_a, sca_b) if sca_a and sca_b else 0.0
    return {
        "Lang_A": lang_a,
        "Word_A": word_a,
        "IPA_A": ipa_a,
        "SCA_A": sca_a,
        "Lang_B": lang_b,
        "Word_B": word_b,
        "IPA_B": ipa_b,
        "SCA_B": sca_b,
        "Concept_ID": concept_id,
        "Label": label,
        "Phylo_Dist": level,
        "Timespan": ts,
        "Score": f"{score:.4f}",
        "Source": source,
    }


# ---------------------------------------------------------------------------
# TSV writing helper
# ---------------------------------------------------------------------------

def write_pairs_tsv(path: Path, pairs: list[dict[str, str]]) -> None:
    """Write a list of pair dicts as a TSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        for row in pairs:
            writer.writerow(row)
    print(f"  Wrote {len(pairs):,} pairs to {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Step 3: True cognate pair generation
# ---------------------------------------------------------------------------

def generate_true_cognates(
    lexicon: dict[tuple[str, str], list[LexEntry]],
    concept_langs: dict[str, list[str]],
    lang_paths: dict[str, list[str]],
    family_map: dict[str, str],
    inherited_pairs: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Generate true cognate pairs at L1, L2, L3 levels.

    Returns (l1_pairs, l2_pairs, l3_pairs).
    """
    l1: list[dict[str, str]] = []
    l2: list[dict[str, str]] = []
    l3: list[dict[str, str]] = []

    buckets = {"L1": l1, "L2": l2, "L3": l3}
    thresholds = {"L1": 0.5, "L2": 0.5, "L3": 0.4}

    # Step 3a: incorporate expert pairs FIRST (they have priority)
    print("Step 3a: Adding expert cognate pairs ...")
    expert_added = 0
    for row in inherited_pairs:
        if all(len(b) >= PAIR_CAP for b in buckets.values()):
            break

        lang_a = row.get("Lang_A", "")
        lang_b = row.get("Lang_B", "")
        if not lang_a or not lang_b:
            continue

        _, level = compute_distance(lang_a, lang_b, lang_paths)
        if level not in buckets or len(buckets[level]) >= PAIR_CAP:
            continue

        cid = row.get("Concept_ID", "")
        ipa_a = row.get("IPA_A", "")
        ipa_b = row.get("IPA_B", "")
        word_a = row.get("Word_A", "")
        word_b = row.get("Word_B", "")
        sca_a = _lookup_sca(lexicon, lang_a, cid, word_a, ipa_a)
        sca_b = _lookup_sca(lexicon, lang_b, cid, word_b, ipa_b)

        score_str = row.get("Score", "0")
        try:
            score = float(score_str)
        except (ValueError, TypeError):
            score = 0.0

        if sca_a and sca_b:
            sca_sim = normalised_similarity(sca_a, sca_b)
        else:
            sca_sim = score

        rec = make_pair_record(
            lang_a, word_a, ipa_a, sca_a,
            lang_b, word_b, ipa_b, sca_b,
            cid, "true_cognate", lang_paths,
            source=row.get("Source", "expert"),
            score_override=sca_sim,
        )
        rec["Phylo_Dist"] = level
        buckets[level].append(rec)
        expert_added += 1

    print(f"  Added {expert_added:,} expert pairs "
          f"(L1={len(l1):,} L2={len(l2):,} L3={len(l3):,})")

    # Step 3b: Fill remaining slots from lexicon data
    print("Step 3b: Generating true cognates from lexicon data ...")

    concepts_processed = 0
    total_concepts = len(concept_langs)

    for cid, langs in concept_langs.items():
        concepts_processed += 1
        if concepts_processed % 500 == 0:
            print(f"  Concept {concepts_processed}/{total_concepts} "
                  f"(L1={len(l1):,} L2={len(l2):,} L3={len(l3):,})")

        # Early termination
        if all(len(b) >= PAIR_CAP for b in buckets.values()):
            break

        # Group languages by top-level family
        family_groups: dict[str, list[str]] = defaultdict(list)
        for iso in langs:
            fam = get_top_family(iso, lang_paths, family_map)
            family_groups[fam].append(iso)

        # Within each family, generate pairs
        for fam, fam_langs in family_groups.items():
            if len(fam_langs) < 2:
                continue

            # Sample pairs if too many languages
            if len(fam_langs) > 50:
                sampled = random.sample(fam_langs, 50)
            else:
                sampled = fam_langs

            pair_count_this_concept: dict[str, int] = {"L1": 0, "L2": 0, "L3": 0}

            for i in range(len(sampled)):
                for j in range(i + 1, len(sampled)):
                    iso_a = sampled[i]
                    iso_b = sampled[j]
                    if iso_a == iso_b:
                        continue

                    _, level = compute_distance(iso_a, iso_b, lang_paths)
                    if level not in buckets or len(buckets[level]) >= PAIR_CAP:
                        continue
                    if pair_count_this_concept.get(level, 0) >= MAX_PAIRS_PER_CONCEPT_PER_LEVEL:
                        continue

                    thresh = thresholds[level]
                    entries_a = lexicon.get((iso_a, cid), [])
                    entries_b = lexicon.get((iso_b, cid), [])
                    if not entries_a or not entries_b:
                        continue

                    # Pick one entry from each language (first with SCA)
                    ea = _pick_best_entry(entries_a)
                    eb = _pick_best_entry(entries_b)
                    if ea is None or eb is None:
                        continue

                    sca_sim = normalised_similarity(ea[2], eb[2]) if ea[2] and eb[2] else 0.0
                    if sca_sim < thresh:
                        continue

                    rec = make_pair_record(
                        iso_a, ea[0], ea[1], ea[2],
                        iso_b, eb[0], eb[1], eb[2],
                        cid, "true_cognate", lang_paths,
                        source="lexicon",
                        score_override=sca_sim,
                    )
                    rec["Phylo_Dist"] = level  # ensure correct level
                    buckets[level].append(rec)
                    pair_count_this_concept[level] = pair_count_this_concept.get(level, 0) + 1

    print(f"  From lexicon: L1={len(l1):,} L2={len(l2):,} L3={len(l3):,}")

    # Cap each bucket
    for level_name, bucket in buckets.items():
        if len(bucket) > PAIR_CAP:
            random.shuffle(bucket)
            buckets[level_name] = bucket[:PAIR_CAP]
            if level_name == "L1":
                l1[:] = buckets[level_name]
            elif level_name == "L2":
                l2[:] = buckets[level_name]
            else:
                l3[:] = buckets[level_name]

    print(f"  Final: L1={len(l1):,} L2={len(l2):,} L3={len(l3):,}")
    return l1, l2, l3


def _pick_best_entry(entries: list[LexEntry]) -> LexEntry | None:
    """Pick an entry preferring ones with non-empty SCA."""
    for e in entries:
        if e[2]:  # has SCA
            return e
    return entries[0] if entries else None


def _lookup_sca(
    lexicon: dict[tuple[str, str], list[LexEntry]],
    iso: str,
    concept_id: str,
    word: str,
    ipa: str = "",
) -> str:
    """Try to find the SCA encoding for a given word from the lexicon.

    Falls back to computing SCA on-the-fly from IPA if all lookups fail.
    """
    entries = lexicon.get((iso, concept_id), [])
    # Exact word match first
    for e in entries:
        if e[0] == word and e[2]:
            return e[2]
    # Any entry with SCA for this (iso, concept)
    for e in entries:
        if e[2]:
            return e[2]
    # Use the word index for cross-concept fallback (fast)
    sca = _word_sca_index.get((iso, word), "")
    if sca:
        return sca
    # Last resort: compute from IPA on-the-fly
    if ipa:
        return ipa_to_sound_class(ipa)
    return ""


# ---------------------------------------------------------------------------
# Step 4: False positives
# ---------------------------------------------------------------------------

def generate_false_positives(
    lexicon: dict[tuple[str, str], list[LexEntry]],
    concept_langs: dict[str, list[str]],
    lang_paths: dict[str, list[str]],
    family_map: dict[str, str],
    similarity_pairs: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Cross-family pairs with same concept and SCA similarity >= 0.5."""
    print("Step 4: Generating false positives ...")
    fps: list[dict[str, str]] = []

    # 4a: From lexicon — cross-family same-concept pairs
    concepts_processed = 0
    for cid, langs in concept_langs.items():
        if len(fps) >= PAIR_CAP:
            break
        concepts_processed += 1
        if concepts_processed % 500 == 0:
            print(f"  Concept {concepts_processed}/{len(concept_langs)} "
                  f"(fps={len(fps):,})")

        # Group by family
        family_groups: dict[str, list[str]] = defaultdict(list)
        for iso in langs:
            fam = get_top_family(iso, lang_paths, family_map)
            family_groups[fam].append(iso)

        families = list(family_groups.keys())
        if len(families) < 2:
            continue

        # Cross-family pairs
        pair_count = 0
        for fi in range(len(families)):
            for fj in range(fi + 1, len(families)):
                if len(fps) >= PAIR_CAP:
                    break
                if pair_count >= MAX_CROSS_FAMILY_PAIRS_PER_CONCEPT:
                    break

                fam_a_langs = family_groups[families[fi]]
                fam_b_langs = family_groups[families[fj]]

                # Sample one from each
                iso_a = random.choice(fam_a_langs)
                iso_b = random.choice(fam_b_langs)

                entries_a = lexicon.get((iso_a, cid), [])
                entries_b = lexicon.get((iso_b, cid), [])
                if not entries_a or not entries_b:
                    continue

                ea = _pick_best_entry(entries_a)
                eb = _pick_best_entry(entries_b)
                if ea is None or eb is None:
                    continue

                sca_sim = normalised_similarity(ea[2], eb[2]) if ea[2] and eb[2] else 0.0
                if sca_sim < 0.5:
                    continue

                rec = make_pair_record(
                    iso_a, ea[0], ea[1], ea[2],
                    iso_b, eb[0], eb[1], eb[2],
                    cid, "false_positive", lang_paths,
                    source="lexicon",
                    score_override=sca_sim,
                )
                fps.append(rec)
                pair_count += 1

    print(f"  From lexicon: {len(fps):,} false positives")

    # 4b: From similarity_pairs file
    sim_added = 0
    for row in similarity_pairs:
        if len(fps) >= PAIR_CAP:
            break
        lang_a = row.get("Lang_A", "")
        lang_b = row.get("Lang_B", "")
        if not lang_a or not lang_b:
            continue

        fam_a = get_top_family(lang_a, lang_paths, family_map)
        fam_b = get_top_family(lang_b, lang_paths, family_map)
        if fam_a == fam_b:
            continue  # only want cross-family

        cid = row.get("Concept_ID", "")
        sca_a = _lookup_sca(lexicon, lang_a, cid, row.get("Word_A", ""),
                            row.get("IPA_A", ""))
        sca_b = _lookup_sca(lexicon, lang_b, cid, row.get("Word_B", ""),
                            row.get("IPA_B", ""))

        score_str = row.get("Score", "0")
        try:
            score = float(score_str)
        except (ValueError, TypeError):
            score = 0.0

        if sca_a and sca_b:
            sca_sim = normalised_similarity(sca_a, sca_b)
        else:
            sca_sim = score

        if sca_sim < 0.5:
            continue

        rec = make_pair_record(
            lang_a, row.get("Word_A", ""), row.get("IPA_A", ""), sca_a,
            lang_b, row.get("Word_B", ""), row.get("IPA_B", ""), sca_b,
            cid, "false_positive", lang_paths,
            source=row.get("Source", "similarity"),
            score_override=sca_sim,
        )
        fps.append(rec)
        sim_added += 1

    print(f"  Added {sim_added:,} from similarity pairs")

    if len(fps) > PAIR_CAP:
        random.shuffle(fps)
        fps = fps[:PAIR_CAP]

    print(f"  Final false positives: {len(fps):,}")
    return fps


# ---------------------------------------------------------------------------
# Step 5: True negatives
# ---------------------------------------------------------------------------

def generate_true_negatives(
    lexicon: dict[tuple[str, str], list[LexEntry]],
    lang_paths: dict[str, list[str]],
    family_map: dict[str, str],
) -> list[dict[str, str]]:
    """Random cross-family pairs with different concepts and SCA sim < 0.3."""
    print("Step 5: Generating true negatives ...")
    negs: list[dict[str, str]] = []

    # Build index of all (iso, concept_id) keys for sampling
    all_keys = list(lexicon.keys())
    if len(all_keys) < 2:
        print("  Not enough lexicon entries for true negatives.")
        return negs

    attempts = 0
    while len(negs) < PAIR_CAP and attempts < TRUE_NEG_SAMPLE_ATTEMPTS:
        attempts += 1
        if attempts % 50_000 == 0:
            print(f"  Attempt {attempts:,}, negatives so far: {len(negs):,}")

        # Pick two random entries
        key_a = random.choice(all_keys)
        key_b = random.choice(all_keys)
        iso_a, cid_a = key_a
        iso_b, cid_b = key_b

        # Must be different concepts
        if cid_a == cid_b:
            continue
        # Must be different families
        fam_a = get_top_family(iso_a, lang_paths, family_map)
        fam_b = get_top_family(iso_b, lang_paths, family_map)
        if fam_a == fam_b:
            continue
        # Must be different languages
        if iso_a == iso_b:
            continue

        entries_a = lexicon[key_a]
        entries_b = lexicon[key_b]
        ea = _pick_best_entry(entries_a)
        eb = _pick_best_entry(entries_b)
        if ea is None or eb is None:
            continue
        if not ea[2] or not eb[2]:
            continue

        sca_sim = normalised_similarity(ea[2], eb[2])
        if sca_sim >= 0.4:
            continue

        # Use concept_a for the record (arbitrary; both concepts are different)
        rec = make_pair_record(
            iso_a, ea[0], ea[1], ea[2],
            iso_b, eb[0], eb[1], eb[2],
            f"{cid_a} / {cid_b}", "true_negative", lang_paths,
            source="random_sample",
            score_override=sca_sim,
        )
        negs.append(rec)

    print(f"  Generated {len(negs):,} true negatives in {attempts:,} attempts")
    return negs


# ---------------------------------------------------------------------------
# Step 6: Borrowings
# ---------------------------------------------------------------------------

def generate_borrowings(
    borrowing_pairs: list[dict[str, str]],
    lexicon: dict[tuple[str, str], list[LexEntry]],
    lang_paths: dict[str, list[str]],
) -> list[dict[str, str]]:
    """Process borrowing pairs from WOLD."""
    print("Step 6: Processing borrowing pairs ...")
    rows: list[dict[str, str]] = []

    for row in borrowing_pairs:
        lang_a = row.get("Lang_A", "")
        lang_b = row.get("Lang_B", "")
        if not lang_a or not lang_b:
            continue

        cid = row.get("Concept_ID", "")
        word_a = row.get("Word_A", "")
        word_b = row.get("Word_B", "")
        ipa_a = row.get("IPA_A", "")
        ipa_b = row.get("IPA_B", "")

        sca_a = _lookup_sca(lexicon, lang_a, cid, word_a, ipa_a)
        sca_b = _lookup_sca(lexicon, lang_b, cid, word_b, ipa_b)

        score_str = row.get("Score", "0")
        try:
            score = float(score_str)
        except (ValueError, TypeError):
            score = 0.0

        if sca_a and sca_b:
            sca_sim = normalised_similarity(sca_a, sca_b)
        else:
            sca_sim = score

        rec = make_pair_record(
            lang_a, word_a, ipa_a, sca_a,
            lang_b, word_b, ipa_b, sca_b,
            cid, "borrowing", lang_paths,
            source=row.get("Source", "wold"),
            score_override=sca_sim,
        )
        rows.append(rec)

    print(f"  Processed {len(rows):,} borrowing pairs")
    return rows


# ---------------------------------------------------------------------------
# Step 7: Religious subsets
# ---------------------------------------------------------------------------

def generate_religious_pairs(
    all_pairs: list[dict[str, str]],
    lexicon: dict[tuple[str, str], list[LexEntry]],
    concept_langs: dict[str, list[str]],
    lang_paths: dict[str, list[str]],
    family_map: dict[str, str],
) -> tuple[
    list[dict[str, str]],
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
]:
    """Filter existing pairs for religious concepts and generate additional ones.

    Returns (all_religious_pairs, per_family_dict, subdomain_dict).
    """
    print("Step 7: Building religious concept subsets ...")

    # 7a: Filter all existing pairs — SKIP compound concept IDs (true negatives)
    religious: list[dict[str, str]] = []
    skipped_compound = 0
    for rec in all_pairs:
        cid = rec.get("Concept_ID", "")
        # Skip compound concept IDs (true negatives use "cid_a / cid_b")
        if "/" in cid:
            skipped_compound += 1
            continue
        if is_religious(cid):
            religious.append(rec)

    print(f"  Filtered {len(religious):,} religious pairs from existing data "
          f"(skipped {skipped_compound:,} compound IDs)")

    # 7b: Generate additional within-family and cross-family pairs for
    #     religious concepts not already covered
    religious_concepts = [c for c in concept_langs if is_religious(c)]
    print(f"  Found {len(religious_concepts)} religious concepts in lexicon")

    additional = 0
    for cid in religious_concepts:
        langs = concept_langs[cid]
        if len(langs) < 2:
            continue

        # Group by family
        family_groups: dict[str, list[str]] = defaultdict(list)
        for iso in langs:
            fam = get_top_family(iso, lang_paths, family_map)
            family_groups[fam].append(iso)

        # Within-family pairs
        for fam, fam_langs in family_groups.items():
            if len(fam_langs) < 2:
                continue
            sample_size = min(len(fam_langs), 20)
            sampled = random.sample(fam_langs, sample_size) if len(fam_langs) > sample_size else fam_langs
            for i in range(len(sampled)):
                for j in range(i + 1, len(sampled)):
                    iso_a, iso_b = sampled[i], sampled[j]
                    entries_a = lexicon.get((iso_a, cid), [])
                    entries_b = lexicon.get((iso_b, cid), [])
                    if not entries_a or not entries_b:
                        continue
                    ea = _pick_best_entry(entries_a)
                    eb = _pick_best_entry(entries_b)
                    if ea is None or eb is None:
                        continue
                    sca_sim = normalised_similarity(ea[2], eb[2]) if ea[2] and eb[2] else 0.0
                    _, level = compute_distance(iso_a, iso_b, lang_paths)
                    label = "true_cognate" if level in ("L1", "L2", "L3") and sca_sim >= 0.4 else "false_positive"
                    rec = make_pair_record(
                        iso_a, ea[0], ea[1], ea[2],
                        iso_b, eb[0], eb[1], eb[2],
                        cid, label, lang_paths,
                        source="religious_gen",
                        score_override=sca_sim,
                    )
                    religious.append(rec)
                    additional += 1

        # Cross-family pairs (limited)
        families = list(family_groups.keys())
        if len(families) >= 2:
            cross_count = 0
            for fi in range(len(families)):
                for fj in range(fi + 1, len(families)):
                    if cross_count >= 10:
                        break
                    iso_a = random.choice(family_groups[families[fi]])
                    iso_b = random.choice(family_groups[families[fj]])
                    entries_a = lexicon.get((iso_a, cid), [])
                    entries_b = lexicon.get((iso_b, cid), [])
                    if not entries_a or not entries_b:
                        continue
                    ea = _pick_best_entry(entries_a)
                    eb = _pick_best_entry(entries_b)
                    if ea is None or eb is None:
                        continue
                    sca_sim = normalised_similarity(ea[2], eb[2]) if ea[2] and eb[2] else 0.0
                    rec = make_pair_record(
                        iso_a, ea[0], ea[1], ea[2],
                        iso_b, eb[0], eb[1], eb[2],
                        cid, "false_positive", lang_paths,
                        source="religious_gen",
                        score_override=sca_sim,
                    )
                    religious.append(rec)
                    additional += 1
                    cross_count += 1

    print(f"  Generated {additional:,} additional religious pairs")
    print(f"  Total religious pairs: {len(religious):,}")

    # Build per-family subsets
    per_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for rec in religious:
        fam_a = get_top_family(rec["Lang_A"], lang_paths, family_map)
        fam_b = get_top_family(rec["Lang_B"], lang_paths, family_map)
        if fam_a in _TOP_FAMILY_SET:
            per_family[fam_a].append(rec)
        if fam_b in _TOP_FAMILY_SET and fam_b != fam_a:
            per_family[fam_b].append(rec)

    # Build sub-domain subsets
    subdomain_pairs: dict[str, list[dict[str, str]]] = {}
    for sd_name, sd_concepts in RELIGIOUS_SUBDOMAINS.items():
        sd_list: list[dict[str, str]] = []
        for rec in religious:
            cid = rec.get("Concept_ID", "")
            if _in_subdomain(cid, sd_concepts):
                sd_list.append(rec)
        subdomain_pairs[sd_name] = sd_list
        print(f"  Sub-domain {sd_name}: {len(sd_list):,} pairs")

    return religious, dict(per_family), subdomain_pairs


# ---------------------------------------------------------------------------
# Step 8: Timespan stratification
# ---------------------------------------------------------------------------

def stratify_by_timespan(
    all_pairs: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    """Split all pairs into timespan buckets."""
    print("Step 8: Stratifying by timespan ...")
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for rec in all_pairs:
        ts = rec.get("Timespan", "modern_modern")
        buckets[ts].append(rec)
    for ts, pairs in sorted(buckets.items()):
        print(f"  {ts}: {len(pairs):,}")
    return dict(buckets)


# ---------------------------------------------------------------------------
# Step 9: Per-family sets
# ---------------------------------------------------------------------------

def stratify_by_family(
    all_pairs: list[dict[str, str]],
    lang_paths: dict[str, list[str]],
    family_map: dict[str, str],
) -> dict[str, list[dict[str, str]]]:
    """Split pairs into per-family buckets for TOP_FAMILIES."""
    print("Step 9: Stratifying by family ...")
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for rec in all_pairs:
        fam_a = get_top_family(rec["Lang_A"], lang_paths, family_map)
        fam_b = get_top_family(rec["Lang_B"], lang_paths, family_map)
        # Include pair if either language belongs to a top family
        if fam_a == fam_b and fam_a in _TOP_FAMILY_SET:
            buckets[fam_a].append(rec)
        else:
            # Cross-family pair: include in both relevant families
            if fam_a in _TOP_FAMILY_SET:
                buckets[fam_a].append(rec)
            if fam_b in _TOP_FAMILY_SET:
                buckets[fam_b].append(rec)

    for fam in TOP_FAMILIES:
        count = len(buckets.get(fam, []))
        print(f"  {fam}: {count:,}")
    return dict(buckets)


# ---------------------------------------------------------------------------
# Step 10: Statistics
# ---------------------------------------------------------------------------

def write_stats(
    output_dir: Path,
    l1: list, l2: list, l3: list,
    fps: list, negs: list, borrows: list,
    religious: list,
    timespan_buckets: dict[str, list],
    family_buckets: dict[str, list],
    religious_by_family: dict[str, list],
) -> None:
    """Write validation_stats.tsv with summary counts."""
    print("Step 10: Writing statistics ...")
    stats_path = output_dir / "validation_stats.tsv"
    rows: list[tuple[str, str, str]] = []

    rows.append(("Category", "Subset", "Count"))
    rows.append(("true_cognates", "L1", str(len(l1))))
    rows.append(("true_cognates", "L2", str(len(l2))))
    rows.append(("true_cognates", "L3", str(len(l3))))
    rows.append(("true_cognates", "total", str(len(l1) + len(l2) + len(l3))))
    rows.append(("false_positives", "all", str(len(fps))))
    rows.append(("true_negatives", "all", str(len(negs))))
    rows.append(("borrowings", "all", str(len(borrows))))
    rows.append(("religious", "all", str(len(religious))))

    for ts_name in ["ancient_ancient", "ancient_modern", "medieval_modern", "modern_modern"]:
        count = len(timespan_buckets.get(ts_name, []))
        rows.append(("timespan", ts_name, str(count)))

    for fam in TOP_FAMILIES:
        count = len(family_buckets.get(fam, []))
        rows.append(("per_family", fam, str(count)))

    for fam in sorted(religious_by_family.keys()):
        count = len(religious_by_family[fam])
        rows.append(("religious_by_family", fam, str(count)))

    total_all = len(l1) + len(l2) + len(l3) + len(fps) + len(negs) + len(borrows)
    rows.append(("total", "all_pairs", str(total_all)))

    # Compute label distribution
    label_counts: dict[str, int] = defaultdict(int)
    for pairs_list in [l1, l2, l3, fps, negs, borrows]:
        for rec in pairs_list:
            label_counts[rec.get("Label", "unknown")] += 1
    for label, count in sorted(label_counts.items()):
        rows.append(("label_distribution", label, str(count)))

    # Compute era distribution
    era_counts: dict[str, int] = defaultdict(int)
    for pairs_list in [l1, l2, l3, fps, negs, borrows]:
        for rec in pairs_list:
            ts = rec.get("Timespan", "unknown")
            era_counts[ts] += 1
    for era, count in sorted(era_counts.items()):
        rows.append(("era_distribution", era, str(count)))

    # Unique languages
    all_langs: set[str] = set()
    for pairs_list in [l1, l2, l3, fps, negs, borrows]:
        for rec in pairs_list:
            all_langs.add(rec.get("Lang_A", ""))
            all_langs.add(rec.get("Lang_B", ""))
    all_langs.discard("")
    rows.append(("coverage", "unique_languages", str(len(all_langs))))

    # Unique concepts
    all_concepts: set[str] = set()
    for pairs_list in [l1, l2, l3, fps, negs, borrows]:
        for rec in pairs_list:
            cid = rec.get("Concept_ID", "")
            if cid:
                all_concepts.add(cid)
    rows.append(("coverage", "unique_concepts", str(len(all_concepts))))

    # Unique families
    family_set: set[str] = set()
    for lang in all_langs:
        fam = get_top_family(lang, _global_lang_paths, _global_family_map)
        family_set.add(fam)
    rows.append(("coverage", "unique_families", str(len(family_set))))

    with stats_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        for row in rows:
            writer.writerow(row)

    print(f"  Wrote stats to {stats_path.relative_to(REPO_ROOT)}")
    print(f"  Total pairs across main sets: {total_all:,}")
    print(f"  Unique languages: {len(all_langs):,}")
    print(f"  Unique concepts: {len(all_concepts):,}")


# Global references set in main() for use in stats helper
_global_lang_paths: dict[str, list[str]] = {}
_global_family_map: dict[str, str] = {}

# Fast word→SCA index: (iso, word) → sca, built in main()
_word_sca_index: dict[tuple[str, str], str] = {}


def build_word_sca_index(
    lexicon: dict[tuple[str, str], list[LexEntry]],
) -> dict[tuple[str, str], str]:
    """Build (iso, word) → SCA index for fast cross-concept lookups."""
    idx: dict[tuple[str, str], str] = {}
    for (iso, _cid), entries in lexicon.items():
        for word, _ipa, sca in entries:
            if sca and (iso, word) not in idx:
                idx[(iso, word)] = sca
    return idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _global_lang_paths, _global_family_map

    random.seed(SEED)
    print("=" * 70)
    print("build_validation_sets.py — Stratified ML Validation Datasets")
    print("=" * 70)

    # ----- Step 1: Build and resolve phylogenetic tree --------------------
    print("\nStep 1: Building phylogenetic tree ...")

    if not FAMILY_MAP_PATH.exists():
        print(f"ERROR: family_map.json not found at {FAMILY_MAP_PATH}", file=sys.stderr)
        sys.exit(1)

    with FAMILY_MAP_PATH.open(encoding="utf-8") as fh:
        family_map: dict[str, str] = json.load(fh)
    print(f"  Loaded family_map.json: {len(family_map):,} languages")

    raw_tree = build_raw_tree()
    tree = resolve_tree(raw_tree, family_map)
    lang_paths = build_lang_paths(tree)
    print(f"  Tree covers {len(lang_paths):,} languages")

    # Store globals for stats
    _global_lang_paths = lang_paths
    _global_family_map = family_map

    # Write phylo_tree.json
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tree_path = OUTPUT_DIR / "phylo_tree.json"
    with tree_path.open("w", encoding="utf-8") as fh:
        json.dump(tree, fh, indent=2, ensure_ascii=False)
    print(f"  Wrote {tree_path.relative_to(REPO_ROOT)}")

    # Quick distance sanity checks
    for pair, expected in [
        (("eng", "ang"), "L1"),
        (("eng", "deu"), "L2"),
        (("eng", "fra"), "L3"),
        (("eng", "fin"), "L4"),
    ]:
        _, level = compute_distance(pair[0], pair[1], lang_paths)
        status = "OK" if level == expected else f"MISMATCH (got {level})"
        print(f"  Distance {pair[0]}-{pair[1]}: {level} ({status})")

    # ----- Step 2: Load data ----------------------------------------------
    print("\nStep 2: Loading data ...")

    if not LEXICONS_DIR.exists():
        print(f"ERROR: Lexicons directory not found: {LEXICONS_DIR}", file=sys.stderr)
        sys.exit(1)

    lexicon = load_lexicons(LEXICONS_DIR)
    concept_langs = build_concept_index(lexicon)
    print(f"  {len(concept_langs):,} unique concepts across all lexicons")

    # Build fast word→SCA index for cross-concept lookups
    global _word_sca_index
    _word_sca_index = build_word_sca_index(lexicon)
    print(f"  Built word-SCA index: {len(_word_sca_index):,} entries")

    inherited_path = COGNATE_DIR / "cognate_pairs_inherited.tsv"
    borrowing_path = COGNATE_DIR / "cognate_pairs_borrowing.tsv"
    similarity_path = COGNATE_DIR / "cognate_pairs_similarity.tsv"

    # For very large files, stream-sample instead of loading everything
    print("  Loading cognate pair files (may take a moment for large files) ...")
    inherited_pairs = _load_cognate_pairs_sampled(inherited_path, max_rows=200_000)
    borrowing_pairs = load_cognate_pairs(borrowing_path)
    similarity_pairs = load_cognate_pairs(similarity_path)

    # ----- Step 3: True cognates ------------------------------------------
    print("\nStep 3: Generating true cognate pairs ...")
    l1, l2, l3 = generate_true_cognates(
        lexicon, concept_langs, lang_paths, family_map, inherited_pairs,
    )

    write_pairs_tsv(OUTPUT_DIR / "true_cognates_L1.tsv", l1)
    write_pairs_tsv(OUTPUT_DIR / "true_cognates_L2.tsv", l2)
    write_pairs_tsv(OUTPUT_DIR / "true_cognates_L3.tsv", l3)

    # ----- Step 4: False positives ----------------------------------------
    print("\nStep 4: Generating false positives ...")
    fps = generate_false_positives(
        lexicon, concept_langs, lang_paths, family_map, similarity_pairs,
    )
    write_pairs_tsv(OUTPUT_DIR / "false_positives.tsv", fps)

    # ----- Step 5: True negatives -----------------------------------------
    print("\nStep 5: Generating true negatives ...")
    negs = generate_true_negatives(lexicon, lang_paths, family_map)
    write_pairs_tsv(OUTPUT_DIR / "true_negatives.tsv", negs)

    # ----- Step 6: Borrowings --------------------------------------------
    print("\nStep 6: Processing borrowings ...")
    borrows = generate_borrowings(borrowing_pairs, lexicon, lang_paths)
    write_pairs_tsv(OUTPUT_DIR / "borrowings.tsv", borrows)

    # ----- Collect all pairs for downstream splits ------------------------
    all_pairs: list[dict[str, str]] = l1 + l2 + l3 + fps + negs + borrows

    # ----- Step 7: Religious subsets --------------------------------------
    print("\nStep 7: Building religious subsets ...")
    religious, religious_by_family, subdomain_pairs = generate_religious_pairs(
        all_pairs, lexicon, concept_langs, lang_paths, family_map,
    )

    # Write to religious/ subdirectory
    rel_dir = OUTPUT_DIR / "religious"
    rel_dir.mkdir(parents=True, exist_ok=True)
    write_pairs_tsv(rel_dir / "all_pairs.tsv", religious)

    # Split by label
    rel_true_cog = [r for r in religious if r.get("Label") == "true_cognate"]
    rel_false_pos = [r for r in religious if r.get("Label") == "false_positive"]
    rel_borrowings = [r for r in religious if r.get("Label") == "borrowing"]
    write_pairs_tsv(rel_dir / "true_cognates.tsv", rel_true_cog)
    write_pairs_tsv(rel_dir / "false_positives.tsv", rel_false_pos)
    write_pairs_tsv(rel_dir / "borrowings.tsv", rel_borrowings)

    # Sub-domain files
    for sd_name, sd_pairs in sorted(subdomain_pairs.items()):
        write_pairs_tsv(rel_dir / f"{sd_name}.tsv", sd_pairs)

    # Per-family under religious/by_family/
    rel_family_dir = rel_dir / "by_family"
    rel_family_dir.mkdir(parents=True, exist_ok=True)
    for fam, fam_pairs in sorted(religious_by_family.items()):
        write_pairs_tsv(rel_family_dir / f"{fam}.tsv", fam_pairs)

    # Clean up old flat files if they exist
    old_religious = OUTPUT_DIR / "religious_pairs.tsv"
    if old_religious.exists():
        old_religious.unlink()
        print(f"  Removed old {old_religious.relative_to(REPO_ROOT)}")
    old_rel_family = OUTPUT_DIR / "religious_by_family"
    if old_rel_family.exists() and old_rel_family.is_dir():
        import shutil
        shutil.rmtree(old_rel_family)
        print(f"  Removed old {old_rel_family.relative_to(REPO_ROOT)}")

    # ----- Step 8: Timespan stratification --------------------------------
    print("\nStep 8: Stratifying by timespan ...")
    timespan_buckets = stratify_by_timespan(all_pairs)
    for ts_name in ["ancient_ancient", "ancient_modern", "medieval_modern", "modern_modern"]:
        ts_pairs = timespan_buckets.get(ts_name, [])
        write_pairs_tsv(OUTPUT_DIR / f"timespan_{ts_name}.tsv", ts_pairs)

    # ----- Step 9: Per-family sets ----------------------------------------
    print("\nStep 9: Writing per-family sets ...")
    family_buckets = stratify_by_family(all_pairs, lang_paths, family_map)

    per_family_dir = OUTPUT_DIR / "per_family"
    per_family_dir.mkdir(parents=True, exist_ok=True)
    for fam in TOP_FAMILIES:
        fam_pairs = family_buckets.get(fam, [])
        write_pairs_tsv(per_family_dir / f"{fam}.tsv", fam_pairs)

    # ----- Step 10: Statistics --------------------------------------------
    print("\nStep 10: Writing statistics ...")
    write_stats(
        OUTPUT_DIR,
        l1, l2, l3,
        fps, negs, borrows,
        religious,
        timespan_buckets,
        family_buckets,
        religious_by_family,
    )

    print("\n" + "=" * 70)
    print("Done!  Output written to:")
    print(f"  {OUTPUT_DIR.relative_to(REPO_ROOT)}/")
    print("=" * 70)


def _load_cognate_pairs_sampled(
    path: Path,
    max_rows: int = 200_000,
) -> list[dict[str, str]]:
    """Load cognate pairs with reservoir sampling for very large files.

    For files with millions of rows, we use reservoir sampling to get a
    representative sample of at most *max_rows* entries.
    """
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping.")
        return []

    # First pass: count rows to decide whether to sample
    total = 0
    with path.open(encoding="utf-8") as fh:
        header = fh.readline()  # skip header
        if not header:
            return []
        for _ in fh:
            total += 1

    print(f"  {path.name}: {total:,} rows total", end="")

    if total <= max_rows:
        # Small enough — load everything
        print(" (loading all)")
        return load_cognate_pairs(path)

    # Reservoir sampling
    print(f" (sampling {max_rows:,})")
    reservoir: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for i, row in enumerate(reader):
            if i < max_rows:
                reservoir.append(dict(row))
            else:
                j = random.randint(0, i)
                if j < max_rows:
                    reservoir[j] = dict(row)

    print(f"  Sampled {len(reservoir):,} pairs from {path.name}")
    return reservoir


if __name__ == "__main__":
    main()
