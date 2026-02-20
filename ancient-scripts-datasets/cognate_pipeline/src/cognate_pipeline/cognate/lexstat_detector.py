"""LingPy LexStat wrapper for cognate detection."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from cognate_pipeline.cognate.models import CognateLink, CognateSet, CognateSetMember
from cognate_pipeline.normalise.models import NormalisedLexeme

logger = logging.getLogger(__name__)


class LexStatDetector:
    """Wraps LingPy's LexStat for cognate detection.

    Requires the `lingpy` optional dependency.
    """

    def detect(
        self,
        lexemes: list[NormalisedLexeme],
        threshold: float = 0.5,
    ) -> list[CognateLink]:
        """Run LexStat cognate detection on the given lexemes."""
        try:
            from lingpy import LexStat
        except ImportError:
            raise ImportError(
                "lingpy is required for LexStat detection. "
                "Install with: pip install cognate-pipeline[lingpy]"
            )

        # Build LingPy-format TSV
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write("ID\tDOCULECT\tCONCEPT\tIPA\tTOKENS\n")
            id_map: dict[int, str] = {}
            for idx, lex in enumerate(lexemes, start=1):
                tokens = " ".join(list(lex.ipa_canonical)) if lex.ipa_canonical else lex.form
                tmp.write(
                    f"{idx}\t{lex.language_id}\t{lex.concept_id}\t"
                    f"{lex.ipa_canonical or lex.form}\t{tokens}\n"
                )
                id_map[idx] = lex.id
            tmp_path = tmp.name

        try:
            lex_stat = LexStat(tmp_path)
            lex_stat.get_scorer(runs=100)
            lex_stat.cluster(method="lexstat", threshold=threshold)

            # Extract cognate links from clustering results
            links: list[CognateLink] = []
            cogid_col = lex_stat.header.get("cogid", lex_stat.header.get("lexstatid"))
            if cogid_col is None:
                logger.warning("No cognate ID column found in LexStat output")
                return links

            # Group by cogid to find pairs
            from collections import defaultdict

            cogid_groups: dict[int, list[int]] = defaultdict(list)
            for idx in lex_stat:
                cogid = lex_stat[idx, cogid_col]
                cogid_groups[cogid].append(idx)

            from itertools import combinations

            for cogid, members in cogid_groups.items():
                if len(members) < 2:
                    continue
                concept = lex_stat[members[0], "concept"]
                for a_idx, b_idx in combinations(members, 2):
                    id_a = id_map[a_idx]
                    id_b = id_map[b_idx]
                    if id_a > id_b:
                        id_a, id_b = id_b, id_a
                    links.append(
                        CognateLink(
                            lexeme_id_a=id_a,
                            lexeme_id_b=id_b,
                            concept_id=concept,
                            relationship_type="cognate_candidate",
                            score=1.0,
                            method="lexstat",
                            threshold_used=threshold,
                            evidence={"cogid": cogid},
                        )
                    )

            return links
        finally:
            Path(tmp_path).unlink(missing_ok=True)
