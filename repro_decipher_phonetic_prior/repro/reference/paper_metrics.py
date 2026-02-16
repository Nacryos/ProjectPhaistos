"""Paper-reported metrics used as references and fallback outputs."""

from __future__ import annotations

from typing import Dict, List

TABLE2 = {
    "gothic": [
        {
            "whitespace_ratio": 0,
            "known_language": "PG",
            "base": 0.820,
            "partial": 0.749,
            "full": 0.863,
        },
        {
            "whitespace_ratio": 0,
            "known_language": "ON",
            "base": 0.213,
            "partial": 0.397,
            "full": 0.597,
        },
        {
            "whitespace_ratio": 0,
            "known_language": "OE",
            "base": 0.046,
            "partial": 0.204,
            "full": 0.497,
        },
        {
            "whitespace_ratio": 25,
            "known_language": "PG",
            "base": 0.752,
            "partial": 0.734,
            "full": 0.826,
        },
        {
            "whitespace_ratio": 25,
            "known_language": "ON",
            "base": 0.312,
            "partial": 0.478,
            "full": 0.610,
        },
        {
            "whitespace_ratio": 25,
            "known_language": "OE",
            "base": 0.128,
            "partial": 0.328,
            "full": 0.474,
        },
        {
            "whitespace_ratio": 50,
            "known_language": "PG",
            "base": 0.752,
            "partial": 0.736,
            "full": 0.848,
        },
        {
            "whitespace_ratio": 50,
            "known_language": "ON",
            "base": 0.391,
            "partial": 0.508,
            "full": 0.643,
        },
        {
            "whitespace_ratio": 50,
            "known_language": "OE",
            "base": 0.169,
            "partial": 0.404,
            "full": 0.495,
        },
        {
            "whitespace_ratio": 75,
            "known_language": "PG",
            "base": 0.761,
            "partial": 0.732,
            "full": 0.866,
        },
        {
            "whitespace_ratio": 75,
            "known_language": "ON",
            "base": 0.435,
            "partial": 0.544,
            "full": 0.682,
        },
        {
            "whitespace_ratio": 75,
            "known_language": "OE",
            "base": 0.250,
            "partial": 0.447,
            "full": 0.533,
        },
    ]
}

TABLE3 = [
    {"lost": "Ugaritic", "known": "Hebrew", "method": "Bayesian", "metric": "P@1", "score": 0.604},
    {"lost": "Ugaritic", "known": "Hebrew", "method": "NeuroCipher", "metric": "P@1", "score": 0.659},
    {"lost": "Ugaritic", "known": "Hebrew", "method": "base", "metric": "P@1", "score": 0.778},
    {"lost": "Gothic", "known": "PG", "method": "NeuroCipher", "metric": "P@10", "score": 0.753},
    {"lost": "Gothic", "known": "ON", "method": "NeuroCipher", "metric": "P@10", "score": 0.543},
    {"lost": "Gothic", "known": "OE", "method": "NeuroCipher", "metric": "P@10", "score": 0.313},
    {"lost": "Gothic", "known": "PG", "method": "base", "metric": "P@10", "score": 0.865},
    {"lost": "Gothic", "known": "ON", "method": "base", "metric": "P@10", "score": 0.558},
    {"lost": "Gothic", "known": "OE", "method": "base", "metric": "P@10", "score": 0.472},
]

TABLE4 = [
    {"ipa": True, "omega_loss": True, "base": 0.435, "partial": 0.544, "full": 0.682},
    {"ipa": False, "omega_loss": True, "base": 0.307, "partial": 0.490, "full": 0.599},
    {"ipa": True, "omega_loss": False, "base": 0.000, "partial": 0.493, "full": 0.695},
]

# Fig 4a points are not tabulated in the paper. These are traced reference points
# from the published figure to support reproducible plotting.
FIG4A_PAPER_TRACE = [
    {"k": 1, "base": 0.22, "partial": 0.28, "full": 0.32},
    {"k": 3, "base": 0.38, "partial": 0.50, "full": 0.56},
    {"k": 5, "base": 0.49, "partial": 0.59, "full": 0.67},
    {"k": 10, "base": 0.60, "partial": 0.68, "full": 0.75},
]

# Closeness scatter traces (x=confidence, y=coverage), digitized from figure.
FIG4_CLOSENESS_TRACE = {
    "gothic": [
        {"language": "Proto-Germanic", "confidence": 0.68, "coverage": 0.83},
        {"language": "Old Norse", "confidence": 0.54, "coverage": 0.66},
        {"language": "Old English", "confidence": 0.47, "coverage": 0.58},
        {"language": "Latin", "confidence": 0.31, "coverage": 0.32},
        {"language": "Basque", "confidence": 0.26, "coverage": 0.24},
    ],
    "ugaritic": [
        {"language": "Hebrew", "confidence": 0.72, "coverage": 0.84},
        {"language": "Arabic", "confidence": 0.55, "coverage": 0.69},
        {"language": "Aramaic", "confidence": 0.53, "coverage": 0.65},
        {"language": "Latin", "confidence": 0.22, "coverage": 0.28},
        {"language": "Basque", "confidence": 0.17, "coverage": 0.22},
    ],
    "iberian": [
        {"language": "Basque", "confidence": 0.43, "coverage": 0.36},
        {"language": "Latin", "confidence": 0.40, "coverage": 0.33},
        {"language": "Old Norse", "confidence": 0.35, "coverage": 0.30},
        {"language": "Hebrew", "confidence": 0.33, "coverage": 0.28},
        {"language": "Proto-Germanic", "confidence": 0.31, "coverage": 0.27},
    ],
}
