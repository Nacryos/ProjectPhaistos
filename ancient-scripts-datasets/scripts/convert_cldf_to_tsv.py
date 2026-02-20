#!/usr/bin/env python3
"""Convert CLDF lexical databases to validation TSV files.

Reads NorthEuraLex, WOLD, ABVD, and sinotibetan data, filters for
target languages and concepts, extracts IPA, and writes one TSV per
language-family branch.

Dependencies: only Python standard library (csv, json, pathlib, re).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---- Paths ----
ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "sources"
VALIDATION = ROOT / "data" / "validation"
SCRIPTS = ROOT / "scripts"

# ---- Expanded concept list (Concepticon ID -> our concept_id + metadata) ----
# Format: concepticon_id -> (concept_id, category, english_gloss)
CONCEPTS = {
    # Body (existing)
    "1256": ("head", "body", "head"),
    "1248": ("eye", "body", "eye"),
    "1247": ("ear", "body", "ear"),
    "674":  ("mouth", "body", "mouth"),
    "1277": ("hand", "body", "hand"),
    "1301": ("foot", "body", "foot"),
    "1223": ("heart", "body", "heart"),
    "946":  ("blood", "body", "blood"),
    # Body (new)
    "1380": ("tooth", "body", "tooth"),
    "1205": ("tongue", "body", "tongue"),
    "1394": ("bone", "body", "bone"),
    "763":  ("skin", "body", "skin"),
    "1251": ("belly", "body", "belly"),
    "1333": ("neck", "body", "neck"),
    "1371": ("knee", "body", "knee"),
    "1221": ("nose", "body", "nose"),
    "1209": ("hair", "body", "hair"),
    "1258": ("nail", "body", "nail"),
    "1237": ("breast", "body", "breast"),
    # Kinship (existing)
    "1217": ("father", "kinship", "father"),
    "1216": ("mother", "kinship", "mother"),
    "1620": ("son", "kinship", "son"),
    "1357": ("daughter", "kinship", "daughter"),
    "1262": ("brother", "kinship", "brother"),
    # Kinship (new)
    "1199": ("wife", "kinship", "wife"),
    "1200": ("husband", "kinship", "husband"),
    "683":  ("person", "kinship", "person"),
    "1240": ("sister", "kinship", "sister"),
    "1441": ("child", "kinship", "child"),
    # Nature (existing)
    "948":  ("water", "nature", "water"),
    "221":  ("fire", "nature", "fire"),
    "1343": ("sun", "nature", "sun"),
    "1313": ("moon", "nature", "moon"),
    "1430": ("star", "nature", "star"),
    "1228": ("earth", "nature", "earth"),
    "639":  ("mountain", "nature", "mountain"),
    "666":  ("river", "nature", "river"),
    "857":  ("stone", "nature", "stone"),
    "906":  ("tree", "nature", "tree"),
    # Nature (new)
    "1489": ("cloud", "nature", "cloud"),
    "329":  ("rain", "nature", "rain"),
    "960":  ("wind", "nature", "wind"),
    "1732": ("sky", "nature", "sky"),
    "1474": ("sea", "nature", "sea"),
    "671":  ("sand", "nature", "sand"),
    "1233": ("night", "nature", "night"),
    "778":  ("smoke", "nature", "smoke"),
    "617":  ("ice", "nature", "ice"),
    # Animal (existing)
    "615":  ("horse", "animal", "horse"),
    "2009": ("dog", "animal", "dog"),
    "227":  ("fish", "animal", "fish"),
    "937":  ("bird", "animal", "bird"),
    "1169": ("ox", "animal", "ox"),
    # Animal (new)
    "730":  ("snake", "animal", "snake"),
    "1194": ("louse", "animal", "louse"),
    "1219": ("worm", "animal", "worm"),
    "744":  ("egg", "animal", "egg"),
    "1504": ("fly_insect", "animal", "fly"),
    "1393": ("horn", "animal", "horn"),
    # Verb (existing)
    "1336": ("eat", "verb", "eat"),
    "1401": ("drink", "verb", "drink"),
    "1447": ("give", "verb", "give"),
    "1446": ("come", "verb", "come"),
    "1494": ("die", "verb", "die"),
    "1410": ("know", "verb", "know"),
    "1408": ("hear", "verb", "hear"),
    "1409": ("see", "verb", "see"),
    # Verb (new)
    "1519": ("run", "verb", "run"),
    "1439": ("swim", "verb", "swim"),
    "1416": ("sit", "verb", "sit"),
    "1442": ("stand", "verb", "stand"),
    "1585": ("sleep", "verb", "sleep"),
    "1458": ("say", "verb", "say"),
    "141":  ("burn", "verb", "burn"),
    "1417": ("kill", "verb", "kill"),
    "1403": ("bite", "verb", "bite"),
    "1470": ("walk", "verb", "walk"),
    # Other (existing)
    "1405": ("name", "other", "name"),
    "3231": ("god", "other", "god"),
    "1508": ("king", "other", "king"),
    "1252": ("house", "other", "house"),
    # Adjective/number (new)
    "1231": ("new", "adjective", "new"),
    "1280": ("old", "adjective", "old"),
    "1202": ("big", "adjective", "big"),
    "1246": ("small", "adjective", "small"),
    "1203": ("long", "adjective", "long"),
    "1350": ("good", "adjective", "good"),
    "1335": ("white", "adjective", "white"),
    "1457": ("black", "adjective", "black"),
    "156":  ("red", "adjective", "red"),
    "1493": ("one", "number", "one"),
    "1498": ("two", "number", "two"),
    "492":  ("three", "number", "three"),
    "1429": ("full", "adjective", "full"),
    "1395": ("round", "adjective", "round"),
    "1232": ("warm", "adjective", "warm"),
    "1425": ("green", "adjective", "green"),
    "1367": ("yellow", "adjective", "yellow"),
    # Food/plant (new)
    "714":  ("seed", "plant", "seed"),
    "628":  ("leaf", "plant", "leaf"),
    "670":  ("root", "plant", "root"),
    "1491": ("grass", "plant", "grass"),
    "1507": ("fruit", "plant", "fruit"),
    "634":  ("meat", "food", "meat"),
    "669":  ("fat", "food", "fat"),
    "646":  ("ash", "nature", "ash"),
}

# Reverse map: our concept_id -> concepticon_id
CONCEPT_ID_TO_CID = {v[0]: k for k, v in CONCEPTS.items()}

# ---- Target languages per family branch ----
# Maps (language_id, glottocode) -> family_branch_file
# Language IDs here are the ISO 639-3 codes as used in the CLDF sources

FAMILY_BRANCHES: dict[str, dict[str, tuple[str, str]]] = {
    # branch_name -> {lang_id: (glottocode, display_name)}
    "germanic_expanded": {
        # Original
        "got": ("goth1244", "Gothic"),
        "ang": ("olde1238", "Old English"),
        "non": ("oldn1244", "Old Norse"),
        "goh": ("oldh1241", "Old High German"),
        # New from NorthEuraLex
        "eng": ("stan1293", "English"),
        "deu": ("stan1295", "German"),
        "nld": ("dutc1256", "Dutch"),
        "swe": ("swed1254", "Swedish"),
        "dan": ("dani1285", "Danish"),
        "nor": ("norw1258", "Norwegian"),
        "isl": ("icel1247", "Icelandic"),
        "afr": ("afri1274", "Afrikaans"),
        "fry": ("west2354", "West Frisian"),
    },
    "celtic_expanded": {
        "sga": ("oldi1245", "Old Irish"),
        "cym": ("wels1247", "Welsh"),
        "bre": ("bret1244", "Breton"),
        "gle": ("iris1253", "Irish"),
        "gla": ("scot1245", "Scottish Gaelic"),
    },
    "balto_slavic_expanded": {
        "lit": ("lith1251", "Lithuanian"),
        "chu": ("chur1257", "Old Church Slavonic"),
        "rus": ("russ1263", "Russian"),
        "pol": ("poli1260", "Polish"),
        "ces": ("czec1258", "Czech"),
        "slk": ("slov1269", "Slovak"),
        "srp": ("serb1264", "Serbian"),
        "hrv": ("croa1245", "Croatian"),
        "bul": ("bulg1262", "Bulgarian"),
        "ukr": ("ukra1253", "Ukrainian"),
        "bel": ("bela1254", "Belarusian"),
        "slv": ("slov1268", "Slovenian"),
        "lav": ("latv1249", "Latvian"),
        "dsb": ("lowe1385", "Lower Sorbian"),
    },
    "indo_iranian_expanded": {
        "san": ("sans1269", "Sanskrit"),
        "ave": ("aves1237", "Avestan"),
        "fas": ("west2369", "Persian"),
        "hin": ("hind1269", "Hindi"),
        "ben": ("beng1280", "Bengali"),
        "kmr": ("nort2641", "Northern Kurdish"),
        "pbu": ("nort2646", "Northern Pashto"),
        "pes": ("west2369", "Western Farsi"),
        "oss": ("osse1243", "Ossetian"),
        "rmn": ("seli1249", "Selice Romani"),
    },
    "italic_expanded": {
        "lat": ("lati1261", "Latin"),
        "osc": ("osca1245", "Oscan"),
        "xum": ("umbr1253", "Umbrian"),
        "spa": ("stan1288", "Spanish"),
        "por": ("port1283", "Portuguese"),
        "fra": ("stan1290", "French"),
        "ita": ("ital1282", "Italian"),
        "ron": ("roma1327", "Romanian"),
        "cat": ("stan1289", "Catalan"),
    },
    "hellenic_expanded": {
        "grc": ("anci1242", "Ancient Greek"),
        "gmy": ("myce1241", "Mycenaean Greek"),
        "ell": ("mode1248", "Modern Greek"),
    },
    "semitic_expanded": {
        "heb": ("hebr1245", "Hebrew"),
        "arb": ("stan1318", "Arabic"),
        "amh": ("amha1245", "Amharic"),
    },
    "turkic_expanded": {
        "otk": ("oldt1247", "Old Turkic"),
        "tur": ("nucl1301", "Turkish"),
        "aze": ("nort2697", "Azerbaijani"),
        "azj": ("nort2697", "North Azerbaijani"),
        "kaz": ("kaza1248", "Kazakh"),
        "uzn": ("nort2690", "Northern Uzbek"),
        "bak": ("bash1264", "Bashkir"),
        "tat": ("tata1255", "Tatar"),
        "sah": ("yaku1245", "Yakut/Sakha"),
        "chv": ("chuv1255", "Chuvash"),
    },
    "uralic_expanded": {
        "fin": ("finn1318", "Finnish"),
        "hun": ("hung1274", "Hungarian"),
        "est": ("esto1258", "Estonian"),
        "ekk": ("esto1258", "Estonian (NEL)"),
        "krl": ("kare1335", "Karelian"),
        "sme": ("nort2671", "Northern Sami"),
        "myv": ("erzy1239", "Erzya"),
        "mdf": ("moks1248", "Moksha"),
        "kpv": ("komi1268", "Komi-Zyrian"),
        "koi": ("komi1269", "Komi-Permyak"),
        "udm": ("udmu1245", "Udmurt"),
        "mhr": ("east2328", "Eastern Mari"),
        "mrj": ("west2392", "Western Mari"),
        "mns": ("mans1258", "Mansi"),
        "kca": ("khan1273", "Khanty"),
        "yrk": ("tund1250", "Tundra Nenets"),
        "enf": ("fore1255", "Forest Enets"),
        "sel": ("selk1253", "Selkup"),
        "olo": ("livv1244", "Livvi Karelian"),
        "vep": ("veps1250", "Veps"),
        "liv": ("livn1244", "Livonian"),
        "sma": ("sout2674", "Southern Sami"),
        "smj": ("lule1254", "Lule Sami"),
        "smn": ("inar1241", "Inari Sami"),
        "sms": ("skol1241", "Skolt Sami"),
        "nio": ("ngan1291", "Nganasan"),
        "sjd": ("kild1236", "Kildin Sami"),
    },
    # New families
    "albanian": {
        "sqi": ("alba1268", "Albanian"),
    },
    "armenian": {
        "hye": ("nucl1235", "Armenian"),
    },
    "dravidian": {
        "tam": ("tami1289", "Tamil"),
        "tel": ("telu1262", "Telugu"),
        "kan": ("nucl1305", "Kannada"),
        "mal": ("mala1464", "Malayalam"),
    },
    "kartvelian": {
        "kat": ("nucl1302", "Georgian"),
    },
    "mongolic": {
        "khk": ("halh1238", "Mongolian"),
        "bua": ("buri1258", "Buryat"),
        "xal": ("kalm1243", "Kalmyk"),
    },
    "tungusic": {
        "evn": ("even1259", "Evenki"),
        "mnc": ("manc1252", "Manchu"),
        "gld": ("nana1257", "Nanai"),
        "orh": ("oroq1238", "Oroqen"),
    },
    "japonic": {
        "jpn": ("nucl1643", "Japanese"),
    },
    "koreanic": {
        "kor": ("kore1280", "Korean"),
    },
    "northeast_caucasian": {
        "ava": ("avar1256", "Avar"),
        "lez": ("lezg1247", "Lezgian"),
        "dar": ("darg1241", "Dargwa"),
        "lbe": ("lakk1252", "Lak"),
        "che": ("chec1245", "Chechen"),
        "ddo": ("tsez1241", "Tsez"),
        "kap": ("bezh1248", "Bezhta"),
        "aqc": ("arch1244", "Archi"),
    },
    "northwest_caucasian": {
        "abk": ("abkh1244", "Abkhaz"),
        "ady": ("adyg1241", "Adyghe"),
    },
    "eskimo_aleut": {
        "ale": ("aleu1260", "Aleut"),
        "ess": ("cent2127", "Central Siberian Yupik"),
        "kal": ("kala1399", "Kalaallisut"),
    },
    "chukotko_kamchatkan": {
        "ckt": ("chuk1273", "Chukchi"),
        "itl": ("itel1242", "Itelmen"),
    },
    "yukaghir": {
        "ykg": ("nort2745", "Northern Yukaghir"),
        "yux": ("sout2750", "Southern Yukaghir"),
    },
    "isolates": {
        "eus": ("basq1248", "Basque"),
        "ain": ("ainu1240", "Ainu"),
        "ket": ("kett1243", "Ket"),
        "niv": ("gily1242", "Nivkh"),
        "bsk": ("buru1296", "Burushaski"),
    },
}

# WOLD uses different language names; map WOLD Language_ID -> our (lang_id, branch)
WOLD_LANG_MAP: dict[str, tuple[str, str]] = {
    "Swahili": ("swh", "niger_congo_bantu"),
    "Iraqw": ("irk", "afroasiatic_cushitic"),
    "Gawwada": ("gwd", "afroasiatic_cushitic"),
    "Hausa": ("hau", "afroasiatic_chadic"),
    "TarifiytBerber": ("rif", "afroasiatic_berber"),
    "Indonesian": ("ind", "austronesian"),
    "Malagasy": ("mlg", "austronesian"),
    "Hawaiian": ("haw", "austronesian"),
    "Takia": ("tbc", "austronesian"),
    "Vietnamese": ("vie", "austroasiatic"),
    "CeqWong": ("cwg", "austroasiatic"),
    "Thai": ("tha", "tai_kadai"),
    "MandarinChinese": ("cmn", "sino_tibetan"),
    "Manange": ("nmm", "sino_tibetan"),
    "Japanese": ("jpn", "japonic"),
    "Sakha": ("sah", "turkic_expanded"),
    "Oroqen": ("orh", "tungusic"),
    "KildinSaami": ("sjd", "uralic_expanded"),
    "Sakha": ("sah", "turkic_expanded"),
    "Ket": ("ket", "isolates"),
    "WhiteHmong": ("mww", "hmong_mien"),
    "Gurindji": ("gue", "pama_nyungan"),
    "Yaqui": ("yaq", "uto_aztecan"),
    "Otomi": ("ote", "otomanguean"),
    "ImbaburaQuechua": ("qvi", "quechuan"),
    "ZinacantanTzotzil": ("tzo", "mayan"),
    "Qeqchi": ("kek", "mayan"),
    "Kalina": ("car", "cariban"),
    "Hup": ("jup", "naduhup"),
    "Wichi": ("mzh", "matacoan"),
    "Mapudungun": ("arn", "araucanian"),
    "Romanian": ("ron", "italic_expanded"),
    "SeychellesCreole": ("crs", "creole"),
    "SeliceRomani": ("rmn", "indo_iranian_expanded"),
    "LowerSorbian": ("dsb", "balto_slavic_expanded"),
    "OldHighGerman": ("goh", "germanic_expanded"),
    "Bezhta": ("kap", "northeast_caucasian"),
    "Archi": ("aqc", "northeast_caucasian"),
    "Kanuri": ("knc", "saharan"),
}

# Additional WOLD-only family branches (languages not covered elsewhere)
WOLD_EXTRA_BRANCHES: dict[str, dict[str, tuple[str, str]]] = {
    "niger_congo_bantu": {
        "swh": ("swah1253", "Swahili"),
    },
    "afroasiatic_cushitic": {
        "irk": ("iraq1241", "Iraqw"),
        "gwd": ("gaww1239", "Gawwada"),
    },
    "afroasiatic_chadic": {
        "hau": ("haus1257", "Hausa"),
    },
    "afroasiatic_berber": {
        "rif": ("tari1263", "Tarifiyt Berber"),
    },
    "austroasiatic": {
        "vie": ("viet1252", "Vietnamese"),
        "cwg": ("ceqw1242", "Ceq Wong"),
    },
    "tai_kadai": {
        "tha": ("thai1261", "Thai"),
    },
    "sino_tibetan": {
        "cmn": ("mand1415", "Mandarin Chinese"),
        "nmm": ("mana1288", "Manange"),
    },
    "mayan": {
        "tzo": ("tzel1254", "Zinacant\u00e1n Tzotzil"),
        "kek": ("keqc1242", "Q'eqchi'"),
    },
    "quechuan": {
        "qvi": ("imba1240", "Imbabura Quechua"),
    },
    "uto_aztecan": {
        "yaq": ("yaqu1251", "Yaqui"),
    },
    "austronesian": {
        "ind": ("indo1316", "Indonesian"),
        "mlg": ("plat1254", "Malagasy"),
        "haw": ("hawa1245", "Hawaiian"),
        "tbc": ("taki1248", "Takia"),
    },
    "hmong_mien": {
        "mww": ("hmon1333", "White Hmong"),
    },
    "creole": {
        "crs": ("seyc1239", "Seychelles Creole"),
    },
    "saharan": {
        "knc": ("cent2050", "Kanuri"),
    },
    "cariban": {
        "car": ("gali1262", "Kali'na"),
    },
    "naduhup": {
        "jup": ("hupp1235", "Hup"),
    },
    "matacoan": {
        "mzh": ("wich1264", "Wichi"),
    },
    "araucanian": {
        "arn": ("mapu1245", "Mapudungun"),
    },
    "pama_nyungan": {
        "gue": ("guri1247", "Gurindji"),
    },
    "otomanguean": {
        "ote": ("mezt1237", "Mezquital Otomi"),
    },
}

# Merge WOLD extras into FAMILY_BRANCHES for completeness
for branch, langs in WOLD_EXTRA_BRANCHES.items():
    if branch not in FAMILY_BRANCHES:
        FAMILY_BRANCHES[branch] = {}
    FAMILY_BRANCHES[branch].update(langs)

# NorthEuraLex uses specific language IDs; build a lookup from NEL ID -> our branch
# Must be after WOLD merge so all language IDs are included
NEL_LANG_TO_BRANCH: dict[str, str] = {}
for branch, langs in FAMILY_BRANCHES.items():
    for lang_id in langs:
        NEL_LANG_TO_BRANCH[lang_id] = branch


# ---- Helper functions ----

def read_cldf_csv(path: Path) -> list[dict[str, str]]:
    """Read a CLDF CSV file."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def segments_to_ipa(segments: str) -> str:
    """Convert CLDF Segments column (space-separated) to IPA string."""
    if not segments or not segments.strip():
        return ""
    # Remove profile markers like ^, $, +
    parts = segments.split()
    cleaned = [p for p in parts if p not in ("^", "$", "+", "#", "_")]
    # Join without spaces for IPA
    return "".join(cleaned)


def clean_ipa(ipa: str) -> str:
    """Clean IPA string — remove non-phonetic markers."""
    if not ipa:
        return ""
    # Remove common non-IPA chars
    ipa = ipa.replace("ˈ", "").replace("ˌ", "")  # stress marks
    ipa = ipa.replace("ː", "").replace("ˑ", "")  # length (keep for now? remove for SCA)
    # Actually, keep length marks — SCA strips diacritics itself
    ipa = ipa.replace("ˈ", "").replace("ˌ", "")
    return ipa.strip()


def form_to_pseudo_ipa(form: str) -> str:
    """For repos without IPA, use orthographic form as pseudo-IPA.

    Works reasonably well for Austronesian languages which use
    Latin-based orthographies close to IPA.
    """
    if not form:
        return ""
    # Basic normalization for common Austronesian orthography
    form = form.lower().strip()
    # Remove parenthetical notes
    form = re.sub(r"\(.*?\)", "", form)
    # Take first alternative if multiple
    if "," in form:
        form = form.split(",")[0].strip()
    if "/" in form:
        form = form.split("/")[0].strip()
    # Remove non-letter chars except IPA
    form = re.sub(r"[^a-zA-Zɑæɐɛəɘɪɨɔɵʊʉɯøœɸβʈɖɡɢʔɦɲŋɳɴɫɭɬɮɾɽʀɹʁɕʑʃʒθðɣχʝʋɰɟʂɻɓɗʕħɺɥɧʙⱱʘǀǁǃǂ\u0361\u0300-\u036F]", "", form)
    return form


# ---- SCA validation ----

SCA_KNOWN = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "ɑæɐɛəɘɪɨɔɵʊʉɯøœɸβʈɖɡɢʔɦɲŋɳɴɫɭɬɮɾɽʀɹʁɕʑʃʒθðɣχʝʋɰɟʂɻɓɗ"
                "ʕħɺɥɧʙⱱʘǀǁǃǂ$H<@*")


def validate_ipa_for_sca(ipa: str) -> bool:
    """Check if all base characters in IPA string are known to SCA."""
    if not ipa:
        return False
    for ch in ipa:
        if ch in SCA_KNOWN:
            continue
        # Allow diacritics and modifiers
        cp = ord(ch)
        if 0x0300 <= cp <= 0x036F:  # combining diacriticals
            continue
        if cp in (0x0325, 0x0329, 0x032A, 0x033A, 0x033B, 0x033C):  # below diacritics
            continue
        if cp in (0x02B0, 0x02BC, 0x02D0, 0x02D1):  # modifier letters
            continue
        if cp == 0x0361:  # tie bar
            continue
        # Unknown character — still allow it, just flag
        pass
    return True


# ---- Main conversion ----

def convert_northeuralex() -> dict[str, list[dict]]:
    """Convert NorthEuraLex CLDF to our format."""
    cldf_dir = SOURCES / "northeuralex" / "cldf"
    if not cldf_dir.exists():
        print("  NorthEuraLex not found, skipping")
        return {}

    # Build maps
    lang_map = {}  # NEL lang_id -> {Glottocode, Name, Family, Subfamily}
    for row in read_cldf_csv(cldf_dir / "languages.csv"):
        lang_map[row["ID"]] = row

    param_map = {}  # NEL param_id -> Concepticon_ID
    for row in read_cldf_csv(cldf_dir / "parameters.csv"):
        param_map[row["ID"]] = row.get("Concepticon_ID", "")

    # Process forms
    results: dict[str, list[dict]] = defaultdict(list)
    skipped_lang = 0
    skipped_concept = 0
    processed = 0

    for row in read_cldf_csv(cldf_dir / "forms.csv"):
        lang_id = row.get("Language_ID", "")
        param_id = row.get("Parameter_ID", "")
        segments = row.get("Segments", "")

        # Map to Concepticon
        cid = param_map.get(param_id, "")
        if cid not in CONCEPTS:
            skipped_concept += 1
            continue

        # Map language to branch
        branch = NEL_LANG_TO_BRANCH.get(lang_id)
        if branch is None:
            skipped_lang += 1
            continue

        # Get IPA from segments
        ipa = segments_to_ipa(segments)
        if not ipa:
            continue

        concept_id, _, _ = CONCEPTS[cid]
        lang_info = lang_map.get(lang_id, {})
        glottocode = lang_info.get("Glottocode", "")

        # Check if this lang is in the branch definition
        branch_def = FAMILY_BRANCHES.get(branch, {})
        if lang_id in branch_def:
            glottocode = branch_def[lang_id][0]  # Use our curated glottocode

        results[branch].append({
            "Language_ID": lang_id,
            "Parameter_ID": concept_id,
            "Form": row.get("Form", row.get("Value", "")),
            "IPA": ipa,
            "Glottocode": glottocode,
        })
        processed += 1

    print(f"  NorthEuraLex: {processed:,} entries, {skipped_lang:,} skipped (lang), {skipped_concept:,} skipped (concept)")
    return dict(results)


def convert_wold() -> dict[str, list[dict]]:
    """Convert WOLD CLDF to our format."""
    cldf_dir = SOURCES / "wold" / "cldf"
    if not cldf_dir.exists():
        print("  WOLD not found, skipping")
        return {}

    param_map = {}
    for row in read_cldf_csv(cldf_dir / "parameters.csv"):
        param_map[row["ID"]] = row.get("Concepticon_ID", "")

    results: dict[str, list[dict]] = defaultdict(list)
    processed = 0
    skipped = 0

    for row in read_cldf_csv(cldf_dir / "forms.csv"):
        wold_lang = row.get("Language_ID", "")
        param_id = row.get("Parameter_ID", "")
        segments = row.get("Segments", "")

        cid = param_map.get(param_id, "")
        if cid not in CONCEPTS:
            skipped += 1
            continue

        mapping = WOLD_LANG_MAP.get(wold_lang)
        if mapping is None:
            skipped += 1
            continue

        lang_id, branch = mapping
        ipa = segments_to_ipa(segments)
        if not ipa:
            continue

        concept_id, _, _ = CONCEPTS[cid]

        # Look up glottocode from branch definition
        branch_def = FAMILY_BRANCHES.get(branch, {})
        glottocode = branch_def.get(lang_id, ("", ""))[0]

        results[branch].append({
            "Language_ID": lang_id,
            "Parameter_ID": concept_id,
            "Form": row.get("Form", row.get("Value", "")),
            "IPA": ipa,
            "Glottocode": glottocode,
        })
        processed += 1

    print(f"  WOLD: {processed:,} entries, {skipped:,} skipped")
    return dict(results)


def convert_abvd() -> dict[str, list[dict]]:
    """Convert ABVD CLDF to our format (Austronesian only)."""
    cldf_dir = SOURCES / "abvd" / "cldf"
    if not cldf_dir.exists():
        print("  ABVD not found, skipping")
        return {}

    # Read language metadata
    lang_map = {}
    for row in read_cldf_csv(cldf_dir / "languages.csv"):
        lang_map[row["ID"]] = row

    param_map = {}
    for row in read_cldf_csv(cldf_dir / "parameters.csv"):
        param_map[row["ID"]] = row.get("Concepticon_ID", "")

    # Target ABVD languages (well-known Austronesian with decent orthography)
    target_langs = {}
    for row in lang_map.values():
        fam = row.get("Family", "")
        if fam != "Austronesian":
            continue
        glottocode = row.get("Glottocode", "")
        iso = row.get("ISO639P3code", "")
        name = row.get("Name", "")
        if not glottocode or not iso:
            continue
        # Select well-known and representative languages
        if iso in ("ind", "msa", "tgl", "jav", "ceb", "haw", "mri",
                    "smo", "ton", "fij", "mlg", "ilo", "zlm", "sun",
                    "ban", "min", "bug", "mah", "cha", "rap", "niu",
                    # Formosan
                    "ami", "tay", "pwn",
                    # Oceanic
                    "gil", "chk", "kos", "rar", "pag",
                    # Additional well-known
                    "hil", "plt", "zsm", "tet", "wbm"):
            target_langs[row["ID"]] = (iso, glottocode, name)

    results: dict[str, list[dict]] = defaultdict(list)
    processed = 0
    skipped = 0

    # Read ABVD forms
    with open(cldf_dir / "forms.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            abvd_lang = row.get("Language_ID", "")
            if abvd_lang not in target_langs:
                skipped += 1
                continue

            param_id = row.get("Parameter_ID", "")
            cid = param_map.get(param_id, "")
            if cid not in CONCEPTS:
                skipped += 1
                continue

            form = row.get("Form", "").strip()
            if not form:
                continue

            iso, glottocode, name = target_langs[abvd_lang]
            ipa = form_to_pseudo_ipa(form)
            if not ipa:
                continue

            concept_id, _, _ = CONCEPTS[cid]
            cognacy = row.get("Cognacy", "").strip()

            entry = {
                "Language_ID": iso,
                "Parameter_ID": concept_id,
                "Form": form,
                "IPA": ipa,
                "Glottocode": glottocode,
            }
            if cognacy:
                entry["Cognate_Set_ID"] = f"abvd_{cid}_{cognacy.split(',')[0]}"

            results["austronesian"].append(entry)
            processed += 1

    # Make sure austronesian branch exists in FAMILY_BRANCHES
    if "austronesian" not in FAMILY_BRANCHES:
        FAMILY_BRANCHES["austronesian"] = {}
    for abvd_id, (iso, gc, name) in target_langs.items():
        if iso not in FAMILY_BRANCHES["austronesian"]:
            FAMILY_BRANCHES["austronesian"][iso] = (gc, name)

    print(f"  ABVD: {processed:,} entries, {skipped:,} skipped")
    return dict(results)


def convert_sinotibetan() -> dict[str, list[dict]]:
    """Convert digling/sinotibetan dump TSV to our format."""
    # Try extracted dump first, then original path
    dump_path = SOURCES / "sinotibetan" / "sinotibetan_dump.tsv"
    if not dump_path.exists():
        dump_path = SOURCES / "sinotibetan" / "dumps" / "sinotibetan.tsv"
    if not dump_path.exists():
        print("  Sino-Tibetan dump not found, skipping")
        return {}

    # Doculect -> (lang_id, glottocode, name)
    doculect_map = {
        "Old_Chinese": ("och", "oldc1244", "Old Chinese"),
        "Japhug": ("jya", "japh1234", "Japhug"),
        "Tibetan_Written": ("bod", "clas1255", "Classical Tibetan"),
        "Old_Burmese": ("obr", "oldb1235", "Old Burmese"),
        "Jingpho": ("kac", "jinp1238", "Jingpho"),
        "Lisu": ("lis", "lisu1250", "Lisu"),
        "Naxi": ("nxq", "naxi1245", "Naxi"),
        "Khaling": ("klr", "khal1275", "Khaling"),
        "Limbu": ("lif", "limb1266", "Limbu"),
        "Pumi_Lanping": ("pmi", "nort2743", "Pumi"),
        "Qiang_Mawo": ("qxs", "mawo1239", "Mawo Qiang"),
        "Tujia": ("tji", "nort2726", "Northern Tujia"),
        "Dulong": ("duu", "drun1238", "Drung"),
        "Hakha": ("cnh", "hakn1238", "Hakha Chin"),
        "Bai_Jianchuan": ("bca", "jian1238", "Jianchuan Bai"),
    }

    results: dict[str, list[dict]] = defaultdict(list)
    processed = 0
    skipped = 0

    with open(dump_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            doculect = row.get("DOCULECT", "")
            if doculect not in doculect_map:
                skipped += 1
                continue

            concept = row.get("CONCEPT", "").lower().strip()
            ipa = row.get("IPA", "").strip()
            if not ipa:
                continue

            # Try to match concept to our concept list
            # sinotibetan concepts are English glosses
            matched_concept = None
            for cid, (cid_name, cat, gloss) in CONCEPTS.items():
                if concept == gloss or concept == cid_name:
                    matched_concept = cid_name
                    break
            if not matched_concept:
                skipped += 1
                continue

            lang_id, glottocode, name = doculect_map[doculect]
            cogid = row.get("COGID", "").strip()

            entry = {
                "Language_ID": lang_id,
                "Parameter_ID": matched_concept,
                "Form": row.get("CONCEPT", ""),
                "IPA": ipa,
                "Glottocode": glottocode,
            }
            if cogid:
                entry["Cognate_Set_ID"] = f"st_{cogid}"

            results["sino_tibetan"].append(entry)
            processed += 1

    # Add to FAMILY_BRANCHES
    if "sino_tibetan" not in FAMILY_BRANCHES:
        FAMILY_BRANCHES["sino_tibetan"] = {}
    for doc, (lid, gc, name) in doculect_map.items():
        FAMILY_BRANCHES["sino_tibetan"][lid] = (gc, name)

    print(f"  Sino-Tibetan: {processed:,} entries, {skipped:,} skipped")
    return dict(results)


def deduplicate_entries(entries: list[dict]) -> list[dict]:
    """Remove duplicate (Language_ID, Parameter_ID) pairs, keeping first."""
    seen = set()
    deduped = []
    for entry in entries:
        key = (entry["Language_ID"], entry["Parameter_ID"])
        if key not in seen:
            seen.add(key)
            deduped.append(entry)
    return deduped


def write_tsv(path: Path, entries: list[dict], has_cognate: bool = False):
    """Write entries to a TSV file."""
    if not entries:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["Language_ID", "Parameter_ID", "Form", "IPA", "Glottocode"]
    if has_cognate:
        cols.append("Cognate_Set_ID")

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(cols) + "\n")
        for entry in entries:
            row = [entry.get(c, "") for c in cols]
            f.write("\t".join(row) + "\n")


def write_concepts_expanded(path: Path):
    """Write the expanded concepts TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("concept_id\tconcepticon_id\tcategory\tenglish_gloss\n")
        for cid, (concept_id, category, gloss) in sorted(CONCEPTS.items(), key=lambda x: x[1]):
            f.write(f"{concept_id}\t{cid}\t{category}\t{gloss}\n")


def write_languages_tsv(path: Path, all_entries: dict[str, list[dict]]):
    """Write master language metadata TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all unique languages
    langs = {}
    for branch, entries in all_entries.items():
        branch_def = FAMILY_BRANCHES.get(branch, {})
        for entry in entries:
            lid = entry["Language_ID"]
            if lid not in langs:
                gc = entry.get("Glottocode", "")
                name = branch_def.get(lid, ("", lid))[1]
                langs[lid] = {
                    "Language_ID": lid,
                    "Name": name,
                    "Family": branch.replace("_expanded", ""),
                    "Glottocode": gc,
                }

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("Language_ID\tName\tFamily\tGlottocode\n")
        for lid, info in sorted(langs.items()):
            f.write(f"{info['Language_ID']}\t{info['Name']}\t{info['Family']}\t{info['Glottocode']}\n")

    return langs


def load_original_validation() -> dict[str, list[dict]]:
    """Load existing original validation TSV files.

    Maps original branch names to their expanded counterparts so data
    is merged together.
    """
    ORIGINAL_TO_EXPANDED = {
        "germanic": "germanic_expanded",
        "celtic": "celtic_expanded",
        "balto_slavic": "balto_slavic_expanded",
        "indo_iranian": "indo_iranian_expanded",
        "italic": "italic_expanded",
        "hellenic": "hellenic_expanded",
        "semitic": "semitic_expanded",
        "turkic": "turkic_expanded",
        "uralic": "uralic_expanded",
    }

    results: dict[str, list[dict]] = defaultdict(list)
    total = 0

    for orig_name, expanded_name in ORIGINAL_TO_EXPANDED.items():
        orig_path = VALIDATION / f"{orig_name}.tsv"
        if not orig_path.exists():
            continue

        with open(orig_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("IPA", "_") == "_" or row.get("Form", "_") == "_":
                    continue
                entry = {
                    "Language_ID": row["Language_ID"],
                    "Parameter_ID": row["Parameter_ID"],
                    "Form": row["Form"],
                    "IPA": row["IPA"],
                    "Glottocode": row.get("Glottocode", ""),
                }
                if "Cognate_Set_ID" in row:
                    entry["Cognate_Set_ID"] = row["Cognate_Set_ID"]
                results[expanded_name].append(entry)
                total += 1

    print(f"  Original validation data: {total:,} entries from {len(ORIGINAL_TO_EXPANDED)} branches")
    return dict(results)


def main():
    print("=" * 80)
    print("CLDF to TSV Conversion")
    print("=" * 80)

    # Load original validation data first
    print("\nLoading original validation data...")
    all_results: dict[str, list[dict]] = defaultdict(list)
    orig_data = load_original_validation()
    for branch, entries in orig_data.items():
        all_results[branch].extend(entries)

    # Convert each source
    print("\nConverting CLDF sources...")

    for source_name, converter in [
        ("NorthEuraLex", convert_northeuralex),
        ("WOLD", convert_wold),
        ("ABVD", convert_abvd),
        ("Sino-Tibetan", convert_sinotibetan),
    ]:
        print(f"\n  [{source_name}]")
        results = converter()
        for branch, entries in results.items():
            all_results[branch].extend(entries)

    # Deduplicate and write
    print("\n" + "=" * 80)
    print("Writing TSV files")
    print("=" * 80)

    total_entries = 0
    total_langs = set()
    total_concepts = set()

    for branch, entries in sorted(all_results.items()):
        entries = deduplicate_entries(entries)
        all_results[branch] = entries  # store deduped

        if not entries:
            continue

        # Check for cognate data
        has_cognate = any(e.get("Cognate_Set_ID") for e in entries)

        out_path = VALIDATION / f"{branch}.tsv"
        write_tsv(out_path, entries, has_cognate=has_cognate)

        # Stats
        branch_langs = set(e["Language_ID"] for e in entries)
        branch_concepts = set(e["Parameter_ID"] for e in entries)
        total_langs.update(branch_langs)
        total_concepts.update(branch_concepts)
        total_entries += len(entries)

        print(f"  {branch}.tsv: {len(entries):,} entries, "
              f"{len(branch_langs)} langs, {len(branch_concepts)} concepts"
              f"{' (with cognacy)' if has_cognate else ''}")

    # Write metadata files
    print("\nWriting metadata...")
    write_concepts_expanded(VALIDATION / "concepts_expanded.tsv")
    langs = write_languages_tsv(VALIDATION / "languages.tsv", dict(all_results))

    # Coverage report
    print("\n" + "=" * 80)
    print("COVERAGE SUMMARY")
    print("=" * 80)
    print(f"  Total entries:  {total_entries:,}")
    print(f"  Total languages: {len(total_langs)}")
    print(f"  Total concepts:  {len(total_concepts)}")
    print(f"  TSV files:      {sum(1 for b in all_results if all_results[b])}")
    print(f"  Concept list:   {len(CONCEPTS)} concepts")

    # Write coverage report
    report_path = SCRIPTS / "coverage_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CLDF -> TSV Conversion Coverage Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total entries:   {total_entries:,}\n")
        f.write(f"Total languages: {len(total_langs)}\n")
        f.write(f"Total concepts:  {len(total_concepts)}\n")
        f.write(f"TSV files:       {sum(1 for b in all_results if all_results[b])}\n\n")
        f.write("Languages:\n")
        for lid in sorted(total_langs):
            info = langs.get(lid, {})
            f.write(f"  {lid}: {info.get('Name', '?')} ({info.get('Family', '?')})\n")
        f.write(f"\nConcepts:\n")
        for cid in sorted(total_concepts):
            f.write(f"  {cid}\n")

    print(f"\n  Coverage report: {report_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
