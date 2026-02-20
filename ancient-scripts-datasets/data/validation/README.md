# Phylogenetic Validation Dataset

Balanced validation data across 9 major language families for testing cognate detection pipelines.

## Format

Each branch TSV has columns: `Language_ID`, `Parameter_ID`, `Form`, `IPA`, `Glottocode`

- `IPA` column provides phonetic transcriptions (triggers `TranscriptionType.IPA` in the pipeline)
- `_` marks missing/unattested forms (skipped by ingester)
- `concepts.tsv` lists all 40 shared concept IDs

## Branches

| File | Family | Languages | Entries |
|---|---|---|---|
| germanic.tsv | Germanic | got, ang, non, goh | ~160 |
| celtic.tsv | Celtic | sga, cym, bre | ~120 |
| balto_slavic.tsv | Balto-Slavic | lit, chu, rus | ~120 |
| indo_iranian.tsv | Indo-Iranian | san, ave, fas | ~120 |
| italic.tsv | Italic | lat, osc, xum | ~120 |
| hellenic.tsv | Hellenic | grc, gmy | ~80 |
| semitic.tsv | Semitic | heb, arb, amh | ~120 |
| turkic.tsv | Turkic | otk, tur, aze | ~120 |
| uralic.tsv | Uralic | fin, hun, est | ~120 |

## Concept Categories (40 concepts)

- **Body** (8): head, eye, ear, mouth, hand, foot, heart, blood
- **Kinship** (5): father, mother, son, daughter, brother
- **Nature** (10): water, fire, sun, moon, star, earth, mountain, river, stone, tree
- **Animals** (5): horse, dog, fish, bird, ox
- **Verbs** (8): eat, drink, give, come, die, know, hear, see
- **Other** (4): name, god, king, house

## Names Subset

`names.tsv` is a cross-family validation file covering deity names (theonyms), proper/personal names (anthroponyms), and place names (toponyms) across 30 languages from all 9 families plus additional ancient languages (Hittite, Ugaritic, Akkadian, Phoenician, Aramaic).

Unlike the branch files, entries sharing a `Parameter_ID` include both true cognates (same etymological root) and false positives (unrelated names in the same semantic slot). The pipeline should cluster true cognates together while keeping false positives apart.

### Cognate annotations

`names.tsv` has an extra 6th column `Cognate_Set_ID` providing ground-truth cognate judgements. Forms sharing the same `Cognate_Set_ID` within a `Parameter_ID` are etymologically related; forms with different IDs are not. Null entries use `_`. The ingester ignores this column (not in column_mapping), but evaluation scripts can read it directly.

| Statistic | Value |
|---|---|
| Total entries | 168 (160 non-null) |
| Languages | 30 (cross-family) |
| Concepts | 15 |
| True cognate pairs | 227 (27%) |
| False-positive pairs | 605 (73%) |

### Name Categories (15 concepts)

**Deity names (10):**
- `sky_father` — PIE *dyēws cognates (Zeus, Jupiter, Dyaus, Týr, Dievas) vs unrelated (El, Tengri, Ukko)
- `thunder_god` — PIE *Perkʷ- (Perkūnas, Perun, Parjanya) + PIE *tonh₂r- (Þórr, Þunor, Donar) vs Semitic Baal
- `sun_god` — PIE *sóh₂wl̥ (Sūrya, Sōl, Sunne) + Semitic (Šemeš, Šams, Šapšu) vs Hēlios, Hvar
- `moon_god` — PIE *meh₁ns- (Máni, Mōna, Māno, Mēna, Mėnulis) vs Selēnē, Lūna, Yareaḥ
- `war_god` — Germanic *Wōdanaz (Óðinn, Wōden, Wuotan) vs Arēs, Mārs, Indra
- `love_deity` — Semitic *ʿAṯtar- (ʿAṯtartu, ʿAštoret, Ištar) vs Aphrodītē, Venus, Freyja
- `death_god` — PIE *Yemo- (Yama, Yima) + Semitic *mwt (Māvet, Mawt, Mōtu) vs Hādēs, Hel
- `sea_god` — Semitic *yamm- (Yammu, Yām) vs Poseidōn, Neptūnus, Njǫrðr
- `smith_god` — Celtic (Goibniu, Gofannon) + Germanic (Wēland, Vǫlundr) vs Hēphaistos, Koṯaru
- `supreme_god` — Semitic *ʔil- (El, Ilāh, Ilu) + Turkic *teŋri (Teŋri, Tanrı) + Uralic (Jumala, Jumal)

**Proper names (2):**
- `caesar_title` — Latin loanword chain: Caesar → Kaisar → Cěsarĭ → Car → Qayṣar → Kayser
- `alexander_name` — Greek loanword chain: Alexandros → Iskandar → İskender → Eskandar

**Place names (3):**
- `sacred_river` — PIE *deh₂nu- river names (Dānuvius, Dunaj, Dānu) vs Neilos, Yardēn
- `sacred_mountain` — Mostly unrelated (Olumpos, Sinay, Meru, Ásgarðr) — false positive testing
- `holy_city` — Mostly unrelated (Yerušalayim, Athēnai, Rōma) — false positive testing

## IPA Sources

| Branch | Reference |
|---|---|
| Germanic | Wright's Gothic Grammar, Campbell's OE Grammar, Gordon's ON, Braune's OHG |
| Celtic | Thurneysen's Old Irish Grammar, GPC (Welsh), Hemon's Breton Grammar |
| Balto-Slavic | Leskien's OCS Handbook, Lithuanian reference grammar, Wade's Russian |
| Indo-Iranian | Whitney's Sanskrit Grammar, Skjaervo's Avestan, Lambton's Persian |
| Italic | Buck's Oscan-Umbrian Grammar, Allen's Vox Latina |
| Hellenic | Ventris & Chadwick, Allen's Vox Graeca |
| Semitic | Fischer's Arabic Grammar, Leslau's Amharic |
| Turkic | Tekin's Old Turkic Grammar, Lewis's Turkish Grammar |
| Uralic | Karlsson's Finnish, Rounds's Hungarian, Viitso's Estonian |
