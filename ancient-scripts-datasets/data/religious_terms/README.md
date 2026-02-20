# Religious Terms Subset

A curated subset of the decipherment datasets, filtered to terms related to **religious rites, customs, and practices** — broadly defined to include deity names, ritual vocabulary, sacred places, theological concepts, votive formulas, and activities that would have been subjects of prayer (childbirth, harvest, war, healing, sea voyages, etc.).

## Files

### `ugaritic_hebrew_religious.tsv`
**~170 cognate pairs** extracted from the Ugaritic-Hebrew cognate dataset, organized into:

| Category | Count | Examples |
|---|---|---|
| **Deity names** | 30+ entries | El, Baal, Anat, Dagon, Shamash, Yarikh, Mot, Yam, Kothar, Resheph |
| **Theophoric names** | 20+ entries | Abdibaal, Ilresheph, Servant-of-Anat, Baal-Shalom, etc. |
| **Ritual verbs** | 35+ entries | sacrifice (dbH/zbH), bless (brk), pour libation (nsk), vow (ndr), purify (b<r), atone (kpr) |
| **Ritual nouns** | 10+ entries | altar (mdbH), offering (mnH), incense (qTr), tithe (<$r) |
| **Sacred places** | 10+ entries | temple (hkl), house/temple (bt), grave (qbr), sanctuary (mqd$), high place (bmh) |
| **Theological concepts** | 30+ entries | holy (qd$), soul (np$), spirit (rH), peace ($lm), grace (Hnn), righteousness (Sdq) |
| **High-risk activities** | 40+ entries | birth (yld), harvest (qSr), seed (zr<), vineyard (krm), rain (mTr), war (mlHm), ship (any) |

Key sound correspondences visible in the data:
- Ugaritic **d** = Hebrew **z** (e.g., dbH/zbH "sacrifice", dr</zr< "seed")
- Ugaritic **v** (theta) = Hebrew **$** (e.g., vr/$wr "bull")
- Ugaritic **x** = Hebrew **H** (e.g., xlq/Hlq "portion")
- Ugaritic **$p$** = Hebrew **$m$** ("sun/Shamash")

### `gothic_religious.tsv`
**~65 terms** from the Gothic Bible (Wulfila's 4th-century translation), organized into:

| Category | Count | Examples |
|---|---|---|
| **Deity terms** | 7 | guþ (God), frauja (Lord), Xristus, ahma (Spirit), aggilus (angel), unhulþo (demon), Satana |
| **Ritual verbs** | 7 | hunsl (sacrifice), bidjan (pray), weihnai (hallow), daupjan (baptize), fastan (fast) |
| **Ritual nouns** | 12 | alh (temple), gaqumþ (assembly), witoþ (law/Torah), giba (offering), hlaifs (bread), wein (wine), stikls (cup) |
| **Sin/salvation** | 9 | frawaurht (sin), lausjan (redeem), nasjan (save), galaubjan (believe), usstass (resurrection) |
| **Sacred concepts** | 14 | saiwala (soul), himins (heaven), halja (hell), þiudangardi (kingdom), wulþus (glory), galga (cross) |
| **High-risk/prayed-for** | 10 | sauhtins (sickness), hailjan (heal), gabaurþs (birth), dauþus (death), asans (harvest), fraistubni (temptation) |

### `iberian_religious.tsv`
**~40 elements** from the Iberian inscription corpus, organized into:

| Category | Count | Examples |
|---|---|---|
| **Proposed deity names** | 4 | neitin (war deity), iltiŕ, beleś, atin |
| **Votive formula elements** | 8 | iunstir (dedication), ekiar ("made"), ḿi (dedicatory), eban ("gave"), seltar (tomb), śalir (silver/payment) |
| **Onomastic elements** | 8 | ban, bilos, baite, kutuŕ, oŕtin, sakaŕ, saltu, biuŕ |
| **Sanctuary inscriptions** | 14 | Full texts from Ullastret, Liria, Castellón, Alcoy, Valencia, Huesca, Murcia |
| **Structural patterns** | 5 | [name]:ekiar, [name]:seltar:ḿi, neitin+iunstir, [name]:eban, śalir+numeral |

**IMPORTANT CAVEAT:** Iberian is an undeciphered language. All semantic interpretations are scholarly proposals based on archaeological context, find-spots (sanctuaries, necropoleis), and structural analysis — not confirmed translations.

## Category Definitions

We use a deliberately **broad** definition of "religious":

1. **Deity names & epithets** — Names of gods, divine titles, celestial beings
2. **Theophoric names** — Personal names containing deity references (invoked in prayer)
3. **Ritual verbs** — Actions of worship: sacrifice, pray, bless, vow, purify, anoint, fast, pour libations
4. **Ritual nouns** — Objects and places of worship: altars, temples, offerings, incense
5. **Theological concepts** — Abstract sacred ideas: holiness, sin, grace, righteousness, soul, heaven, hell
6. **Votive formulas** — Dedicatory patterns: "X made this", "X gave this to deity Y"
7. **High-risk activities** — Domains requiring divine intervention:
   - **Fertility**: childbirth, firstborn, barrenness
   - **Agriculture**: harvest, sowing, rain, dew, vineyard, grain, oil
   - **Livestock**: sacrificial animals (bull, goat, sheep, calf, ox)
   - **Conflict**: war, battle, enemies
   - **Health**: sickness, healing
   - **Travel**: sea voyages, journeys, roads
   - **Sustenance**: bread, wine, water

## Usage Notes

- All TSV files use tab separation; lines beginning with `#` are comments
- The Ugaritic-Hebrew file preserves the original transliteration scheme from the NeuroDecipher cognate dataset
- The Gothic file includes one example verse reference per term for verification
- The Iberian file includes Hesperia reference codes (e.g., GI.15.09) that map back to `iberian.csv`
- Cross-language comparison is possible via the `category` and `subcategory` columns, which use a shared taxonomy

## Sources

- Ugaritic-Hebrew cognates: [j-luo93/NeuroDecipher](https://github.com/j-luo93/NeuroDecipher) (Snyder et al. 2010)
- Gothic Bible: [Project Wulfila](https://www.wulfila.be/) (Streitberg 1919 edition)
- Iberian inscriptions: [Hesperia database](http://hesperia.ucm.es/) via [j-luo93/DecipherUnsegmented](https://github.com/j-luo93/DecipherUnsegmented)
- Iberian interpretation: Untermann 1990; Velaza 2015; Rodriguez Ramos 2014
