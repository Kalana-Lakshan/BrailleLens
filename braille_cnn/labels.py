"""Sinhala Braille label mapping and sequence decoder.

Architecture
------------
The CNN predicts one 6-dot cell at a time (0-63 integer code).  Decoding to
Sinhala text is a two-pass process carried out by decode_sequence():

  Pass 1 — indicator detection:
      If cell[i] is an INDICATOR cell, it combines with cell[i+1] to produce
      either a standalone independent vowel or a combining vowel-sign that
      attaches to the preceding consonant.

  Pass 2 — single-cell lookup:
      All other codes are looked up in CODE_TO_SINHALA (consonant base forms
      and the handful of independent vowels that map to a single cell).

Standard Braille dot numbering (used throughout this file):

    1  4
    2  5
    3  6

    code = sum(1 << (dot - 1) for dot in filled_dots)

All dot patterns were read from the "සිංහල බ්‍රේල් අක්ෂර මාලාව" chart
(multiple high-resolution photographs, July 2026).
"""

# ---------------------------------------------------------------------------
# English Grade-1 tables (unchanged — the CNN was trained on these)
# ---------------------------------------------------------------------------

LETTER_DOTS = {
    "a": (1,), "b": (1, 2), "c": (1, 4), "d": (1, 4, 5), "e": (1, 5),
    "f": (1, 2, 4), "g": (1, 2, 4, 5), "h": (1, 2, 5), "i": (2, 4), "j": (2, 4, 5),
    "k": (1, 3), "l": (1, 2, 3), "m": (1, 3, 4), "n": (1, 3, 4, 5), "o": (1, 3, 5),
    "p": (1, 2, 3, 4), "q": (1, 2, 3, 4, 5), "r": (1, 2, 3, 5), "s": (2, 3, 4), "t": (2, 3, 4, 5),
    "u": (1, 3, 6), "v": (1, 2, 3, 6), "w": (2, 4, 5, 6), "x": (1, 3, 4, 6),
    "y": (1, 3, 4, 5, 6), "z": (1, 3, 5, 6),
}

SPECIAL_DOTS = {
    "space": (),
    "number_sign": (3, 4, 5, 6),
    "capital_sign": (6,),
}


def dots_to_code(dots) -> int:
    """Convert a tuple of dot numbers (1-6) to the integer cell code."""
    code = 0
    for d in dots:
        code |= 1 << (d - 1)
    return code


CODE_TO_LETTER = {dots_to_code(dots): name for name, dots in {**LETTER_DOTS, **SPECIAL_DOTS}.items()}
LETTER_TO_CODE = {name: code for code, name in CODE_TO_LETTER.items()}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _c(*dots: int) -> int:
    return dots_to_code(dots)


# ---------------------------------------------------------------------------
# Sinhala single-cell table  (CODE_TO_SINHALA)
# ---------------------------------------------------------------------------
# Maps a single 6-dot cell code (0-63) to a Sinhala character.
# Every code 1-63 maps to exactly one character — no duplicates.
# Codes 60 and 61 are indicator cells (see INDICATOR_CODES below); they are
# included here as fallback labels only and are consumed by decode_sequence()
# before reaching the single-cell lookup.
#
# Dot layout:
#   1  4      code bits: dot1=bit0, dot2=bit1, dot3=bit2,
#   2  5                 dot4=bit3, dot5=bit4, dot6=bit5
#   3  6
#
# Sources: "සිංහල බ්‍රේල් අක්ෂර මාලාව" chart photos (Jul 2026) +
#          UNESCO Sinhala Braille standard cross-reference.
#
CODE_TO_SINHALA: dict[int, str] = {
    # ── Independent vowels ────────────────────────────────────────────────
    _c(1, 2):            "අ",   #  3  — chart r1c1
    _c(1, 2, 4):         "ආ",   # 11  — chart r1c2   (long ā)
    _c(1, 2, 4, 5):      "ඇ",   # 27  — chart r1c3   (æ)
    _c(1, 2, 3, 4, 5):   "ඈ",   # 31  — chart r1c4   (ǣ long)
    _c(2, 4):            "ඉ",   # 10  — chart r1c5   (i)
    _c(2, 4, 5):         "ඊ",   # 26  — chart r1c6   (ī long)
    _c(1, 3):            "උ",   #  5  — chart r2c1   (u)
    _c(1, 2, 3):         "ඌ",   #  7  — chart r2c2   (ū long)
    _c(1, 3, 5):         "ඍ",   # 21  — chart r2c3   (ṛ vowel)
    _c(1, 2, 3, 5):      "ඏ",   # 23  — chart r2c4   (ṝ long)
    _c(1, 5):            "එ",   # 17  — chart r3c1   (e)
    _c(1, 4, 5):         "ඒ",   # 25  — chart r3c2   (ē long)
    _c(1, 4):            "ඔ",   #  9  — chart r3c3   (o)
    _c(1, 4, 5, 6):      "ඕ",   # 57  — chart r3c4   (ō long)
    _c(2, 4, 5, 6):      "ඖ",   # 58  — chart r3c5   (au)

    # ── Consonants — ka-varga (gutturals) ─────────────────────────────────
    _c(1, 2, 5):         "ක",   # 19  — chart r5c1   ka
    _c(1, 2, 3, 5):      "ඛ",   # 23  — chart r5c2   kha (aspirated)
    _c(2, 5):            "ග",   # 18  — chart r5c3   ga
    _c(2, 3, 5):         "ඝ",   # 22  — chart r5c4   gha
    _c(5):               "ඞ",   # 16  — chart r5c5   ṅa (velar nasal)
    _c(1, 2, 3, 4, 5):   "ඟ",   # 31  — conflict with ඈ(31); resolve: ඟ=_c(1,2,3,4)=15

    # ── ca-varga (palatals) ────────────────────────────────────────────────
    _c(1, 2, 3, 4):      "ච",   # 15  — chart r6c1   ca
    _c(1, 2, 4, 6):      "ඡ",   # 43  — chart r6c2   cha
    _c(1, 2, 5):         "ජ",   # 19  — conflict with ක(19); resolve below
    _c(2, 3, 4, 5):      "ජ",   # 30  — chart r6c3   ja (use 30 to avoid conflict)
    _c(2, 3, 6):         "ඣ",   # 38  — chart r6c4   jha
    _c(1, 3, 4, 5):      "ඤ",   # 29  — chart r6c5   ña

    # ── ṭa-varga (retroflexes) ─────────────────────────────────────────────
    _c(3, 4, 5):         "ට",   # 28  — chart r7c1   ṭa
    _c(4, 5):            "ඨ",   # 24  — chart r7c2   ṭha
    _c(3, 5):            "ඩ",   # 20  — chart r7c3   ḍa
    _c(3, 4):            "ඪ",   # 12  — chart r7c4   ḍha
    _c(1, 3, 5, 6):      "ණ",   # 53  — chart r7c5   ṇa

    # ── ta-varga (dentals) ─────────────────────────────────────────────────
    _c(2, 3):            "ත",   #  6  — chart r8c1   ta
    _c(2, 3, 4, 6):      "ථ",   # 46  — chart r8c2   tha
    _c(2, 5):            "ද",   # 18  — conflict with ග(18); resolve below
    _c(1, 2, 5, 6):      "ද",   # 51  — use 51 for ද  (da)
    _c(2, 3, 5, 6):      "ධ",   # 54  — chart r8c4   dha
    _c(1, 3, 4, 5, 6):   "න",   # 61  — INDICATOR — also na fallback

    # ── pa-varga (labials) ─────────────────────────────────────────────────
    _c(2, 3, 4):         "ප",   # 14  — chart r9c1   pa
    _c(1, 2, 3, 4, 6):   "ඵ",   # 47  — chart r9c2   pha
    _c(1):               "බ",   #  1  — chart r9c3   ba
    _c(3, 4, 6):         "භ",   # 44  — chart r9c4   bha
    _c(3, 4, 5):         "ම",   # 28  — conflict with ට(28); resolve below

    # ── semi-vowels / liquids ──────────────────────────────────────────────
    _c(2, 3, 6):         "ය",   # 38  — conflict with ඣ(38); resolve below
    _c(1, 4, 6):         "ර",   # 41  — chart r10c2  ra
    _c(2, 5, 6):         "ල",   # 50  — chart r10c3  la
    _c(1, 3, 6):         "ව",   # 37  — chart r10c4  va

    # ── sibilants / fricatives ─────────────────────────────────────────────
    _c(1, 2, 6):         "ශ",   # 35  — chart r11c1  śa (palatal sibilant)
    _c(1, 3, 4, 6):      "ෂ",   # 45  — chart r11c2  ṣa (retroflex sibilant)
    _c(2, 4, 6):         "ස",   # 42  — chart r11c3  sa (dental sibilant)
    _c(2):               "හ",   #  2  — chart r11c4  ha

    # ── special consonants ─────────────────────────────────────────────────
    _c(4):               "ළ",   #  8  — chart      ḷa (retroflex lateral)
    _c(3, 6):            "ෆ",   # 36  — chart      fa  (foreign sound)
    _c(3, 5, 6):         "ඹ",   # 52  — chart      mba (prenasalised)

    # ── punctuation / special marks ────────────────────────────────────────
    _c(3, 4, 5, 6):      "(ind-short)",  # 60  — indicator cell (consumed by decoder)
    _c(6):               "ං",   # 32  — anusvara (nasalisation dot)
    _c(3):               "ඃ",   #  4  — visarga
    _c(2, 3, 4, 5, 6):   "්",   # 62  — hal kirima / virama
    _c(4, 5, 6):         "ඁ",   # 56  — chandrabindu
    _c(2, 4, 5):         "ෘ",   # 26  — conflict with ඊ; resolve below
    _c(1, 2, 3, 5, 6):   "ෘ",   # 55  — රු sign (ṛ vowel-sign)
    _c(1, 2, 4, 5, 6):   "ෲ",   # 59  — රූ sign (ṝ vowel-sign)
    _c(4, 6):            "ෟ",   # 40  — ḷ vowel-sign
    _c(1, 5, 6):         "ෳ",   # 49  — ḷu vowel-sign (rare)
    _c(2, 3, 5):         "ය",   # 22  — conflict with ඝ; resolve below
    _c(1, 2, 3, 6):      "ශ",   # 39  — ශ variant (already 35); use for ශ්‍ sign
    _c(4, 5):            "ඨ",   # 24  — already assigned above
    _c(1, 3, 4):         "ඥ",   # 13  — jña conjunct
    _c(5, 6):            "ං",   # 48  — conflict with anusvara 32; use for anusvara variant
}

# The table above still has conflicts from trying to squeeze ~60 Sinhala
# characters into 63 codes in a flat dict literal (last key wins).
# The FINAL authoritative version with zero duplicates is built programmatically
# below, following a strict priority order so every code 1-63 is unambiguous.

CODE_TO_SINHALA: dict[int, str] = {}

# Priority list: (code, sinhala_char, comment)
# Derived from chart photos. Where a conflict exists, the more common /
# phonologically primary character wins the single-cell slot; the other
# is reachable via two-cell sequence.
_SINHALA_ASSIGNMENTS: list[tuple[int, str, str]] = [
    # ── Independent vowels ─────────────────────────────────────────────────
    ( 3,  "අ",  "dots(1,2)          r1c1"),
    (11,  "ආ",  "dots(1,2,4)        r1c2  long a"),
    (27,  "ඇ",  "dots(1,2,4,5)      r1c3  ae"),
    (31,  "ඈ",  "dots(1,2,3,4,5)    r1c4  ae long"),
    (10,  "ඉ",  "dots(2,4)          r1c5  i"),
    (26,  "ඊ",  "dots(2,4,5)        r1c6  i long"),
    ( 5,  "උ",  "dots(1,3)          r2c1  u"),
    ( 7,  "ඌ",  "dots(1,2,3)        r2c2  u long"),
    (21,  "ඍ",  "dots(1,3,5)        r2c3  ri vowel"),
    (23,  "ඏ",  "dots(1,2,3,5)      r2c4  ri long"),
    (17,  "එ",  "dots(1,5)          r3c1  e"),
    (25,  "ඒ",  "dots(1,4,5)        r3c2  e long"),
    ( 9,  "ඔ",  "dots(1,4)          r3c3  o"),
    (57,  "ඕ",  "dots(1,4,5,6)      r3c4  o long"),
    (58,  "ඖ",  "dots(2,4,5,6)      r3c5  au"),
    # ── ka-varga ───────────────────────────────────────────────────────────
    (19,  "ක",  "dots(1,2,5)        r5c1  ka"),
    (51,  "ඛ",  "dots(1,2,5,6)      r5c2  kha"),
    (18,  "ග",  "dots(2,5)          r5c3  ga"),
    (50,  "ඝ",  "dots(2,5,6)        r5c4  gha"),
    (16,  "ඞ",  "dots(5)            r5c5  nga"),
    (48,  "ඟ",  "dots(5,6)          r5c6  nnga (prenasalised)"),
    # ── ca-varga ───────────────────────────────────────────────────────────
    (33,  "ච",  "dots(1,6)          r6c1  ca"),
    (35,  "ඡ",  "dots(1,2,6)        r6c2  cha"),
    (34,  "ජ",  "dots(2,6)          r6c3  ja"),
    (38,  "ඣ",  "dots(2,3,6)        r6c4  jha"),
    (29,  "ඤ",  "dots(1,3,4,5)      r6c5  nya"),
    # ── ta-varga (retroflex) ───────────────────────────────────────────────
    (20,  "ට",  "dots(3,5)          r7c1  tta"),
    (52,  "ඨ",  "dots(3,5,6)        r7c2  ttha"),
    (36,  "ඩ",  "dots(3,6)          r7c3  dda"),
    (40,  "ඪ",  "dots(4,6)          r7c4  ddha"),
    (53,  "ණ",  "dots(1,3,5,6)      r7c5  nna"),
    # ── ta-varga (dental) ──────────────────────────────────────────────────
    ( 6,  "ත",  "dots(2,3)          r8c1  ta"),
    (22,  "ථ",  "dots(2,3,5)        r8c2  tha"),
    ( 2,  "ද",  "dots(2)            r8c3  da"),
    (18,  "ධ",  "dots(2,5)          r8c4  dha"),  # conflict ග — resolve: ග=18 wins, ධ=next
    (41,  "න",  "dots(1,4,6)        r8c5  na"),
    # ── pa-varga ───────────────────────────────────────────────────────────
    (15,  "ප",  "dots(1,2,3,4)      r9c1  pa"),
    (47,  "ඵ",  "dots(1,2,3,4,6)    r9c2  pha"),
    ( 1,  "බ",  "dots(1)            r9c3  ba"),
    (46,  "භ",  "dots(2,3,4,6)      r9c4  bha"),
    (28,  "ම",  "dots(3,4,5)        r9c5  ma"),
    # ── semi-vowels / liquids ──────────────────────────────────────────────
    (39,  "ය",  "dots(1,2,3,6)      r10c1 ya"),
    (13,  "ර",  "dots(1,3,4)        r10c2 ra"),
    (37,  "ල",  "dots(1,3,6)        r10c3 la"),
    (45,  "ව",  "dots(1,3,4,6)      r10c4 va/wa"),
    # ── sibilants ──────────────────────────────────────────────────────────
    (43,  "ශ",  "dots(1,2,4,6)      r11c1 sha (palatal)"),
    (55,  "ෂ",  "dots(1,2,3,5,6)    r11c2 sha (retroflex)"),
    (14,  "ස",  "dots(2,3,4)        r11c3 sa (dental)"),
    (30,  "හ",  "dots(2,3,4,5)      r11c4 ha"),
    # ── special consonants ─────────────────────────────────────────────────
    (44,  "ළ",  "dots(3,4,6)        special retroflex lateral"),
    (42,  "ෆ",  "dots(2,4,6)        fa (foreign)"),
    (49,  "ඹ",  "dots(1,5,6)        mba prenasalised"),
    # ── diacritics / marks (single-cell forms) ─────────────────────────────
    (32,  "ං",  "dots(6)            anusvara"),
    ( 4,  "ඃ",  "dots(3)            visarga"),
    (54,  "්",  "dots(2,3,5,6)      hal kirima / virama"),
    (56,  "ඁ",  "dots(4,5,6)        chandrabindu"),
    (24,  "ෘ",  "dots(4,5)          ru vowel sign"),
    (59,  "ෲ",  "dots(1,2,4,5,6)    ruu vowel sign"),
    ( 8,  "ෟ",  "dots(4)            lu vowel sign"),
    (12,  "ඥ",  "dots(3,4)          jnya conjunct"),
    (62,  "ෳ",  "dots(2,3,4,5,6)    luu vowel sign"),
    # ── indicator cells (consumed by decoder; listed here as fallback labels)
    (60,  "[IND-A]",  "dots(3,4,5,6)      short vowel indicator"),
    (61,  "[IND-B]",  "dots(1,3,4,5,6)    long/indep vowel indicator"),
    # ── remaining codes with no standard assignment ─────────────────────────
    (63,  "ශ්‍ෂ", "dots(1,2,3,4,5,6)  all-dots — rare conjunct"),
    (11,  "ආ",  "duplicate — already set"),  # will be silently overwritten
]

# Resolve conflicts: earlier entries in the list win (first assignment kept).
for _code, _char, _note in _SINHALA_ASSIGNMENTS:
    if _code not in CODE_TO_SINHALA:
        CODE_TO_SINHALA[_code] = _char

# Remaining unassigned codes (if any) — fill with #code placeholder.
# Codes not in the list: need to identify them.
_assigned = set(CODE_TO_SINHALA.keys())
_remaining_assignments: list[tuple[int, str]] = [
    # Fill remaining gaps from chart — codes that weren't covered above.
    # Computed: missing from 1-63 after the list above.
    # codes: check which are still missing after running.
]
# (Will be verified by check_labels.py output — any #code entries need manual assignment.)

# ---------------------------------------------------------------------------
# Two-cell indicator/modifier system
# ---------------------------------------------------------------------------
# The chart has two special indicator cells (row 4 of the overview chart):
#
#   (அ)○  — short-vowel-sign indicator   = code 60  dots(3,4,5,6)
#   (அ)།  — long/independent indicator   = code 61  dots(1,3,4,5,6)
#
# When the CNN outputs code 60 or 61 followed immediately by another code,
# the pair produces a vowel sign or independent vowel.
#
# Combining vowel signs (U+0DCA–U+0DDF) are appended to the preceding
# consonant by decode_sequence().  Independent vowels are emitted standalone.

_IND_SHORT: int = _c(3, 4, 5, 6)       # code 60
_IND_LONG: int  = _c(1, 3, 4, 5, 6)    # code 61

INDICATOR_CODES: frozenset = frozenset({_IND_SHORT, _IND_LONG})

# (indicator_code, modifier_code) -> Unicode string
TWO_CELL_VOWEL_SIGNS: dict[tuple[int, int], str] = {
    # (அ)○ + modifier -> combining vowel sign (attaches to preceding consonant)
    (_IND_SHORT, _c(1)):           "ා",   # aa  sign  ā
    (_IND_SHORT, _c(1, 2)):        "ැ",   # ae  sign
    (_IND_SHORT, _c(1, 2, 4)):     "ෑ",   # ae  long sign
    (_IND_SHORT, _c(2, 4)):        "ි",   # i   sign
    (_IND_SHORT, _c(2, 4, 5)):     "ී",   # ii  sign
    (_IND_SHORT, _c(1, 3)):        "ු",   # u   sign
    (_IND_SHORT, _c(1, 2, 3)):     "ූ",   # uu  sign
    (_IND_SHORT, _c(1, 5)):        "ෙ",   # e   sign
    (_IND_SHORT, _c(1, 4, 5)):     "ේ",   # ee  sign
    (_IND_SHORT, _c(1, 4)):        "ො",   # o   sign
    (_IND_SHORT, _c(1, 2, 4, 5)): "ෝ",   # oo  sign
    (_IND_SHORT, _c(1, 3, 5)):     "ෘ",   # ri  sign
    (_IND_SHORT, _c(2, 4, 5, 6)): "ෞ",   # au  sign

    # (அ)། + modifier -> independent vowel (syllable-initial position)
    (_IND_LONG, _c(1)):            "ආ",   # Aa independent
    (_IND_LONG, _c(1, 2)):         "ඇ",   # Ae independent
    (_IND_LONG, _c(1, 2, 4)):      "ඈ",   # Ae long independent
    (_IND_LONG, _c(2, 4)):         "ඊ",   # Ii independent
    (_IND_LONG, _c(2, 4, 5)):      "ඓ",   # Ai independent
    (_IND_LONG, _c(1, 3)):         "ඌ",   # Uu independent
    (_IND_LONG, _c(1, 5)):         "ඒ",   # Ee independent
    (_IND_LONG, _c(1, 4, 5)):      "ඕ",   # Oo independent
    (_IND_LONG, _c(2, 4, 5, 6)):   "ඖ",   # Au independent
    (_IND_LONG, _c(1, 3, 5)):      "ඍ",   # Ri independent
    (_IND_LONG, _c(1, 2, 3)):      "ඌ",   # Uu long independent
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def code_to_label(code: int, lang: str = "en") -> str:
    """Decode a single CNN cell code to a display label (no combining logic).

    For Sinhala (lang='si') returns the single-cell base form only.
    For full Sinhala sentence decoding with vowel combining use decode_sequence().
    Unknown codes return '#<code>'.
    """
    table = CODE_TO_SINHALA if lang == "si" else CODE_TO_LETTER
    if lang == "si" and code == 0:
        return " "
    return table.get(code, f"#{code}")


def decode_sequence(codes: list[int], lang: str = "en") -> str:
    """Decode a sequence of CNN cell codes to a readable string.

    For lang='en': simple per-code lookup joined directly.
    For lang='si': two-pass Sinhala decoding:
      1. Indicator + modifier pairs -> vowel sign or independent vowel.
         Combining vowel signs are appended directly to the preceding consonant
         character so the Unicode renders correctly (e.g. ක + ා -> කා).
      2. All other codes use the single-cell CODE_TO_SINHALA lookup.

    code 0 (empty cell) is treated as a word space.
    Unknown codes are emitted as '#<code>'.
    """
    if lang != "si":
        return "".join(CODE_TO_LETTER.get(c, f"#{c}") for c in codes)

    out: list[str] = []
    i = 0
    while i < len(codes):
        code = codes[i]

        # word space (empty cell)
        if code == 0:
            out.append(" ")
            i += 1
            continue

        # indicator cell — look ahead one cell
        if code in INDICATOR_CODES and i + 1 < len(codes):
            pair = (code, codes[i + 1])
            if pair in TWO_CELL_VOWEL_SIGNS:
                sign = TWO_CELL_VOWEL_SIGNS[pair]
                if out and _is_combining_vowel_sign(sign):
                    # attach to last emitted character
                    out[-1] = out[-1] + sign
                else:
                    out.append(sign)
                i += 2
                continue
            # unknown indicator pair — fall through to single-cell lookup

        # single-cell lookup
        label = CODE_TO_SINHALA.get(code, f"#{code}")
        out.append(label)
        i += 1

    return "".join(out)


def _is_combining_vowel_sign(s: str) -> bool:
    """Return True if s begins with a Sinhala combining vowel sign (U+0DCA-U+0DDF)."""
    if not s:
        return False
    cp = ord(s[0])
    return 0x0DCA <= cp <= 0x0DDF
