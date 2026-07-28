"""Mapping between the CNN's 64 dot-pattern classes and Grade-1 English Braille letters.

Kept separate from the model: the CNN only ever predicts a 0-63 dot-pattern
class, and this lookup table decodes that class to a letter (or special
symbol) where one exists. Patterns with no assigned meaning are labeled
"#<code>".
"""

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
    code = 0
    for d in dots:
        code |= 1 << (d - 1)
    return code


CODE_TO_LETTER = {dots_to_code(dots): name for name, dots in {**LETTER_DOTS, **SPECIAL_DOTS}.items()}
LETTER_TO_CODE = {name: code for code, name in CODE_TO_LETTER.items()}

# Sinhala letter for each dot-pattern code, transcribed from Figure 12 ("OBT
# Database for Decoding Braille Characters") in docs/sinhala paper.pdf, page 8.
# Only codes with an unambiguous, cleanly printed Sinhala glyph are included.
# Left out on purpose:
#   - punctuation-only cells (comma, semicolon, quotes, brackets, etc.)
#   - Grade-2 English word contractions (the/and/wh/ou/of/er/with/for/...)
#     which the chart pairs with a Sinhala-looking mark that is ambiguous
#     and not needed for single-character Sinhala output
#   - combining Sinhala vowel-sign codes (roughly 4, 8, 12, 18, 20, 28, 30,
#     31, 34, 35, 40, 42-56, 59, 61-63) whose glyphs render as broken/
#     placeholder dotted-circle marks in the source PDF itself (pdftoppm
#     warns of missing "Symbol"/"ArialUnicode" fonts when rasterizing this
#     file) and could not be read reliably even at 500 DPI.
# This was transcribed by eye from a small academic chart, not copied from a
# machine-readable source -- please spot check against the PDF (or a native
# Sinhala reader) before relying on it for real TTS output.
CODE_TO_SINHALA = {
    1: "අ",   # a
    3: "බ",   # b
    5: "ක",   # k
    7: "ල",   # l
    9: "ව",   # c
    10: "ඉ",  # i
    11: "ෆ",  # f
    13: "ම",  # m
    14: "ස",  # s
    15: "ප",  # p
    17: "එ",  # e
    19: "හ",  # h
    21: "ඔ",  # o
    23: "ර",  # r
    25: "ද",  # d
    26: "ජ",  # j
    27: "ග",  # g
    29: "න",  # n
    33: "ච",  # ch
    37: "උ",  # u
    39: "ව",  # v (Sinhala has one letter for the v/w sound)
    41: "ශ",  # sh
    57: "ථ",  # th
    58: "ව",  # w
}


def code_to_label(code: int, lang: str = "en") -> str:
    """Decode a predicted dot-pattern code to a display label.

    lang="en" uses the Grade-1 English letter table; lang="si" uses the
    Sinhala table above. Codes with no entry in the requested language fall
    back to "#<code>".
    """
    table = CODE_TO_SINHALA if lang == "si" else CODE_TO_LETTER
    return table.get(code, f"#{code}")
