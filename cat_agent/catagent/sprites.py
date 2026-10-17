"""Pixel-art frames for the cat, drawn from ASCII grids so there are no image files to ship."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPixmap

PALETTE = {
    "K": "#3b2314",  # outline
    "O": "#ec8b33",  # ginger
    "D": "#b8601c",  # ginger stripes
    "W": "#fff6ea",  # white
    "S": "#e6d9c8",  # white shadow
    "P": "#f2a0b0",  # pink (nose, inner ears)
    "E": "#2b2b2b",  # eyes
    "G": "#ffffff",  # eye shine
    "F": "#1d1d1f",  # glasses frame
    "L": "#cfe8ff",  # glasses lens
    "C": "#8fd0ff",  # lens lit up by the screen
    "N": "#5b6570",  # laptop edge
    "M": "#a9b2bc",  # laptop lid
    "H": "#e8edf2",  # laptop logo
}

# Sitting cat, facing forward, tail curled on the right.
BASE = [
    "..KK........KK......",
    ".KOPK......KPOK.....",
    ".KOPPKKKKKKPPOK.....",
    ".KOOODOODOODOOOK....",
    "KOOOOOOOOOOOOOOK....",
    "KOOEGOOOOOOEGOOK....",
    "KOOEEOOOOOOEEOOK....",
    "KOWWWWWPPWWWWWOK....",
    "KWWWWWKWWKWWWWWK....",
    ".KWWWWWKKWWWWWK.....",
    "..KKWWWWWWWWKK......",
    "..KOOWWWWWWOOK......",
    ".KOOOWWWWWWOOOK.....",
    ".KODOWWWWWWODOK..KK.",
    ".KOOOWWWWWWOOOK.KOOK",
    ".KODOWWWWWWODOK.KDK.",
    ".KOOWSWWWWSWOOKKOK..",
    ".KOWWKWWWWKWWOOOOK..",
    "..KKKKKKKKKKKKKKK...",
]


def _with_rows(base, rows):
    out = list(base)
    for i, row in rows.items():
        out[i] = row
    return out


FRAMES = {
    "idle": BASE,
    "blink": _with_rows(BASE, {
        5: "KOOOOOOOOOOOOOOK....",
        6: "KOOKKOOOOOOKKOOK....",
    }),
    "tail": _with_rows(BASE, {
        11: "..KOOWWWWWWOOK...KK.",
        12: ".KOOOWWWWWWOOOK.KOOK",
        13: ".KODOWWWWWWODOK.KDK.",
        14: ".KOOOWWWWWWOOOK.KOK.",
        15: ".KODOWWWWWWODOK.KOK.",
        16: ".KOOWSWWWWSWOOKKOK..",
    }),
    "look_up": _with_rows(BASE, {
        5: "KOOEEOOOOOOEEOOK....",
        6: "KOOEGOOOOOOEGOOK....",
    }),
}
FRAMES["sleep"] = FRAMES["blink"]

# Coding: glasses on, eyes down at the laptop, paws tapping away at the keyboard.
_GLASSES = {
    4: "KOFFFFOOOOFFFFOK....",
    5: "KOFLLFFFFFFLLFOK....",
    6: "KOFLEFOOOOFLEFOK....",
    7: "KOFFFFWPPWFFFFOK....",
}
_LAPTOP = {
    11: "..KOOWWWWWWOOK......",
    12: ".KONNNNNNNNNNOK.....",
    13: ".KONMMMMMMMMNOK..KK.",
    14: ".KONMMHMMHMMNOK.KOOK",
    15: ".KONMMHHHHMMNOK.KDK.",
    16: ".KONMMMHHMMMNOK.KOK.",
    17: "NNNNNNNNNNNNNNNNNK..",
    18: ".NNNNNNNNNNNNNNNN...",
}
# Each paw is a 2x2 white blob over the edge of the lid; up on one side, down on the other.
_PAWS_LEFT_UP = {
    13: ".KWWMMMMMMMMNOK..KK.",
    14: ".KWWMMHMMHMMNOK.KOOK",
    15: ".KONMMHHHHMMWWK.KDK.",
    16: ".KONMMMHHMMMWWK.KOK.",
}
_PAWS_RIGHT_UP = {
    13: ".KONMMMMMMMMWWK..KK.",
    14: ".KONMMHMMHMMWWK.KOOK",
    15: ".KWWMMHHHHMMNOK.KDK.",
    16: ".KWWMMMHHMMMNOK.KOK.",
}
FRAMES["code_a"] = _with_rows(BASE, {**_GLASSES, **_LAPTOP, **_PAWS_LEFT_UP})
FRAMES["code_b"] = _with_rows(BASE, {**_GLASSES, **_LAPTOP, **_PAWS_RIGHT_UP})
FRAMES["code_glow"] = _with_rows(FRAMES["code_a"], {
    5: "KOFCCFFFFFFCCFOK....",
    6: "KOFCEFOOOOFCEFOK....",
})

WIDTH = max(len(r) for r in BASE)
HEIGHT = len(BASE)


def render(frame: str, scale: int) -> QPixmap:
    grid = FRAMES[frame]
    pm = QPixmap(WIDTH * scale, HEIGHT * scale)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            if ch in PALETTE:
                p.fillRect(x * scale, y * scale, scale, scale, QColor(PALETTE[ch]))
    p.end()
    return pm
