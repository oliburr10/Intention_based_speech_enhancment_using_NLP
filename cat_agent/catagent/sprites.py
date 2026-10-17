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
