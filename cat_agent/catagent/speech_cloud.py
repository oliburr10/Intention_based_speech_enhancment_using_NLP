"""The pixel-art speech cloud the cat talks in: results, tips and "can I do this?" questions."""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .cat_widget import floating_window_flags

U = 4                         # screen pixels per cloud pixel
DOME_ROWS = 6                 # the top of the cloud steps in over this many rows...
DOME_WIDTH = 0.28             # ...starting this fraction of the width in from each side
BOTTOM = [1, 2]               # how far the last rows are pulled in, rounding the bottom corners
TAIL_ROWS = 4                 # the tail hangs this many pixels below the cloud, pointing at the cat
MIN_W, MAX_W = 150, 320

OUTLINE = QColor("#141414")
FILL = QColor("#ffffff")
SHADOW = QColor("#ebe3d7")
TEXT = "#141414"

BUTTON_STYLE = """
QPushButton { background: #141414; color: white; border: none; padding: 4px 12px;
              font-weight: bold; font-size: 12px; }
QPushButton:hover { background: #3b2314; }
QPushButton#secondary { background: #ebe3d7; color: #141414; }
QPushButton#secondary:hover { background: #d9cfc0; }
"""


def dome_inset(w: int, y: int) -> int:
    """How far row y of the cloud's top is pulled in: a stepped dome, like a pixel cloud."""
    if y >= DOME_ROWS:
        return 0
    return round(DOME_WIDTH * w * (1 - y / DOME_ROWS) ** 1.5)


def cloud_cells(w: int, h: int, tail_x: int) -> set[tuple[int, int]]:
    """The cells (in cloud pixels) covered by a cloud w wide and h tall, including its tail."""
    body_h = h - TAIL_ROWS
    cells = set()
    for y in range(body_h):
        inset = dome_inset(w, y)
        from_bottom = body_h - 1 - y
        if from_bottom < len(BOTTOM):
            inset = max(inset, BOTTOM[::-1][from_bottom])
        cells.update((x, y) for x in range(inset, w - inset))
    for r in range(TAIL_ROWS):
        cells.update((x, body_h + r) for x in range(tail_x + r, tail_x + TAIL_ROWS))
    return cells


class SpeechCloud(QWidget):
    answered = Signal(bool)  # the Allow / Don't buttons
    show_me = Signal()
    closed = Signal()

    def __init__(self):
        super().__init__(None, floating_window_flags())
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setStyleSheet(BUTTON_STYLE)
        self._tail_x = 10
        self._sticky = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4 * U + 6, (DOME_ROWS - 1) * U, 4 * U + 6, (TAIL_ROWS + 2) * U + 8)
        layout.setSpacing(4)
        self._title = QLabel()
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPixelSize(15)
        self._title.setFont(title_font)
        self._body = QLabel()
        self._body.setWordWrap(True)
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        for label in (self._title, self._body):
            label.setStyleSheet(f"color: {TEXT}; background: transparent;")
        self._body.setStyleSheet(f"color: {TEXT}; background: transparent; font-size: 13px;")
        layout.addWidget(self._title)
        layout.addWidget(self._body)

        self._buttons = QHBoxLayout()
        self._buttons.addStretch()
        self._dont = self._button("Don't", "secondary", lambda: self._answer(False))
        self._allow = self._button("Allow", "", lambda: self._answer(True))
        self._show_me = self._button("Show me", "", self._on_show_me)
        layout.addLayout(self._buttons)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)

    def _button(self, text: str, name: str, slot) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(name)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(slot)
        button.hide()
        self._buttons.addWidget(button)
        return button

    # ---- what to say ---------------------------------------------------------------------

    def say(self, title: str, body: str = "", *, ask: bool = False, file_button: bool = False) -> None:
        """Fill the cloud. ask=True shows Allow/Don't and keeps it up until answered."""
        self._title.setText(title)
        self._title.setVisible(bool(title))
        self._body.setText(body)
        self._body.setVisible(bool(body))
        self._allow.setVisible(ask)
        self._dont.setVisible(ask)
        self._show_me.setVisible(file_button and not ask)
        self._sticky = ask or file_button
        self._fit()
        self._hide_timer.stop()
        if not self._sticky:
            self._hide_timer.start(min(30_000, 6_000 + 60 * len(title + body)))

    def _fit(self) -> None:
        margins = self.layout().contentsMargins()
        fm_title = QFontMetrics(self._title.font())
        fm_body = QFontMetrics(self._body.font())
        # isHidden, not isVisible: the cloud itself may not be on screen yet.
        natural = max(
            fm_title.horizontalAdvance(self._title.text()) + 8 if not self._title.isHidden() else 0,
            fm_body.horizontalAdvance(self._body.text()) if not self._body.isHidden() else 0,
            150 if not (self._allow.isHidden() and self._show_me.isHidden()) else 0,
        )
        width = min(MAX_W, max(MIN_W, natural + margins.left() + margins.right() + 4))
        width += -width % U
        self.setFixedWidth(width)
        layout = self.layout()
        layout.activate()
        # Only a word-wrapped body makes the height depend on the width.
        height = layout.heightForWidth(width) if layout.hasHeightForWidth() else layout.sizeHint().height()
        height += -height % U
        self.setFixedHeight(height)

    def point_tail_at(self, x: int) -> None:
        """Aim the tail at a screen x position (call after moving the cloud)."""
        w = self.width() // U
        self._tail_x = max(4, min(w - TAIL_ROWS - 6, (x - self.x()) // U - TAIL_ROWS))
        self.update()

    # ---- interaction -----------------------------------------------------------------------

    def _answer(self, allow: bool) -> None:
        self.hide()
        self.answered.emit(allow)

    def _on_show_me(self) -> None:
        self.hide()
        self.show_me.emit()

    def hideEvent(self, event):
        self.closed.emit()
        super().hideEvent(event)

    def mousePressEvent(self, event):
        if self._allow.isHidden():  # a plain message: click to dismiss
            self.hide()

    # ---- drawing ---------------------------------------------------------------------------

    def paintEvent(self, event):
        w, h = self.width() // U, self.height() // U
        cells = cloud_cells(w, h, self._tail_x)
        p = QPainter(self)
        for x, y in cells:
            edge = any((x + dx, y + dy) not in cells for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            if edge:
                color = OUTLINE
            elif (x + 2, y) not in cells or (x, y + 2) not in cells:
                color = SHADOW  # a soft inner edge along the right and bottom, like the reference
            else:
                color = FILL
            p.fillRect(x * U, y * U, U, U, color)
        p.end()
