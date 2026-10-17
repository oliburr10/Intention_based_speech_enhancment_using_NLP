"""The cat itself: a small always-on-top window you can drag around and click."""

import random
import time

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QMenu, QWidget

from . import config, sprites

BADGE_SPACE = 26   # room above the cat for the "...", "!" and "z" badges
DRAG_THRESHOLD = 4  # pixels the mouse must move before a press counts as a drag


def floating_window_flags() -> Qt.WindowType:
    return (
        Qt.WindowType.FramelessWindowHint
        | Qt.WindowType.WindowStaysOnTopHint
        | Qt.WindowType.Tool
        | Qt.WindowType.NoDropShadowWindowHint
    )


class CatWidget(QWidget):
    clicked = Signal()
    look_requested = Signal()
    new_conversation_requested = Signal()
    watch_toggled = Signal(bool)
    quit_requested = Signal()
    moved = Signal()

    def __init__(self):
        super().__init__(None, floating_window_flags())
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self._pixmaps = {name: sprites.render(name, config.SCALE) for name in sprites.FRAMES}
        sprite_w, sprite_h = sprites.WIDTH * config.SCALE, sprites.HEIGHT * config.SCALE
        self.setFixedSize(sprite_w + BADGE_SPACE, sprite_h + BADGE_SPACE)

        self._frame = "idle"
        self._frame_until = 0.0
        self._busy = False
        self._alert = False
        self._watching = False
        self._last_attention = time.monotonic()
        self._tick_count = 0
        self._press_pos: QPoint | None = None
        self._drag_offset: QPoint | None = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(150)

    # ---- state set by the app -----------------------------------------------------------

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.poke()

    def set_alert(self, alert: bool) -> None:
        self._alert = alert
        self.poke()

    def set_watching(self, watching: bool) -> None:
        self._watching = watching

    def poke(self) -> None:
        """Something happened: wake up."""
        self._last_attention = time.monotonic()
        self.update()

    @property
    def sleeping(self) -> bool:
        idle = time.monotonic() - self._last_attention
        return not self._busy and not self._alert and idle > config.SLEEP_AFTER_SEC

    # ---- animation ----------------------------------------------------------------------

    def _tick(self) -> None:
        self._tick_count += 1
        now = time.monotonic()
        if self._busy:
            self._frame = "look_up"
        elif self.sleeping:
            self._frame = "sleep"
        elif now >= self._frame_until:
            roll = random.random()
            if self._frame != "idle":
                self._frame, self._frame_until = "idle", now + random.uniform(2.0, 5.0)
            elif roll < 0.5:
                self._frame, self._frame_until = "blink", now + 0.15
            else:
                self._frame, self._frame_until = "tail", now + 0.6
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.drawPixmap(0, BADGE_SPACE, self._pixmaps[self._frame])
        if self._alert:
            self._badge(p, "!", QColor("#e5484d"))
        elif self._busy:
            self._badge(p, "." * (1 + self._tick_count // 3 % 3), QColor("#ffffff"))
        elif self.sleeping:
            self._badge(p, "z" if self._tick_count // 6 % 2 else "zZ", QColor("#ffffff"))
        p.end()

    def _badge(self, p: QPainter, text: str, fill: QColor) -> None:
        x = self.width() - BADGE_SPACE - 14
        rect_w, rect_h = 34, 20
        path = QPainterPath()
        path.addRoundedRect(x, 2, rect_w, rect_h, 6, 6)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillPath(path, fill)
        p.setPen(QPen(QColor(sprites.PALETTE["K"]), 2))
        p.drawPath(path)
        font = QFont()
        font.setBold(True)
        font.setPixelSize(14)
        p.setFont(font)
        p.setPen(QColor("#ffffff") if fill.name() == "#e5484d" else QColor(sprites.PALETTE["K"]))
        p.drawText(x, 2, rect_w, rect_h, Qt.AlignmentFlag.AlignCenter, text)

    # ---- dragging vs clicking -----------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.globalPosition().toPoint()
            self._drag_offset = None

    def mouseMoveEvent(self, event):
        if self._press_pos is None:
            return
        pos = event.globalPosition().toPoint()
        if self._drag_offset is None:
            if (pos - self._press_pos).manhattanLength() < DRAG_THRESHOLD:
                return
            self._drag_offset = self._press_pos - self.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self.move(pos - self._drag_offset)
        self.moved.emit()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self._press_pos is None:
            return
        was_drag = self._drag_offset is not None
        self._press_pos = self._drag_offset = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.poke()
        if not was_drag:
            self.clicked.emit()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.addAction("Chat", self.clicked.emit)
        menu.addAction("Look at my screen", self.look_requested.emit)
        watch = QAction("Watch mode (pop up tips)", menu, checkable=True, checked=self._watching)
        watch.toggled.connect(self.watch_toggled.emit)
        menu.addAction(watch)
        menu.addAction("New conversation", self.new_conversation_requested.emit)
        menu.addSeparator()
        menu.addAction(f"Quit {config.CAT_NAME}", self.quit_requested.emit)
        menu.exec(event.globalPos())
