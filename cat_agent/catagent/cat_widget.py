"""The cat itself: a small always-on-top window you can drag around and click."""

import math
import random
import time
from typing import Callable

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QMenu, QWidget

from . import config, sprites

CODE_BADGES = ["</>", "{ }", "( )", "</>", ";"]
BADGE_SPACE = 26   # room above the cat for the "...", "!" and "</>" badges
BREATH_SEC = 2.6   # one slow breath while asleep
Z_RISE_SEC = 3.0   # how long each floating z takes to drift up and fade away
Z_COUNT = 3
Z_UNIT = 2         # screen pixels per pixel of a floating z
DRAG_THRESHOLD = 4  # pixels the mouse must move before a press counts as a drag
# Played when you click the sleeping cat: (frame, seconds).
WAKE_UP = [("wake", 0.35), ("stretch", 0.9), ("yawn", 0.55), ("idle", 0.15)]


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
    auto_edit_toggled = Signal(bool)
    quit_requested = Signal()
    stop_requested = Signal()
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
        self._coding = False
        self._pause_until = 0.0
        self._alert = False
        self._watching = False
        self._auto_edit = False
        self._last_attention = time.monotonic()
        self._tick_count = 0
        self._press_pos: QPoint | None = None
        self._drag_offset: QPoint | None = None
        self._sequence: list[tuple[str, float]] = []
        self._sequence_until = 0.0
        self._after_sequence: Callable[[], None] | None = None
        # The app sets this so the cat stays awake while its input bar or speech cloud is open.
        self.keep_awake: Callable[[], bool] = lambda: False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(150)

    # ---- state set by the app -----------------------------------------------------------

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        if not busy:
            self._coding = False
        self.poke()

    def set_coding(self, coding: bool) -> None:
        """While busy on code, the cat puts its glasses on and types on its laptop."""
        self._coding = coding
        self.update()

    def set_alert(self, alert: bool) -> None:
        self._alert = alert
        self.poke()

    def set_watching(self, watching: bool) -> None:
        self._watching = watching

    def set_auto_edit(self, on: bool) -> None:
        self._auto_edit = on

    def poke(self) -> None:
        """Something happened: wake up."""
        self._last_attention = time.monotonic()
        self.update()

    @property
    def sleeping(self) -> bool:
        idle = time.monotonic() - self._last_attention
        waking = bool(self._sequence)
        return not (self._busy or self._alert or waking) and idle > config.SLEEP_AFTER_SEC

    def wake_up(self, then: Callable[[], None] | None = None) -> None:
        """Open eyes, stretch, yawn, then call `then`."""
        self._sequence = list(WAKE_UP)
        self._sequence_until = 0.0
        self._after_sequence = then
        self.poke()

    # ---- animation ----------------------------------------------------------------------

    def _tick(self) -> None:
        self._tick_count += 1
        now = time.monotonic()
        if self.keep_awake():
            self._last_attention = now
        if self._sequence and not self._busy:
            if now >= self._sequence_until:
                self._frame, duration = self._sequence.pop(0)
                self._sequence_until = now + duration
                if not self._sequence and self._after_sequence:
                    QTimer.singleShot(int(duration * 1000), self._after_sequence)
                    self._after_sequence = None
            self._last_attention = now
        elif self._busy and self._coding:
            if now < self._pause_until:
                self._frame = "code_glow"  # stops typing to read the screen
            else:
                self._frame = "code_a" if self._tick_count % 2 else "code_b"
                if random.random() < 0.05:
                    self._pause_until = now + random.uniform(0.6, 1.5)
        elif self._busy:
            self._frame = "look_up"
        elif self.sleeping:
            self._frame = "sleep_b" if now % BREATH_SEC < BREATH_SEC * 0.4 else "sleep_a"
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
        elif self._busy and self._coding:
            self._badge(p, CODE_BADGES[self._tick_count // 5 % len(CODE_BADGES)], QColor("#ffffff"))
        elif self._busy:
            self._badge(p, "." * (1 + self._tick_count // 3 % 3), QColor("#ffffff"))
        elif self.sleeping:
            self._draw_zs(p)
        p.end()

    def _draw_zs(self, p: QPainter) -> None:
        """Pixel z's that float up from the sleeping cat's head, growing and fading as they go."""
        now = time.monotonic()
        start_x = 7 * config.SCALE
        start_y = BADGE_SPACE + 8 * config.SCALE
        for i in range(Z_COUNT):
            t = (now / Z_RISE_SEC + i / Z_COUNT) % 1.0
            size = 4 + round(t * 2)  # 4x4 pixels growing to 6x6
            x = start_x + t * 44 + math.sin(t * 2 * math.pi) * 3
            y = start_y - t * 48 - size * Z_UNIT
            fade_in, fade_out = min(1.0, t / 0.15), min(1.0, (1.0 - t) / 0.35)
            p.setOpacity(min(fade_in, fade_out))
            self._pixel_z(p, round(x / Z_UNIT) * Z_UNIT, round(y / Z_UNIT) * Z_UNIT, size)
        p.setOpacity(1.0)

    def _pixel_z(self, p: QPainter, x: int, y: int, size: int) -> None:
        cells = [(c, 0) for c in range(size)] + [(c, size - 1) for c in range(size)]
        cells += [(size - 1 - r, r) for r in range(1, size - 1)]
        u = Z_UNIT
        for color, grow in ((QColor(sprites.PALETTE["K"]), 1), (QColor("#ffffff"), 0)):
            for cx, cy in cells:
                p.fillRect(x + cx * u - grow, y + cy * u - grow, u + 2 * grow, u + 2 * grow, color)

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
        if was_drag:
            self.poke()
        elif self.sleeping:
            self.wake_up(then=self.clicked.emit)
        elif not self._sequence:
            self.poke()
            self.clicked.emit()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.addAction(f"Talk to {config.CAT_NAME}", self.clicked.emit)
        stop = menu.addAction("Stop what you're doing", self.stop_requested.emit)
        stop.setEnabled(self._busy)
        menu.addAction("Look at my screen", self.look_requested.emit)
        watch = QAction("Watch mode (pop up tips)", menu, checkable=True, checked=self._watching)
        watch.toggled.connect(self.watch_toggled.emit)
        menu.addAction(watch)
        auto_edit = QAction("Edit files without asking", menu, checkable=True, checked=self._auto_edit)
        auto_edit.toggled.connect(self.auto_edit_toggled.emit)
        menu.addAction(auto_edit)
        menu.addAction("New conversation", self.new_conversation_requested.emit)
        menu.addSeparator()
        menu.addAction(f"Quit {config.CAT_NAME}", self.quit_requested.emit)
        menu.exec(event.globalPos())
