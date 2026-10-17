"""Wires the cat, the chat window, the tip bubble and the agent together."""

import json
import sys

from PySide6.QtCore import QPoint, QRect, QTimer
from PySide6.QtWidgets import QApplication

from . import config, context
from .agent import AgentBridge
from .cat_widget import CatWidget
from .chat_window import ChatWindow
from .tip_bubble import TipBubble

GAP = 8


def _hide_dock_icon() -> None:
    """Make the cat a background app with no Dock icon (needs pyobjc, optional)."""
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
    except ImportError:
        return
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)


class CatApp:
    def __init__(self, app: QApplication):
        self.app = app
        self.watcher = context.ContextWatcher()
        self.agent = AgentBridge(self.watcher)
        self.cat = CatWidget()
        self.chat = ChatWindow()
        self.bubble = TipBubble()
        self._last_tip = ""
        self._hidden_for_capture: list = []
        self._watched_context = None

        self.cat.clicked.connect(self.toggle_chat)
        self.cat.moved.connect(self.follow_cat)
        self.cat.look_requested.connect(self.look_at_screen)
        self.cat.new_conversation_requested.connect(self.agent.new_conversation)
        self.cat.watch_toggled.connect(self.set_watch_mode)
        self.cat.quit_requested.connect(self.quit)

        self.chat.send_requested.connect(self.send)
        self.chat.stop_requested.connect(self.agent.stop_turn)
        self.chat.look_requested.connect(self.look_at_screen)
        self.chat.new_conversation_requested.connect(self.agent.new_conversation)
        self.chat.permission_answered.connect(self.agent.resolve_permission)

        self.bubble.chat_requested.connect(self.open_chat)
        self.bubble.dismissed.connect(lambda: self.cat.set_alert(False))

        self.agent.message.connect(self.on_agent_message)
        self.agent.busy_changed.connect(self.on_busy)
        self.agent.permission_requested.connect(self.on_permission)
        self.agent.tip.connect(self.on_tip)
        self.agent.hide_requested.connect(self.set_hidden_for_capture)

        self.watch_timer = QTimer()
        self.watch_timer.timeout.connect(self.watch_tick)
        self.save_timer = QTimer()
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self._save_state)

    def start(self) -> None:
        self.watcher.start()
        self.agent.start()
        state = self._load_state()
        screen = self.app.primaryScreen().availableGeometry()
        default = QPoint(screen.right() - self.cat.width() - 40, screen.bottom() - self.cat.height() - 40)
        pos = QPoint(*state["pos"]) if "pos" in state else default
        self.cat.move(self._clamp(QRect(pos, self.cat.size())))
        self.cat.show()
        if state.get("watch"):
            self.set_watch_mode(True)

    # ---- chat ------------------------------------------------------------------------------

    def toggle_chat(self) -> None:
        if self.chat.isVisible():
            self.chat.hide()
        else:
            self.open_chat()

    def open_chat(self) -> None:
        self.cat.set_alert(False)
        self.follow_cat()
        self.chat.show()
        self.chat.focus_input()

    def send(self, text: str) -> None:
        notes = []
        ctx = self.watcher.last()
        if ctx is not None:
            notes.append(ctx.describe())
        if self._last_tip:
            notes.append(f"Tip you popped up earlier: {self._last_tip}")
            self._last_tip = ""
        if notes:
            text = "<context>\n" + "\n".join(notes) + "\n</context>\n\n" + text
        self.agent.send(text)

    def look_at_screen(self) -> None:
        self.open_chat()
        prompt = "Take a look at my screen. What am I working on, and is there anything I should fix?"
        self.chat.add("user", "👀 Look at my screen")
        self.send(prompt)

    def on_agent_message(self, kind: str, text: str) -> None:
        self.chat.add(kind, text)
        self.cat.poke()

    def on_busy(self, busy: bool) -> None:
        self.cat.set_busy(busy)
        self.chat.set_busy(busy)

    def on_permission(self, request_id: str, tool_name: str, details: str) -> None:
        self.follow_cat()
        self.chat.ask_permission(request_id, tool_name, details)

    def on_tip(self, title: str, message: str, file_path: str, line: int) -> None:
        where = f" ({file_path}:{line})" if file_path else ""
        self._last_tip = f"{title}: {message}{where}"
        self.cat.set_alert(True)
        self.bubble.show_tip(title, message, file_path, line)
        self.follow_cat()

    def set_hidden_for_capture(self, hidden: bool) -> None:
        if hidden:
            self._hidden_for_capture = [w for w in (self.cat, self.chat, self.bubble) if w.isVisible()]
            for w in self._hidden_for_capture:
                w.hide()
        else:
            for w in self._hidden_for_capture:
                w.show()
            self._hidden_for_capture = []

    # ---- watch mode ----------------------------------------------------------------------

    def set_watch_mode(self, on: bool) -> None:
        if on and not context.IS_MAC:
            self.chat.add("error", "Watch mode needs macOS.")
            on = False
        self.cat.set_watching(on)
        if on:
            self.watch_timer.start(config.WATCH_INTERVAL_SEC * 1000)
        else:
            self.watch_timer.stop()
        self._save_state(watch=on)

    def watch_tick(self) -> None:
        """Glance at the screen, but only when the user has moved on to something new and settled."""
        ctx = self.watcher.last()
        if ctx is None or ctx == self._watched_context or self.chat.isVisible():
            return
        if self.watcher.seconds_since_change() < 20:
            return
        self._watched_context = ctx
        self.agent.watch_check(ctx.describe())

    # ---- layout ------------------------------------------------------------------------------

    def follow_cat(self) -> None:
        """Place the chat window and bubble next to the cat, on whichever side has room."""
        cat = self.cat.frameGeometry()
        screen = self.app.screenAt(cat.center()) or self.app.primaryScreen()
        area = screen.availableGeometry()
        for widget in (self.chat, self.bubble):
            size = widget.size() if widget is self.chat else widget.sizeHint()
            x = cat.left() - size.width() - GAP
            if x < area.left():
                x = cat.right() + GAP
            y = cat.bottom() - size.height()
            widget.move(self._clamp(QRect(QPoint(x, y), size), area))
        self.save_timer.start(500)

    def _clamp(self, rect: QRect, area: QRect | None = None) -> QPoint:
        area = area or self.app.primaryScreen().availableGeometry()
        x = min(max(rect.left(), area.left()), area.right() - rect.width())
        y = min(max(rect.top(), area.top()), area.bottom() - rect.height())
        return QPoint(x, y)

    # ---- state ----------------------------------------------------------------------------

    def _load_state(self) -> dict:
        try:
            return json.loads(config.STATE_FILE.read_text())
        except (OSError, ValueError):
            return {}

    def _save_state(self, **changes) -> None:
        state = self._load_state()
        state["pos"] = [self.cat.x(), self.cat.y()]
        state.update(changes)
        config.HOME_DIR.mkdir(parents=True, exist_ok=True)
        config.STATE_FILE.write_text(json.dumps(state))

    def quit(self) -> None:
        self._save_state()
        self.watcher.stop()
        self.agent.shutdown()
        self.app.quit()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(config.CAT_NAME)
    app.setQuitOnLastWindowClosed(False)
    _hide_dock_icon()
    cat_app = CatApp(app)
    cat_app.start()
    return app.exec()
