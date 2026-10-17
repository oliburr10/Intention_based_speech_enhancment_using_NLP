"""The chat window that opens when you click the cat."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QTextCursor, QTextFormat
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from . import config
from .cat_widget import floating_window_flags

STYLE = """
#chat { background: #fffaf3; border: 3px solid #3b2314; border-radius: 12px; }
#header { color: #3b2314; font-weight: bold; font-size: 15px; }
QTextBrowser { background: transparent; border: none; color: #2b2b2b; font-size: 13px; }
QPlainTextEdit { background: #ffffff; border: 2px solid #e6d9c8; border-radius: 8px;
                 padding: 4px; font-size: 13px; color: #2b2b2b; }
QPlainTextEdit:focus { border-color: #ec8b33; }
QPushButton { background: #ec8b33; color: white; border: none; border-radius: 8px;
              padding: 6px 10px; font-weight: bold; }
QPushButton:hover { background: #d97a24; }
QPushButton[flat="true"] { background: transparent; color: #3b2314; padding: 4px 6px; }
QPushButton[flat="true"]:hover { background: #f3e6d6; }
#permission { background: #fff1dc; border: 2px solid #ec8b33; border-radius: 8px; }
#permission QLabel { color: #3b2314; font-size: 12px; }
QPushButton#deny { background: #8a7a6a; }
"""


class _Input(QPlainTextEdit):
    submitted = Signal()

    def keyPressEvent(self, event: QKeyEvent):
        is_enter = event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        if is_enter and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.submitted.emit()
            return
        super().keyPressEvent(event)


class ChatWindow(QWidget):
    send_requested = Signal(str)
    stop_requested = Signal()
    look_requested = Signal()
    new_conversation_requested = Signal()
    permission_answered = Signal(str, bool)

    def __init__(self):
        super().__init__(None, floating_window_flags())
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.resize(400, 500)
        self.setStyleSheet(STYLE)

        self._entries: list[tuple[str, str]] = []
        self._busy = False
        self._permission_queue: list[tuple[str, str, str]] = []

        frame = QFrame(self)
        frame.setObjectName("chat")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 12)

        header = QHBoxLayout()
        title = QLabel(f"🐱 {config.CAT_NAME}")
        title.setObjectName("header")
        header.addWidget(title)
        header.addStretch()
        for text, tip, slot in (
            ("👀", "Look at my screen", self.look_requested.emit),
            ("↺", "New conversation", self.new_conversation_requested.emit),
            ("✕", "Close", self.hide),
        ):
            button = QPushButton(text)
            button.setProperty("flat", True)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            header.addWidget(button)
        layout.addLayout(header)

        self._log = QTextBrowser()
        self._log.setOpenExternalLinks(True)
        layout.addWidget(self._log, 1)

        self._permission = QFrame()
        self._permission.setObjectName("permission")
        perm_layout = QVBoxLayout(self._permission)
        self._permission_label = QLabel()
        self._permission_label.setWordWrap(True)
        self._permission_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        perm_layout.addWidget(self._permission_label)
        perm_buttons = QHBoxLayout()
        perm_buttons.addStretch()
        deny = QPushButton("Don't")
        deny.setObjectName("deny")
        deny.clicked.connect(lambda: self._answer_permission(False))
        allow = QPushButton("Allow")
        allow.clicked.connect(lambda: self._answer_permission(True))
        perm_buttons.addWidget(deny)
        perm_buttons.addWidget(allow)
        perm_layout.addLayout(perm_buttons)
        self._permission.hide()
        layout.addWidget(self._permission)

        row = QHBoxLayout()
        self._input = _Input()
        self._input.setPlaceholderText(f"Ask {config.CAT_NAME} anything…")
        self._input.setFixedHeight(60)
        self._input.submitted.connect(self._submit)
        row.addWidget(self._input, 1)
        self._send = QPushButton("Send")
        self._send.clicked.connect(self._send_or_stop)
        row.addWidget(self._send)
        layout.addLayout(row)

        self.add("assistant", f"Mrrp! Hi {config.USER_NAME}, what are we working on? 🐾")

    # ---- transcript -----------------------------------------------------------------------

    def add(self, kind: str, text: str) -> None:
        self._entries.append((kind, text))
        self._render()

    def _render(self) -> None:
        parts = []
        for kind, text in self._entries:
            if kind == "user":
                parts.append(f"**You:** {text}")
            elif kind == "assistant":
                parts.append(f"**{config.CAT_NAME}:** {text}")
            elif kind == "tool":
                parts.append(f"*🔧 {text}*")
            elif kind == "error":
                parts.append(f"*⚠️ {text}*")
            else:
                parts.append(f"*{text}*")
        if self._busy:
            parts.append(f"*{config.CAT_NAME} is thinking…*")
        self._log.setMarkdown("\n\n".join(parts))
        self._space_blocks()
        bar = self._log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _space_blocks(self) -> None:
        """Qt's Markdown import packs paragraphs tightly; add breathing room and shade code."""
        doc = self._log.document()
        cursor = QTextCursor(doc)
        block = doc.begin()
        while block.isValid():
            fmt = block.blockFormat()
            is_code = fmt.hasProperty(QTextFormat.Property.BlockCodeFence)
            next_is_code = block.next().blockFormat().hasProperty(QTextFormat.Property.BlockCodeFence)
            if is_code:
                fmt.setBackground(QColor("#f3e6d6"))
                fmt.setLeftMargin(4)
            if not (is_code and next_is_code):
                fmt.setBottomMargin(6)
            cursor.setPosition(block.position())
            cursor.setBlockFormat(fmt)
            block = block.next()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._send.setText("Stop" if busy else "Send")
        self._render()

    # ---- input ----------------------------------------------------------------------------

    def _submit(self) -> None:
        text = self._input.toPlainText().strip()
        if not text or self._busy:
            return
        self._input.clear()
        self.add("user", text)
        self.send_requested.emit(text)

    def _send_or_stop(self) -> None:
        if self._busy:
            self.stop_requested.emit()
        else:
            self._submit()

    def focus_input(self) -> None:
        self.raise_()
        self.activateWindow()
        self._input.setFocus()

    # ---- permission prompts -------------------------------------------------------------

    def ask_permission(self, request_id: str, tool_name: str, details: str) -> None:
        self._permission_queue.append((request_id, tool_name, details))
        if len(self._permission_queue) == 1:
            self._show_permission()
        self.show()
        self.focus_input()

    def _show_permission(self) -> None:
        _, tool_name, details = self._permission_queue[0]
        self._permission_label.setText(
            f"<b>{config.CAT_NAME} wants to use {tool_name}:</b><pre>{_escape(details)}</pre>"
        )
        self._permission.show()

    def _answer_permission(self, allow: bool) -> None:
        if not self._permission_queue:
            return
        request_id, _, _ = self._permission_queue.pop(0)
        self.permission_answered.emit(request_id, allow)
        if self._permission_queue:
            self._show_permission()
        else:
            self._permission.hide()


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
