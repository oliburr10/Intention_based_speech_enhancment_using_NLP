"""A speech bubble that pops up next to the cat with a tip."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import context
from .chat_window import STYLE
from .cat_widget import floating_window_flags


class TipBubble(QWidget):
    chat_requested = Signal()
    dismissed = Signal()

    def __init__(self):
        super().__init__(None, floating_window_flags())
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setStyleSheet(STYLE)
        self.setFixedWidth(300)
        self._file = ""
        self._line = 0

        frame = QFrame(self)
        frame.setObjectName("chat")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)

        self._title = QLabel()
        self._title.setObjectName("header")
        self._title.setWordWrap(True)
        self._body = QLabel()
        self._body.setWordWrap(True)
        self._body.setStyleSheet("color: #2b2b2b; font-size: 13px;")
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._title)
        layout.addWidget(self._body)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self._show_me = QPushButton("Show me")
        self._show_me.clicked.connect(self._open_file)
        chat = QPushButton("Chat")
        chat.clicked.connect(self._chat)
        close = QPushButton("✕")
        close.setProperty("flat", True)
        close.clicked.connect(self._dismiss)
        buttons.addWidget(close)
        buttons.addWidget(chat)
        buttons.addWidget(self._show_me)
        layout.addLayout(buttons)

    def show_tip(self, title: str, message: str, file_path: str, line: int) -> None:
        self._title.setText(title)
        self._body.setText(message)
        self._file, self._line = file_path, line
        self._show_me.setVisible(bool(file_path))
        self.layout().activate()
        self.setFixedHeight(self.layout().heightForWidth(self.width()))
        self.show()
        self.raise_()

    def _open_file(self) -> None:
        context.open_in_editor(self._file, self._line)
        self._dismiss()

    def _chat(self) -> None:
        self._dismiss()
        self.chat_requested.emit()

    def _dismiss(self) -> None:
        self.hide()
        self.dismissed.emit()
