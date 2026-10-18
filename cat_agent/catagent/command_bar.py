"""The little input bar that pops up under the cat when you click it."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import config
from .cat_widget import floating_window_flags

STYLE = """
#bar { background: #ffffff; border: 1px solid #e3e1dc; border-radius: 22px; }
QLineEdit { border: none; background: transparent; font-size: 15px; color: #1f1f1f;
            selection-background-color: #f5c99b; }
QPushButton { border: none; background: transparent; color: #4a4a4a; font-size: 20px;
              border-radius: 15px; min-width: 30px; min-height: 30px; }
QPushButton:hover { background: #f1efeb; }
QPushButton:disabled { color: #b9b6b0; }
"""


class CommandBar(QWidget):
    submitted = Signal(str)

    def __init__(self):
        super().__init__(None, floating_window_flags())
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setStyleSheet(STYLE)
        self.setFixedSize(420, 76)  # the bar plus room for its soft shadow
        self._attached: list[str] = []

        bar = QFrame(self)
        bar.setObjectName("bar")
        shadow = QGraphicsDropShadowEffect(bar)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 3)
        shadow.setColor(QColor(0, 0, 0, 45))
        bar.setGraphicsEffect(shadow)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 14)
        outer.addWidget(bar)

        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 4, 10, 4)
        attach = QPushButton("+")
        attach.setToolTip("Attach a file for the cat to look at")
        attach.setCursor(Qt.CursorShape.PointingHandCursor)
        attach.clicked.connect(self._attach)
        row.addWidget(attach)

        self._input = QLineEdit()
        self._input.setPlaceholderText(f"Tell {config.CAT_NAME} what to do…")
        self._input.returnPressed.connect(self._submit)
        row.addWidget(self._input, 1)

        mic = QPushButton("🎙")
        mic.setToolTip("Voice input is coming soon")
        mic.setEnabled(False)
        row.addWidget(mic)

    def open(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()
        self._input.setFocus()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
        else:
            super().keyPressEvent(event)

    def _attach(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Attach a file", str(config.WORKSPACE))
        if path:
            self._attached.append(path)
            names = ", ".join(Path(p).name for p in self._attached)
            self._input.setPlaceholderText(f"📎 {names} — what should I do with it?")
        self.open()

    def _submit(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        if self._attached:
            text += "\n\nAttached files:\n" + "\n".join(self._attached)
        self._attached = []
        self._input.clear()
        self._input.setPlaceholderText(f"Tell {config.CAT_NAME} what to do…")
        self.hide()
        self.submitted.emit(text)
