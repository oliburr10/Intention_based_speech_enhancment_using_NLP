"""Works out what you're doing on your Mac: frontmost app, window, browser tab, document, screenshot."""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

IS_MAC = sys.platform == "darwin"

_FRONT_APP = """
tell application "System Events"
    set p to first application process whose frontmost is true
    set appName to name of p
    set appPid to unix id of p
    set winTitle to ""
    try
        set winTitle to name of front window of p
    end try
end tell
return appName & linefeed & appPid & linefeed & winTitle
"""

_CHROMIUM_TAB = 'tell application "{app}" to return (URL of active tab of front window) & linefeed & (title of active tab of front window)'
_SAFARI_TAB = 'tell application "Safari" to return (URL of current tab of front window) & linefeed & (name of current tab of front window)'
_WORD_DOC = 'tell application "Microsoft Word" to return full name of active document'

_CHROMIUM = {"Google Chrome", "Arc", "Brave Browser", "Microsoft Edge", "Chromium"}


def _osascript(script: str, timeout: float = 3.0) -> str | None:
    try:
        out = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@dataclass
class Context:
    app: str
    pid: int
    window_title: str = ""
    url: str = ""
    page_title: str = ""
    document: str = ""

    def describe(self) -> str:
        parts = [f"App: {self.app}"]
        if self.window_title:
            parts.append(f"Window: {self.window_title}")
        if self.url:
            parts.append(f"Browser tab: {self.page_title} ({self.url})")
        if self.document:
            parts.append(f"Open document: {self.document}")
        return "\n".join(parts)


def front_context() -> Context | None:
    """Read the frontmost app and whatever we can learn about what's open in it."""
    if not IS_MAC:
        return None
    raw = _osascript(_FRONT_APP)
    if not raw:
        return None
    app, pid, title = (raw.split("\n") + ["", "", ""])[:3]
    ctx = Context(app=app, pid=int(pid) if pid.isdigit() else -1, window_title=title)

    tab = None
    if app in _CHROMIUM:
        tab = _osascript(_CHROMIUM_TAB.format(app=app))
    elif app == "Safari":
        tab = _osascript(_SAFARI_TAB)
    if tab:
        ctx.url, _, ctx.page_title = tab.partition("\n")
    elif app == "Microsoft Word":
        ctx.document = _osascript(_WORD_DOC) or ""
    return ctx


def screenshot(max_px: int = 1600) -> bytes:
    """Capture the main display as PNG, downscaled so it doesn't cost a fortune in tokens."""
    if not IS_MAC:
        raise RuntimeError("Screenshots are only supported on macOS.")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "screen.png"
        subprocess.run(["screencapture", "-x", "-m", "-t", "png", str(path)], check=True, timeout=10)
        if not path.exists():
            raise RuntimeError(
                "Screenshot failed. Give your terminal (or Python) Screen Recording permission in "
                "System Settings > Privacy & Security."
            )
        subprocess.run(["sips", "-Z", str(max_px), str(path)], capture_output=True, timeout=10)
        return path.read_bytes()


def open_in_editor(file_path: str, line: int = 0) -> str:
    """Open a file in VS Code at a line if the `code` command exists, otherwise in its default app."""
    path = os.path.expanduser(file_path)
    if not os.path.exists(path):
        return f"No such file: {path}"
    if shutil.which("code"):
        target = f"{path}:{line}" if line else path
        subprocess.Popen(["code", "-g", target])
        return f"Opened {target} in VS Code."
    subprocess.Popen(["open", path])
    return f"Opened {path}."


class ContextWatcher:
    """Polls the frontmost app in the background and remembers the last one that wasn't the cat.

    When you click the cat, the cat becomes the frontmost app, so "what was I just doing" has to
    be remembered from before the click.
    """

    def __init__(self, interval: float = 2.0):
        self._interval = interval
        self._lock = threading.Lock()
        self._last: Context | None = None
        self._changed_at = time.monotonic()
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def last(self) -> Context | None:
        with self._lock:
            return self._last

    def seconds_since_change(self) -> float:
        with self._lock:
            return time.monotonic() - self._changed_at

    def _run(self) -> None:
        own_pid = os.getpid()
        while not self._stop.wait(self._interval):
            ctx = front_context()
            if ctx is None or ctx.pid == own_pid:
                continue
            with self._lock:
                if self._last != ctx:
                    self._changed_at = time.monotonic()
                self._last = ctx
