"""Long-term memory: a plain Markdown file you can open and edit yourself."""

from datetime import date

from . import config


def load() -> str:
    if config.MEMORY_FILE.exists():
        return config.MEMORY_FILE.read_text().strip()
    return ""


def add(fact: str) -> None:
    config.HOME_DIR.mkdir(parents=True, exist_ok=True)
    with config.MEMORY_FILE.open("a") as f:
        f.write(f"- {fact.strip()} ({date.today().isoformat()})\n")
