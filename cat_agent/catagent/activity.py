"""Guesses whether the cat is working on code, so it can put its glasses on and get typing."""

import re

CODE_APPS = {
    "Code", "Visual Studio Code", "Cursor", "Xcode", "PyCharm", "IntelliJ IDEA", "WebStorm",
    "Sublime Text", "Zed", "Nova", "Terminal", "iTerm2", "Warp", "Ghostty",
}

_CODE_WORDS = re.compile(
    r"\b(code|coding|program|bug|bugs|debug|error|exception|traceback|stack ?trace|crash|"
    r"function|method|class|variable|script|compile|compiler|build|tests?|pytest|unittest|"
    r"refactor|repo|git|commit|branch|merge|pull request|python|javascript|typescript|java|"
    r"c\+\+|rust|golang|swift|kotlin|html|css|sql|api|json|yaml|regex|terminal|command line|"
    r"install|package|npm|pip|import|syntax|deploy|docker|notebook|numpy|pandas|pytorch|"
    r"algorithm|loop|array|dict|null|undefined|segfault)\b",
    re.IGNORECASE,
)

_CODE_FILE = re.compile(
    r"\.(py|ipynb|js|jsx|ts|tsx|java|c|cc|cpp|h|hpp|rs|go|rb|php|swift|kt|sh|zsh|sql|html|css|"
    r"scss|json|ya?ml|toml|tex|m|r|jl|lua|dart|vue|svelte)\b",
    re.IGNORECASE,
)


def looks_like_coding(text: str, app: str = "") -> bool:
    """A request counts as coding if you were in a code editor or terminal, or it reads like code talk."""
    return app in CODE_APPS or bool(_CODE_WORDS.search(text) or _CODE_FILE.search(text))


def is_coding_tool(name: str, args: dict) -> bool:
    """Editing files or running commands is coding; so is reading or searching code files."""
    if name in ("Edit", "Write", "NotebookEdit", "Bash"):
        return True
    if name in ("Read", "Grep", "Glob"):
        target = " ".join(str(args.get(k, "")) for k in ("file_path", "path", "pattern", "glob"))
        return bool(_CODE_FILE.search(target))
    return False
