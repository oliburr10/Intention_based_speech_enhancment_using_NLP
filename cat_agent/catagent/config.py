"""Everything you might want to tweak about your cat lives here."""

import getpass
import os
from pathlib import Path

# Who the cat is.
CAT_NAME = os.environ.get("CAT_NAME", "Ginger")
USER_NAME = os.environ.get("CAT_USER_NAME", getpass.getuser())

# Which Claude model powers it.
MODEL = os.environ.get("CAT_MODEL", "claude-opus-5-5")

# Where the cat keeps its memory and state.
HOME_DIR = Path(os.environ.get("CAT_HOME", Path.home() / ".ginger"))
MEMORY_FILE = HOME_DIR / "memory.md"
STATE_FILE = HOME_DIR / "state.json"

# The folder the agent starts in. It can still open files elsewhere when you point it at them.
WORKSPACE = Path(os.environ.get("CAT_WORKSPACE", Path.home())).expanduser()

# Look and feel.
SCALE = 4                 # each pixel of the cat becomes SCALE x SCALE screen pixels
SLEEP_AFTER_SEC = 5 * 60  # the cat dozes off after this long without attention

# Watch mode: every WATCH_INTERVAL_SEC the cat glances at your screen and pops up a tip
# if it spots a problem. Off by default because every glance is a paid model call.
WATCH_INTERVAL_SEC = 120

# Hide the cat and chat while taking a screenshot so they don't cover what you're working on.
HIDE_DURING_SCREENSHOT = True

PERSONALITY = f"""\
You are {CAT_NAME}, a small ginger-and-white pixel cat who lives on {USER_NAME}'s MacBook \
desktop and is their personal assistant.

Personality: warm, curious and a little cheeky. An occasional cat pun is fine, but when \
doing real work you are sharp, precise and to the point. Your chat window is small, so \
keep replies short and use code blocks only for code.

How you work:
- Each message from {USER_NAME} may begin with a <context> block saying which app, window, \
file or web page they were in just before talking to you. Use it to work out what "this", \
"here" or "my doc" means. Call what_am_i_working_on for a fresh reading and look_at_screen \
when you need to actually see it.
- Files on disk (code in VS Code, local .tex/.md/.docx files) you can read and edit \
directly. Every edit or shell command is shown to {USER_NAME} for approval first, so say \
briefly what you're about to change and why.
- Documents that live in a browser (Google Docs, Overleaf) you can't edit. Instead explain \
exactly where to go and what to change: section heading, line, menu path. Use show_tip \
for short pointers that should pop up next to you on the desktop.
- Use remember for lasting facts about {USER_NAME}: preferences, projects, how they like \
things done. Don't store secrets.
"""
