# Ginger: a pixel cat that is your personal agent

![Ginger's animation frames](docs/cat.png)

Ginger is a small ginger-and-white pixel cat that lives on your Mac desktop. You can drag
the cat anywhere. Click it and a chat window opens where you can ask for help with whatever
you're doing. The agent behind it is Claude, running through the
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk), which is the same engine as
Claude Code. That means it can read and edit code, run commands, search your files and the
web, and it also gets a few cat-specific tools:

| Tool | What it does |
|---|---|
| `what_am_i_working_on` | Reports the app, window, browser tab or Word document you were last using |
| `look_at_screen` | Takes a screenshot so the cat can see what you see |
| `show_tip` | Pops up a speech bubble next to the cat saying what's wrong and where, with a **Show me** button that opens the file at the right line |
| `open_in_editor` | Opens a file in VS Code at a given line |
| `remember` | Saves lasting facts about you to `~/.ginger/memory.md` |

## Setup (macOS)

You need Python 3.10 or newer.

```bash
cd cat_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...   # from https://platform.claude.com
python run.py
```

The Agent SDK ships with its own copy of the Claude Code engine, so nothing else needs
installing. To avoid exporting the key every time, add the `export` line to `~/.zshrc`.

### macOS permissions

The first time the cat checks what you're doing, macOS asks for permission. If you
missed the prompt, enable these in **System Settings → Privacy & Security** for the app you
start the cat from (Terminal, iTerm or VS Code):

- **Accessibility**: lets the cat read window titles.
- **Automation**: lets it ask System Events, your browser and Word what's open.
- **Screen Recording**: lets it take screenshots.

Restart the cat after you change these.

## Using it

- **Drag** the cat anywhere. It remembers where you left it.
- **Click** the cat to open or close the chat. **Enter** sends a message and **Shift+Enter**
  starts a new line.
- **Right-click** the cat for these options: *Look at my screen*, *Watch mode*,
  *New conversation*, *Quit*.
- Each message automatically includes the app or file you were in just before clicking the
  cat, so "why is this broken?" just works.
- The cat reads, searches and browses without asking. Before it **edits a file or runs a
  command**, it shows you exactly what it wants to do, with **Allow** and **Don't** buttons.
- The cat's moods: `...` means it's thinking, a red `!` means it has a tip for you, and `zZ`
  means it's asleep (click to wake it).

### Watch mode

With watch mode on, the cat glances at your screen whenever you switch to something new
and stay on it for a little while. It checks at most once every
`WATCH_INTERVAL_SEC` (2 minutes by default). If it spots a real problem, such as an error
message, a failing test, a LaTeX error or an obvious typo, it pops up a tip. If not, it stays
quiet. Watch-mode checks can't edit anything. Each check is a paid model call that includes a
screenshot, so watch mode is **off by default**.

### What it can do in each app

| Where you're working | What the cat can do |
|---|---|
| VS Code or any local code | Read, explain and edit the files directly (with your approval), run tests, open the exact line |
| Word (`.docx` on disk) | Find the document's path and read it. To edit, close the file in Word first, then the cat can edit it with `python-docx` |
| Overleaf | It can see the page and tell you exactly what to change. If you sync the project locally (Overleaf's Git integration), it can edit the `.tex` files directly |
| Google Docs | It can see the page and tell you exactly what to change and where. It can't edit Google Docs directly yet |

## Making it yours

All the settings are in [`catagent/config.py`](catagent/config.py):

- `CAT_NAME`, `PERSONALITY`: who the cat is and how it talks
- `MODEL`: which Claude model it uses
- `WORKSPACE`: the folder it starts in (point it at your code folder)
- `SCALE`: how big the cat is
- `WATCH_INTERVAL_SEC`, `SLEEP_AFTER_SEC`

The cat's look is defined in [`catagent/sprites.py`](catagent/sprites.py) as ASCII grids,
one letter per pixel. Edit the letters to redraw it, or add frames.

Its memory is a plain file, `~/.ginger/memory.md`, that you can open and edit yourself.

## How it fits together

```
run.py
catagent/
  app.py          wires everything together; window placement, watch mode, saved state
  cat_widget.py   the draggable, animated cat window
  chat_window.py  the chat UI and Allow/Don't permission prompts
  tip_bubble.py   the speech bubble for tips
  agent.py        the Claude Agent SDK session, on a background asyncio thread
  tools.py        the cat's own tools (screen, context, tips, memory)
  context.py      macOS helpers: frontmost app, browser tab, Word doc, screenshots
  memory.py       long-term memory file
  sprites.py      the pixel art
  config.py       settings and personality
```

Qt runs the UI on the main thread. The agent runs on its own asyncio loop in a background
thread, and the two communicate through Qt signals.

## Ideas for what's next

- A global hotkey to summon the cat
- Calendar, Reminders and Notes tools (via AppleScript or Shortcuts)
- Google Docs editing through the Google Docs API
- Voice: a wake word plus speech-to-text (the speech enhancement work in this repo could
  clean up the audio first)
- Launch at login (a LaunchAgent or a packaged `.app` via `py2app`)
