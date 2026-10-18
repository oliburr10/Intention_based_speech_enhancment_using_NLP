# Ginger: a pixel cat that is your personal agent

![Ginger waking up, taking an instruction and saying what it did](docs/cat_flow.gif)

Ginger is a small ginger-and-white pixel cat that lives on your Mac desktop. You can drag
the cat anywhere. Click it and an input bar pops up underneath it. Tell it what to do, and
when it's finished it tells you what it did in a pixel speech cloud, such as **Code
changed ✓** or **Text changed ✓**. The agent behind it is Claude, running through the
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk), which is the same engine as
Claude Code. That means it can read and edit code, run commands, search your files and the
web, and it also gets a few cat-specific tools:

| Tool | What it does |
|---|---|
| `what_am_i_working_on` | Reports the app, window, browser tab or Word document you were last using |
| `look_at_screen` | Takes a screenshot so the cat can see what you see |
| `show_tip` | Points at a problem in a local file, which gives the cat's speech cloud a **Show me** button that opens the file at the right line |
| `open_in_editor` | Opens a file in VS Code at a given line |
| `copy_to_clipboard` | Puts code or text on your clipboard for you to paste |
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
- **Click** the cat and an input bar pops up underneath it. Type what you want and press
  **Enter** (or **Esc** to close it). The **+** button attaches a file for the cat to look at.
  The microphone is a placeholder for voice input, which isn't built yet.
- When the cat is done, a **speech cloud** pops up above its head saying what it did:
  *Code changed*, *Text changed*, *File changed*, *Command run*, *Copied to clipboard* or
  *Noted*, plus a sentence or two about it. A question just gets its answer. Click the cloud
  to dismiss it, or it goes away by itself. An action only counts once it actually worked, so
  an edit you turned down never shows up as "Code changed".
- Before the cat **edits a file or runs a command**, the cloud asks first (for example "Can I
  change train.py?"), shows the change, and waits for **Allow** or **Don't**. Reading,
  searching and browsing happen without asking.
- **Right-click** the cat for these options: *Talk to Ginger*, *Stop what you're doing*,
  *Look at my screen*, *Watch mode*, *New conversation*, *Quit*.
- Each instruction automatically includes the app or file you were in just before clicking
  the cat, so "fix this" just works.
- The cat's moods: `...` means it's thinking, and a red `!` means it has a tip for you.
- **Nap time:** after 30 seconds with nothing to do (`SLEEP_AFTER_SEC`), the cat curls up,
  rests its head on its paws and breathes slowly while little z's float up from its head.
  It stays awake while the input bar or a cloud is open. Click it and it opens its eyes, has
  a big stretch and a yawn, and then the input bar appears.

  ![Ginger napping](docs/cat_sleeping.gif)
- **Coding mode:** while the cat is working on a coding task, it puts on its glasses and types
  away on a little laptop, stopping now and then to read the screen. A task counts as coding
  if you asked from a code editor or terminal, if your message sounds like code (bugs, tests,
  `.py` files and so on), or as soon as the cat starts editing files or running commands. The
  glasses come off when it's done. The rules are in [`catagent/activity.py`](catagent/activity.py).

  ![Ginger putting on glasses to code](docs/cat_coding.gif)

### Watch mode

With watch mode on, the cat glances at your screen whenever you switch to something new
and stay on it for a little while. It checks at most once every
`WATCH_INTERVAL_SEC` (2 minutes by default). If it spots a real problem, such as an error
message, a failing test, a LaTeX error or an obvious typo, it says so in its speech cloud
and shows a red `!`. If not, it stays
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
- `SLEEP_AFTER_SEC`: how long before it naps
- `WATCH_INTERVAL_SEC`

The cat's look is defined in [`catagent/sprites.py`](catagent/sprites.py) as ASCII grids,
one letter per pixel. Edit the letters to redraw it, or add frames.

Its memory is a plain file, `~/.ginger/memory.md`, that you can open and edit yourself.

## How it fits together

```
run.py
catagent/
  app.py          wires everything together; window placement, watch mode, saved state
  cat_widget.py   the draggable, animated cat window
  command_bar.py  the input bar that pops up under the cat
  speech_cloud.py the pixel speech cloud: results, tips and Allow/Don't questions
  agent.py        the Claude Agent SDK session, on a background asyncio thread
  activity.py     what counts as coding, and the "Code changed" style labels
  tools.py        the cat's own tools (screen, context, tips, clipboard, memory)
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
