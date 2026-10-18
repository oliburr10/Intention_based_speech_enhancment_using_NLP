"""The cat's own tools, served in-process to the agent as the MCP server "cat"."""

import asyncio
import base64
from typing import Callable

from claude_agent_sdk import create_sdk_mcp_server, tool

from . import config, context, memory

SERVER_NAME = "cat"


def _text(msg: str, is_error: bool = False) -> dict:
    result = {"content": [{"type": "text", "text": msg}]}
    if is_error:
        result["is_error"] = True
    return result


def build_server(
    watcher: context.ContextWatcher,
    show_tip: Callable[[str, str, str, int], None],
    set_hidden: Callable[[bool], None],
):
    """Create the tool server. show_tip and set_hidden are thread-safe callbacks into the UI."""

    @tool(
        "what_am_i_working_on",
        "Which app, window, browser tab or document the user was using most recently "
        "(not counting the cat itself).",
        {},
    )
    async def what_am_i_working_on(args):
        ctx = watcher.last()
        if ctx is None:
            return _text("Unknown: no app has been seen yet (or this isn't macOS).")
        return _text(ctx.describe())

    @tool(
        "look_at_screen",
        "Take a screenshot of the user's main display to see what they are looking at.",
        {},
    )
    async def look_at_screen(args):
        if config.HIDE_DURING_SCREENSHOT:
            set_hidden(True)
            await asyncio.sleep(0.4)  # give the windows time to disappear
        try:
            png = await asyncio.to_thread(context.screenshot)
        except Exception as e:
            return _text(f"Couldn't take a screenshot: {e}", is_error=True)
        finally:
            if config.HIDE_DURING_SCREENSHOT:
                set_hidden(False)
        return {
            "content": [
                {"type": "image", "data": base64.b64encode(png).decode(), "mimeType": "image/png"},
                {"type": "text", "text": "Screenshot of the user's main display."},
            ]
        }

    @tool(
        "show_tip",
        "Pop up a short speech bubble next to the cat on the desktop. Use it to point the user "
        "at something: what's wrong and exactly where. If the problem is in a local file, pass "
        "file_path (and line) so the bubble gets a 'Show me' button that opens it.",
        {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "A few words, e.g. 'Missing bracket'"},
                "message": {"type": "string", "description": "One to three short sentences."},
                "file_path": {"type": "string"},
                "line": {"type": "integer"},
            },
            "required": ["title", "message"],
        },
    )
    async def show_tip_tool(args):
        show_tip(args["title"], args["message"], args.get("file_path", ""), int(args.get("line", 0)))
        return _text("Tip shown.")

    @tool(
        "open_in_editor",
        "Open a local file for the user, in VS Code at a specific line when possible.",
        {
            "type": "object",
            "properties": {"file_path": {"type": "string"}, "line": {"type": "integer"}},
            "required": ["file_path"],
        },
    )
    async def open_in_editor(args):
        return _text(context.open_in_editor(args["file_path"], int(args.get("line", 0))))

    @tool(
        "copy_to_clipboard",
        "Put text (code, a rewritten paragraph, a command) on the user's clipboard so they can "
        "paste it. Use this whenever they ask for something to copy, since your reply only "
        "shows up in a small speech cloud.",
        {"text": str},
    )
    async def copy_to_clipboard(args):
        return _text(context.copy_to_clipboard(args["text"]))

    @tool(
        "remember",
        "Save a lasting fact about the user (a preference, a project, how they like things "
        "done) to long-term memory.",
        {"fact": str},
    )
    async def remember(args):
        memory.add(args["fact"])
        return _text("Remembered.")

    tools = [
        what_am_i_working_on, look_at_screen, show_tip_tool, open_in_editor, copy_to_clipboard, remember,
    ]
    server = create_sdk_mcp_server(SERVER_NAME, tools=tools)
    names = [f"mcp__{SERVER_NAME}__{t.name}" for t in tools]
    return server, names
