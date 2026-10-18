"""The cat's brain: a Claude Agent SDK session running on a background asyncio thread.

Qt owns the main thread, so all agent work happens on a separate event loop. Results come
back to the UI through Qt signals, which are safe to emit from any thread.
"""

import asyncio
import json
import sys
import threading
import uuid
import warnings

from claude_agent_sdk import (
    AssistantMessage,
    CanUseToolShadowedWarning,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from PySide6.QtCore import QObject, Signal

from . import activity, config, context, memory, tools

# Tools the cat may use without asking: they only read or look things up.
# Everything else goes through can_use_tool, which asks you in the speech cloud.
SAFE_TOOLS = ["Read", "Glob", "Grep", "WebSearch", "WebFetch"]
warnings.filterwarnings("ignore", category=CanUseToolShadowedWarning)

WATCH_PROMPT = """\
[Watch mode check-in. The user is not talking to you right now.]
They are currently in:
{context}

Glance at their screen with look_at_screen. If you see a clear, concrete problem they would \
want to know about right now (an error message, a failing test, a bug in visible code, a \
LaTeX error, an obvious typo or wrong fact in their document), call show_tip once with what \
is wrong and exactly where. Otherwise do nothing. Do not edit anything. End by replying \
with a single word: DONE."""


def describe_tool(name: str, args: dict) -> str:
    short = name.removeprefix(f"mcp__{tools.SERVER_NAME}__")
    return f"use {short}: {json.dumps(args)[:200]}"


def permission_question(name: str, args: dict) -> tuple[str, str]:
    """What the cat asks before using a tool that changes something: (question, details)."""
    file_name = str(args.get("file_path", "")).rsplit("/", 1)[-1]
    if name == "Edit":
        old, new = args.get("old_string", ""), args.get("new_string", "")
        return f"Can I change {file_name}?", f"{_clip(old)}\n→\n{_clip(new)}"
    if name == "Write":
        return f"Can I write {file_name}?", _clip(args.get("content", ""))
    if name == "Bash":
        return "Can I run this?", _clip(args.get("command", ""))
    return f"Can I {describe_tool(name, args)}?", ""


def _clip(text: str, limit: int = 220) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


class AgentBridge(QObject):
    finished = Signal(str, str)                   # what it did ("Code changed · Command run"), its reply
    failed = Signal(str)                          # what went wrong
    busy_changed = Signal(bool)
    coding = Signal()                             # the current chat turn turned out to be about code
    permission_requested = Signal(str, str, str)  # request id, question, details
    tip = Signal(str, str, str, int)              # title, message, file path, line
    hide_requested = Signal(bool)                 # hide/show the cat's windows around screenshots

    def __init__(self, watcher: context.ContextWatcher):
        super().__init__()
        self._watcher = watcher
        self._loop = asyncio.new_event_loop()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._pending: dict[str, asyncio.Future] = {}
        self._client: ClaudeSDKClient | None = None
        self._watching = False
        self._reset_turn()
        self.busy = False
        self._thread = threading.Thread(target=self._run, name="cat-agent", daemon=True)

    # ---- called from the Qt thread -------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def send(self, text: str) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, ("chat", text))

    def new_conversation(self) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, ("reset", None))

    def stop_turn(self) -> None:
        for request_id in list(self._pending):
            self.resolve_permission(request_id, False)
        if self._client is not None:
            asyncio.run_coroutine_threadsafe(self._client.interrupt(), self._loop)

    def resolve_permission(self, request_id: str, allow: bool) -> None:
        self._loop.call_soon_threadsafe(self._resolve, request_id, allow)

    def watch_check(self, context_text: str) -> None:
        if not self.busy and not self._watching:
            asyncio.run_coroutine_threadsafe(self._watch_check(context_text), self._loop)

    def shutdown(self) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, ("quit", None))
        self._thread.join(timeout=5)

    # ---- the agent thread -----------------------------------------------------------------

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    def _options(self, watch: bool = False) -> ClaudeAgentOptions:
        server, cat_tools = tools.build_server(
            self._watcher,
            show_tip=self.tip.emit,
            set_hidden=self.hide_requested.emit,
        )
        known = memory.load()
        prompt = config.PERSONALITY
        if known:
            prompt += f"\nWhat you remember about {config.USER_NAME}:\n{known}\n"
        options = ClaudeAgentOptions(
            model=config.MODEL,
            cwd=str(config.WORKSPACE),
            system_prompt={"type": "preset", "preset": "claude_code", "append": prompt},
            mcp_servers={tools.SERVER_NAME: server},
            allowed_tools=SAFE_TOOLS + cat_tools,
            can_use_tool=self._deny_all if watch else self._ask_user,
        )
        if watch:
            options.disallowed_tools = ["Edit", "Write", "NotebookEdit", "Bash"]
            options.max_turns = 8
        return options

    async def _main(self) -> None:
        while True:
            kind, payload = await self._queue.get()
            if kind == "quit":
                break
            if kind == "reset":
                await self._disconnect()
                self.finished.emit("Fresh start! 🐾", "I've forgotten our conversation (but not what I remember about you).")
                continue
            await self._chat_turn(payload)
        await self._disconnect()

    def _reset_turn(self) -> None:
        self._done: list[str] = []                 # things the cat actually did this turn
        self._in_flight: dict[str, str] = {}       # tool_use_id -> what it will have done if it works
        self._reply = ""
        self._error = ""

    async def _chat_turn(self, prompt: str) -> None:
        self._reset_turn()
        self._set_busy(True)
        try:
            if self._client is None:
                client = ClaudeSDKClient(options=self._options())
                await client.connect()
                self._client = client
            await self._client.query(prompt)
            async for msg in self._client.receive_response():
                self._handle(msg)
        except Exception as e:
            self._error = str(e)
            await self._disconnect()
        finally:
            self._set_busy(False)
        if self._error:
            self.failed.emit(self._error)
        else:
            self.finished.emit(" · ".join(self._done), self._reply)

    def _handle(self, msg) -> None:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    self._reply = block.text.strip()
                elif isinstance(block, ToolUseBlock):
                    label = activity.action_label(block.name, block.input)
                    if label:
                        self._in_flight[block.id] = label
                    if activity.is_coding_tool(block.name, block.input):
                        self.coding.emit()
        elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
            # A tool only counts as done once its result comes back without an error,
            # so a denied or failed edit never shows up as "Code changed".
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    label = self._in_flight.pop(block.tool_use_id, None)
                    if label and not block.is_error and label not in self._done:
                        self._done.append(label)
        elif isinstance(msg, ResultMessage) and msg.is_error:
            self._error = "; ".join(msg.errors or []) or msg.subtype

    async def _watch_check(self, context_text: str) -> None:
        self._watching = True
        try:
            async with ClaudeSDKClient(options=self._options(watch=True)) as client:
                await client.query(WATCH_PROMPT.format(context=context_text))
                async for _ in client.receive_response():
                    pass
        except Exception as e:
            print(f"Watch mode check failed: {e}", file=sys.stderr)
        finally:
            self._watching = False

    async def _disconnect(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.disconnect()
            except Exception:
                pass

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.busy_changed.emit(busy)

    # ---- permissions ------------------------------------------------------------------------

    async def _ask_user(self, tool_name, tool_input, ctx):
        request_id = uuid.uuid4().hex
        future = self._loop.create_future()
        self._pending[request_id] = future
        self.permission_requested.emit(request_id, *permission_question(tool_name, tool_input))
        if await future:
            return PermissionResultAllow()
        return PermissionResultDeny(message=f"{config.USER_NAME} said no. Ask what they'd prefer instead.")

    async def _deny_all(self, tool_name, tool_input, ctx):
        return PermissionResultDeny(message="Not allowed during a watch-mode check.")

    def _resolve(self, request_id: str, allow: bool) -> None:
        future = self._pending.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(allow)
