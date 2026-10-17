"""The cat's brain: a Claude Agent SDK session running on a background asyncio thread.

Qt owns the main thread, so all agent work happens on a separate event loop. Results come
back to the UI through Qt signals, which are safe to emit from any thread.
"""

import asyncio
import json
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
    ToolUseBlock,
)
from PySide6.QtCore import QObject, Signal

from . import config, context, memory, tools

# Tools the cat may use without asking: they only read or look things up.
# Everything else goes through can_use_tool, which asks you in the chat window.
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
    """A short human-readable line for a tool call, for the chat log and permission prompts."""
    short = name.removeprefix(f"mcp__{tools.SERVER_NAME}__")
    if name == "Bash":
        return f"Run: {args.get('command', '')}"
    if name in ("Edit", "Write", "Read", "NotebookEdit"):
        return f"{name}: {args.get('file_path', '')}"
    if name in ("Glob", "Grep"):
        return f"{name}: {args.get('pattern', '')}"
    if name == "WebSearch":
        return f"Search the web: {args.get('query', '')}"
    if name == "WebFetch":
        return f"Fetch: {args.get('url', '')}"
    if short == "look_at_screen":
        return "Looking at your screen"
    if short == "what_am_i_working_on":
        return "Checking what you're working on"
    return f"{short}: {json.dumps(args)[:120]}"


def permission_details(name: str, args: dict) -> str:
    """The longer text shown when the cat asks to use a tool."""
    if name == "Edit":
        return (
            f"Edit {args.get('file_path', '')}\n\n"
            f"Replace:\n{args.get('old_string', '')[:600]}\n\nWith:\n{args.get('new_string', '')[:600]}"
        )
    if name == "Write":
        return f"Write {args.get('file_path', '')}\n\n{args.get('content', '')[:800]}"
    if name == "Bash":
        return f"Run this command:\n\n{args.get('command', '')}"
    return describe_tool(name, args)


class AgentBridge(QObject):
    message = Signal(str, str)                    # kind ("assistant" | "tool" | "info" | "error"), text
    busy_changed = Signal(bool)
    permission_requested = Signal(str, str, str)  # request id, tool name, details
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
                self.message.emit("info", "Fresh start! 🐾")
                continue
            await self._chat_turn(payload)
        await self._disconnect()

    async def _chat_turn(self, prompt: str) -> None:
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
            self.message.emit("error", f"Something went wrong: {e}")
            await self._disconnect()
        finally:
            self._set_busy(False)

    def _handle(self, msg) -> None:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    self.message.emit("assistant", block.text)
                elif isinstance(block, ToolUseBlock) and block.name != "ToolSearch":
                    self.message.emit("tool", describe_tool(block.name, block.input))
        elif isinstance(msg, ResultMessage) and msg.is_error:
            detail = "; ".join(msg.errors or []) or msg.subtype
            self.message.emit("error", f"The turn ended with an error: {detail}")

    async def _watch_check(self, context_text: str) -> None:
        self._watching = True
        try:
            async with ClaudeSDKClient(options=self._options(watch=True)) as client:
                await client.query(WATCH_PROMPT.format(context=context_text))
                async for _ in client.receive_response():
                    pass
        except Exception as e:
            self.message.emit("error", f"Watch mode check failed: {e}")
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
        self.permission_requested.emit(request_id, tool_name, permission_details(tool_name, tool_input))
        if await future:
            return PermissionResultAllow()
        return PermissionResultDeny(message=f"{config.USER_NAME} said no. Ask what they'd prefer instead.")

    async def _deny_all(self, tool_name, tool_input, ctx):
        return PermissionResultDeny(message="Not allowed during a watch-mode check.")

    def _resolve(self, request_id: str, allow: bool) -> None:
        future = self._pending.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(allow)
