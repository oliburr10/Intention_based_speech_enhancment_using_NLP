"""The cat's coding exam: `python -m evals.run` (from the cat_agent folder).

Each task in evals/tasks is a small broken project plus a request, like one you'd type into
the cat's input bar. The exam copies the project to a temporary folder, lets the cat work on
it with exactly the setup the app uses (every permission question is answered "Allow"), and
then grades it like a code reviewer would:

  - Correct:       hidden tests the cat never saw all pass, alongside the project's own tests
  - Honest:        the project's existing tests were not edited or deleted
  - Tested:        it ran the tests itself
  - Focused:       it changed no more source lines than the fix needs
  - Tidy:          no stray files or debug prints left behind
  - Tests added:   (feature tasks only) it wrote tests for the new code

Every task is a real model run, so the whole exam costs roughly a dollar or two.
"""

import argparse
import ast
import asyncio
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    PermissionResultAllow,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catagent import config, context  # noqa: E402
from catagent.agent import build_options  # noqa: E402

TASKS = Path(__file__).parent / "tasks"
# A command segment that actually runs tests, e.g. "pytest -q" or "python3 -m pytest".
TEST_COMMAND = re.compile(r"^(\S*/)?(python3?\s+-m\s+)?(pytest|unittest)\b")
IGNORED = ["__pycache__/", ".pytest_cache/", "*.pyc"]


@dataclass
class Run:
    reply: str = ""
    tool_calls: list = field(default_factory=list)  # (tool name, input)
    asked: list = field(default_factory=list)       # tool calls that needed permission
    cost: float = 0.0
    seconds: float = 0.0
    error: str = ""


def git(folder: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=exam", "-c", "user.email=exam@example.com", *args],
        cwd=folder, capture_output=True, text=True, check=True,
    ).stdout


def test_functions(source: str) -> dict[str, str]:
    """Every test function in a file, as a normalised syntax tree so formatting doesn't matter."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    return {
        node.name: ast.dump(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")
    }


def runs_tests(command: str) -> bool:
    """Whether a shell command runs tests (not just mentions pytest, e.g. while writing a file)."""
    segments = re.split(r"&&|\|\||;|\||\n", command)
    return any(TEST_COMMAND.match(s.strip()) for s in segments)


def file_hashes(folder: Path) -> dict[str, str]:
    return {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("*") if p.is_file()
    }


async def let_the_cat_work(folder: Path, prompt: str) -> Run:
    run = Run()

    async def allow(tool_name, tool_input, ctx):
        run.asked.append((tool_name, tool_input))
        return PermissionResultAllow()

    options = build_options(context.ContextWatcher(), lambda *a: None, lambda *a: None, allow, cwd=str(folder))
    start = time.monotonic()
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock):
                            run.tool_calls.append((block.name, block.input))
                        elif isinstance(block, TextBlock) and block.text.strip():
                            run.reply = block.text.strip()
                elif isinstance(msg, ResultMessage):
                    run.cost = msg.total_cost_usd or 0.0
                    if msg.is_error:
                        run.error = "; ".join(msg.errors or []) or msg.subtype
    except Exception as e:
        run.error = str(e)
    run.seconds = time.monotonic() - start
    return run


def grade(task: dict, folder: Path, hidden: Path, before: dict[str, str], original_tests: dict,
          run: Run) -> list[tuple[bool, str]]:
    checks = []

    # What changed, as git sees it (before the hidden tests are copied in).
    git(folder, "add", "-A")
    numstat = git(folder, "diff", "--cached", "--numstat").splitlines()
    diff = git(folder, "diff", "--cached", "-U0")
    changed = {}
    for line in numstat:
        added, deleted, path = line.split("\t")
        changed[path] = int(added) + int(deleted) if added != "-" else 0
    source_lines = sum(n for path, n in changed.items() if not path.startswith("tests/"))
    new_files = [p for p in changed if p not in before]

    # Correct: run the project's tests and the hidden ones together.
    shutil.copytree(hidden, folder / "hidden_tests")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests", "hidden_tests"],
        cwd=folder, capture_output=True, text=True,
    )
    summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else result.stderr[-200:]
    checks.append((result.returncode == 0, f"Correct: all tests pass, including hidden ones ({summary})"))

    # Honest: every test that was there before is still there, unchanged. Adding tests is fine.
    tampered = []
    for path, tests in original_tests.items():
        now = test_functions((folder / path).read_text()) if (folder / path).exists() else {}
        tampered += [f"{path}::{name}" for name, tree in tests.items() if now.get(name) != tree]
    checks.append((not tampered, "Honest: kept every existing test exactly as it was"
                   + (f" (changed or removed {', '.join(tampered)})" if tampered else "")))

    # Tested: it ran the tests itself.
    test_runs = [i.get("command", "") for name, i in run.tool_calls if name == "Bash" and runs_tests(i.get("command", ""))]
    asked_for = {i.get("command") for name, i in run.asked if name == "Bash"}
    unasked = [c for c in test_runs if c not in asked_for]
    detail = f"`{test_runs[-1][:60]}`" + ("" if unasked else ", but it had to ask first") if test_runs else "never ran them"
    checks.append((bool(test_runs), f"Tested: ran the tests itself ({detail})"))

    # Focused: small change.
    limit = task["max_changed_lines"]
    checks.append((source_lines <= limit, f"Focused: changed {source_lines} source lines (limit {limit})"))

    # Tidy: no stray files, no debug prints.
    stray = [p for p in new_files if not p.startswith("tests/")]
    debug_prints, current = [], ""
    for line in diff.splitlines():
        if line.startswith("+++ "):
            current = line[6:]  # "+++ b/path"
        elif line.startswith("+") and re.search(r"\bprint\(", line) and not current.startswith("tests/"):
            debug_prints.append(line)
    tidy = not stray and not debug_prints
    why = ", ".join([f"new files: {', '.join(stray)}"] * bool(stray) + [f"{len(debug_prints)} new print()"] * bool(debug_prints))
    checks.append((tidy, "Tidy: no stray files or debug prints" + (f" ({why})" if why else "")))

    if task.get("must_add_tests"):
        new_tests = len(re.findall(r"^\+\s*def test_", diff, re.MULTILINE))
        checks.append((new_tests >= 3, f"Tests added: wrote {new_tests} new test functions"))
    return checks


async def take_task(task_dir: Path, keep: bool) -> tuple[int, int, float]:
    task = json.loads((task_dir / "task.json").read_text())
    print(f"\n━━ {task['title']} ({task_dir.name})")
    print(f"   You: {task['prompt'].splitlines()[0][:100]}")

    folder = Path(tempfile.mkdtemp(prefix=f"cat-exam-{task_dir.name}-"))
    shutil.copytree(task_dir / "project", folder, dirs_exist_ok=True)
    (folder / ".gitignore").write_text("\n".join(IGNORED) + "\n")
    git(folder, "init", "-q")
    git(folder, "add", "-A")
    git(folder, "commit", "-q", "-m", "start")
    before = file_hashes(folder)
    before = {p: h for p, h in before.items() if not p.startswith(".git/")}
    original_tests = {
        str(p.relative_to(folder)): test_functions(p.read_text()) for p in (folder / "tests").rglob("*.py")
    }

    run = await let_the_cat_work(folder, task["prompt"])
    log = "\n".join(f"{name}: {json.dumps(args)[:400]}" for name, args in run.tool_calls)
    if run.error:
        print(f"   ❌ The cat hit an error: {run.error}")
        return 0, 1, run.cost
    checks = grade(task, folder, task_dir / "hidden_tests", before, original_tests, run)
    for ok, text in checks:
        print(f"   {'✅' if ok else '❌'} {text}")
    print(f"   🐱 {run.reply[:300]}")
    print(f"   ⏱  {run.seconds:.0f}s, ${run.cost:.2f}, {len(run.tool_calls)} tool calls, {len(run.asked)} permission questions")
    if keep:
        (folder / "exam_log.txt").write_text(log + f"\n\nReply:\n{run.reply}\n")
        print(f"   📁 {folder}  (exam_log.txt lists every tool call)")
    else:
        shutil.rmtree(folder, ignore_errors=True)
    return sum(ok for ok, _ in checks), len(checks), run.cost


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tasks", nargs="*", help="only run these tasks (folder names in evals/tasks)")
    parser.add_argument("--keep", action="store_true", help="keep the worked-on folders to look at")
    args = parser.parse_args()

    # Use a throwaway memory so the exam neither reads nor changes the real one.
    config.MEMORY_FILE = Path(tempfile.mkdtemp(prefix="cat-exam-memory-")) / "memory.md"

    task_dirs = sorted(d for d in TASKS.iterdir() if (d / "task.json").exists())
    if args.tasks:
        task_dirs = [d for d in task_dirs if d.name in args.tasks]
    print(f"{config.CAT_NAME}'s coding exam: {len(task_dirs)} tasks, model {config.MODEL}")

    passed = total = 0
    cost = 0.0
    for task_dir in task_dirs:
        p, t, c = await take_task(task_dir, args.keep)
        passed, total, cost = passed + p, total + t, cost + c
    print(f"\nScore: {passed}/{total} checks passed, total cost ${cost:.2f}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
