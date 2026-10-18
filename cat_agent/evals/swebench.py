"""SWE-bench for the cat: `python -m evals.swebench` (from the cat_agent folder).

SWE-bench is the standard benchmark for AI software engineers: real GitHub issues from real
Python projects (Django, sympy, scikit-learn, matplotlib, pytest, ...), each graded by
running the project's own test suite. Verified, the default here, is a set of 500 issues
that people have checked are fair and solvable.

For every task this:
  1. starts the task's official Docker container (the project at the right commit, with
     everything installed) and copies the project out to a folder for the cat;
  2. lets the cat work on it like any other request, with the same setup the app uses,
     plus a run_in_project_env tool that runs commands inside the container with its
     latest edits synced in, so it can reproduce the bug and run the tests;
  3. saves its change as a patch in SWE-bench's predictions format;
and then grades every patch with the official SWE-bench harness, against tests the cat
never saw. The score is the share of issues resolved.

You need Docker running and `pip install -r evals/requirements-swebench.txt`. Each task is
a real coding session (often a few minutes and around $0.50-$3), so start with a small
sample. Results go to ~/.ginger/swebench/<run id>/ and a run can be resumed.
"""

import argparse
import asyncio
import io
import json
import random
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    PermissionResultAllow,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catagent import config, context  # noqa: E402
from catagent.agent import build_options  # noqa: E402

DEFAULT_DATASET = "SWE-bench/SWE-bench_Verified"
RUNS = config.HOME_DIR / "swebench"
WORKDIR = "/testbed"           # where the project lives inside every SWE-bench container
MAX_OUTPUT = 12_000            # characters of command output shown to the cat
ENV_TOOL = "mcp__swe__run_in_project_env"

PROMPT = """\
You're in a checkout of {repo} (the current folder). Someone filed this issue:

<issue>
{problem_statement}
</issue>

Resolve the issue by changing the project's source code, so the problem is fixed in \
general and not just for the example in the issue.

Your own shell doesn't have this project's dependencies. To run Python, reproduce the \
problem or run the project's tests, use the run_in_project_env tool: it runs commands in a \
container where the project is installed, and it always sees your latest edits.

Your change will be checked with tests you can't see, so don't change the existing tests, \
and remove any scratch files you create. When you're done, reply with a short summary."""


def docker(*args: str, input: bytes | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], input=input, capture_output=True, timeout=timeout)


class Testbed:
    """A task's container, plus a copy of its project folder for the cat to edit."""

    def __init__(self, instance: dict, folder: Path):
        self.instance = instance
        self.folder = folder
        self.name = "ginger-swe-" + re.sub(r"[^a-z0-9_.-]", "-", instance["instance_id"].lower())
        self.synced: set[str] = set()
        self.prelude = self._prelude(instance.get("eval_script") or "")

    @staticmethod
    def _prelude(eval_script: str) -> str:
        """The environment activation lines from the task's own test script (e.g. conda)."""
        lines = [
            line.strip() for line in eval_script.splitlines()
            if re.search(r"\bactivate\b", line) and not line.strip().startswith("#")
        ]
        return " && ".join(lines)

    def start(self) -> None:
        docker("rm", "-f", self.name)
        result = docker("run", "-d", "--name", self.name, self.instance["image"], "tail", "-f", "/dev/null", timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"couldn't start {self.instance['image']}: {result.stderr.decode()[-500:]}")
        self.folder.mkdir(parents=True, exist_ok=True)
        result = docker("cp", f"{self.name}:{WORKDIR}/.", str(self.folder), timeout=1800)
        if result.returncode != 0:
            raise RuntimeError(f"couldn't copy the project out: {result.stderr.decode()[-500:]}")

    def _git(self, *args: str) -> str:
        return subprocess.run(["git", "-C", str(self.folder), *args], capture_output=True, text=True).stdout

    def sync(self) -> None:
        """Copy every file the cat has changed (or changed back) into the container."""
        status = self._git("status", "--porcelain=v1", "-z", "--untracked-files=all")
        entries = [e for e in status.split("\0") if e]
        changed = set()
        i = 0
        while i < len(entries):
            code, path = entries[i][:2], entries[i][3:]
            changed.add(path)
            if code[0] in "RC":  # renames and copies are followed by the old path
                i += 1
                changed.add(entries[i])
            i += 1
        paths = changed | self.synced
        self.synced = changed
        present = [p for p in paths if (self.folder / p).is_file()]
        missing = [p for p in paths if not (self.folder / p).exists()]
        if present:
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as tar:
                for p in present:
                    tar.add(self.folder / p, arcname=p)
            docker("exec", "-i", self.name, "tar", "-x", "-C", WORKDIR, input=buffer.getvalue(), timeout=300)
        if missing:
            docker("exec", "-w", WORKDIR, self.name, "rm", "-rf", "--", *missing, timeout=60)

    def run(self, command: str, timeout: int) -> str:
        self.sync()
        script = f"{self.prelude + ' && ' if self.prelude else ''}cd {WORKDIR} && {command}"
        try:
            result = docker("exec", self.name, "bash", "-lc", script, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"(timed out after {timeout}s)"
        output = (result.stdout + result.stderr).decode(errors="replace")
        if len(output) > MAX_OUTPUT:
            output = f"(… {len(output) - MAX_OUTPUT} characters cut …)\n" + output[-MAX_OUTPUT:]
        return f"{output}\n(exit code {result.returncode})"

    def patch(self) -> str:
        """Everything the cat changed, as a patch (new files included)."""
        self._git("add", "-A", "--intent-to-add")
        return self._git("diff")

    def stop(self) -> None:
        docker("rm", "-f", self.name)


async def solve(instance: dict, folder: Path, budget: float, minutes: int) -> dict:
    """Let the cat work on one task. Returns its patch and a record of what it did."""
    testbed = Testbed(instance, folder)
    record = {"instance_id": instance["instance_id"], "tool_calls": [], "reply": "", "cost": 0.0, "error": ""}
    start = time.monotonic()
    try:
        await asyncio.to_thread(testbed.start)

        @tool(
            "run_in_project_env",
            "Run a shell command in the project's environment (a container with the project and "
            "all its dependencies installed), from the project folder, with your latest edits. "
            "Use it to run Python, reproduce the issue and run tests.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "description": "default 600"},
                },
                "required": ["command"],
            },
        )
        async def run_in_project_env(args):
            output = await asyncio.to_thread(testbed.run, args["command"], int(args.get("timeout_seconds", 600)))
            return {"content": [{"type": "text", "text": output}]}

        async def allow(*_):
            return PermissionResultAllow()

        options = build_options(context.ContextWatcher(), lambda *a: None, lambda *a: None, allow, cwd=str(folder))
        options.mcp_servers["swe"] = create_sdk_mcp_server("swe", tools=[run_in_project_env])
        options.allowed_tools.append(ENV_TOOL)
        options.max_budget_usd = budget

        prompt = PROMPT.format(repo=instance["repo"], problem_statement=instance["problem_statement"])

        async def work():
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, ToolUseBlock):
                                record["tool_calls"].append({"tool": block.name, "input": block.input})
                            elif isinstance(block, TextBlock) and block.text.strip():
                                record["reply"] = block.text.strip()
                    elif isinstance(msg, ResultMessage):
                        record["cost"] = msg.total_cost_usd or 0.0
                        if msg.is_error:
                            record["error"] = "; ".join(msg.errors or []) or msg.subtype

        try:
            await asyncio.wait_for(work(), timeout=minutes * 60)
        except asyncio.TimeoutError:
            record["error"] = f"ran out of time ({minutes} min)"
        record["patch"] = await asyncio.to_thread(testbed.patch)
    except Exception as e:
        record["error"] = str(e)
        record.setdefault("patch", "")
    finally:
        await asyncio.to_thread(testbed.stop)
    record["seconds"] = round(time.monotonic() - start)
    return record


def load_tasks(dataset: str, split: str) -> list[dict]:
    from swebench.harness.utils import load_swebench_dataset
    return [dict(i) for i in load_swebench_dataset(dataset, split)]


def pick(tasks: list[dict], args) -> list[dict]:
    if args.instances:
        by_id = {t["instance_id"]: t for t in tasks}
        unknown = [i for i in args.instances if i not in by_id]
        if unknown:
            sys.exit(f"Not in the dataset: {', '.join(unknown)}")
        return [by_id[i] for i in args.instances]
    if args.all:
        return tasks
    return random.Random(args.seed).sample(tasks, min(args.sample, len(tasks)))


def evaluate(run_dir: Path, run_id: str, dataset: str, split: str, ids: list[str], predictions: str, workers: int) -> dict:
    """Grade the patches with the official SWE-bench harness and return its report."""
    print(f"\nGrading {len(ids)} task(s) with the official SWE-bench harness (this runs each project's tests in Docker)...")
    subprocess.run(
        [sys.executable, "-m", "swebench.harness.run_evaluation",
         "--dataset_name", dataset, "--split", split, "--instance_ids", *ids,
         "--predictions_path", predictions, "--max_workers", str(workers),
         "--run_id", run_id, "--report_dir", str(run_dir)],
        cwd=run_dir, check=False,
    )
    reports = sorted(run_dir.glob(f"*.{run_id}.json"), key=lambda p: p.stat().st_mtime)
    if not reports:
        sys.exit("The harness didn't write a report; see the log above.")
    return json.loads(reports[-1].read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the cat on SWE-bench and grade it with the official harness.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face name or a local .json/.jsonl file")
    parser.add_argument("--split", default="test")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--instances", nargs="+", help="specific task IDs, e.g. django__django-11099")
    which.add_argument("--sample", type=int, default=10, help="a random sample of this many tasks (default 10)")
    which.add_argument("--all", action="store_true", help="every task in the split")
    parser.add_argument("--seed", type=int, default=42, help="which random sample (same seed, same tasks)")
    parser.add_argument("--run-id", help="name of the run; reuse one to resume it")
    parser.add_argument("--budget", type=float, default=5.0, help="max dollars per task (default 5)")
    parser.add_argument("--minutes", type=int, default=45, help="max minutes per task (default 45)")
    parser.add_argument("--workers", type=int, default=2, help="parallel graders (default 2)")
    parser.add_argument("--gold", action="store_true",
                        help="grade SWE-bench's own reference fixes instead of the cat's: a free check that grading works")
    parser.add_argument("--no-grade", action="store_true", help="only collect the cat's patches")
    args = parser.parse_args()

    if not shutil.which("docker") or docker("info").returncode != 0:
        sys.exit("Docker isn't running. Start Docker Desktop and try again.")

    print(f"Loading {args.dataset} ({args.split})...")
    tasks = pick(load_tasks(args.dataset, args.split), args)
    ids = [t["instance_id"] for t in tasks]
    missing_image = [t["instance_id"] for t in tasks if not t.get("image")]
    if missing_image and not args.gold:
        sys.exit("This dataset doesn't name a Docker image per task; use a current SWE-bench dataset.")

    run_id = args.run_id or ("gold" if args.gold else "run") + datetime.now().strftime("-%Y%m%d-%H%M")
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} task(s), run {run_id}, results in {run_dir}")

    if args.gold:
        report = evaluate(run_dir, run_id, args.dataset, args.split, ids, "gold", args.workers)
        print(f"\nReference fixes resolved {len(report['resolved_ids'])}/{len(ids)}: grading works." if
              len(report["resolved_ids"]) == len(ids) else
              f"\nOnly {len(report['resolved_ids'])}/{len(ids)} reference fixes passed; grading isn't working properly.")
        return 0

    # Use a throwaway memory so the benchmark neither reads nor changes the real one.
    config.MEMORY_FILE = Path(tempfile.mkdtemp(prefix="cat-swebench-memory-")) / "memory.md"
    model_name = f"{config.CAT_NAME.lower()}-cat-{config.MODEL}"
    predictions_path = run_dir / "predictions.jsonl"
    done = set()
    if predictions_path.exists():
        done = {json.loads(line)["instance_id"] for line in predictions_path.read_text().splitlines() if line}
    (run_dir / "trajectories").mkdir(exist_ok=True)

    total_cost = 0.0
    for n, task in enumerate(tasks, 1):
        iid = task["instance_id"]
        if iid in done:
            print(f"[{n}/{len(tasks)}] {iid}: already done, skipping")
            continue
        print(f"[{n}/{len(tasks)}] {iid}: working...", flush=True)
        with tempfile.TemporaryDirectory(prefix="cat-swe-") as tmp:
            record = asyncio.run(solve(task, Path(tmp) / "project", args.budget, args.minutes))
        total_cost += record["cost"]
        (run_dir / "trajectories" / f"{iid}.json").write_text(json.dumps(record, indent=2))
        with predictions_path.open("a") as f:
            f.write(json.dumps({"instance_id": iid, "model_name_or_path": model_name, "model_patch": record["patch"]}) + "\n")
        changed = len(re.findall(r"^diff --git", record["patch"], re.MULTILINE))
        status = f"error: {record['error']}" if record["error"] else f"changed {changed} file(s)"
        print(f"    {status}; {record['seconds']}s, ${record['cost']:.2f}, {len(record['tool_calls'])} tool calls")

    if args.no_grade:
        print(f"\nPatches saved to {predictions_path}")
        return 0

    report = evaluate(run_dir, run_id, args.dataset, args.split, ids, str(predictions_path), args.workers)
    resolved = set(report["resolved_ids"])
    costs = {}
    for iid in ids:
        trajectory = run_dir / "trajectories" / f"{iid}.json"
        costs[iid] = json.loads(trajectory.read_text())["cost"] if trajectory.exists() else 0.0
    print(f"\n{config.CAT_NAME} on {args.dataset} ({run_id})")
    for iid in ids:
        print(f"  {'✅' if iid in resolved else '❌'} {iid}  (${costs[iid]:.2f})")
    share = 100 * len(resolved) / len(ids)
    print(f"\nResolved {len(resolved)}/{len(ids)} ({share:.0f}%), model cost ${sum(costs.values()):.2f}")
    if len(ids) < 50:
        print("A small sample gives a rough number: rerun with a bigger --sample (or --all) for a precise one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
