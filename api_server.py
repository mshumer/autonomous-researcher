from __future__ import annotations

import json
import queue
import re
import subprocess
import sys
import threading
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from dotenv import load_dotenv, set_key
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load environment variables from the repo's .env file so spawned processes inherit them.
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

MAIN_PATH = BASE_DIR / "main.py"

# Regex for stripping ANSI escape sequences (Rich colour codes, etc.).
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    """
    Remove ANSI colour / style escape sequences from terminal output.

    This is useful for building UIs that want plain text, while still
    retaining the original coloured output for advanced terminals.
    """
    return ANSI_ESCAPE_RE.sub("", text)


class UserCredentials(BaseModel):
    """API credentials passed per-request for multi-user deployments."""
    google_api_key: Optional[str] = Field(None, description="Google AI Studio API key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    modal_token_id: Optional[str] = Field(None, description="Modal token ID")
    modal_token_secret: Optional[str] = Field(None, description="Modal token secret")


class SingleExperimentRequest(BaseModel):
    """
    Request body for running a single-agent experiment.

    This directly maps to:
        python main.py "<task>" --mode single [--gpu GPU] [--model MODEL]
    """

    task: str = Field(
        ...,
        description=(
            "A short natural-language hypothesis to test. This is passed to the "
            "single research agent as the main experiment description."
        ),
        examples=[
            "Fine-tuning a small transformer on CIFAR-10 improves accuracy by more than 5%."
        ],
    )
    gpu: Optional[str] = Field(
        None,
        description=(
            "Optional GPU type to request for the Modal sandbox. "
            "Examples: 'T4', 'A10G', 'A100', 'any'. "
            "If omitted, the system uses its default (CPU-only or configured GPU)."
        ),
        examples=["T4"],
    )
    model: str = Field(
        "gemini-3-pro-preview",
        description=(
            "The LLM model to use for the experiment. "
            "Options: 'gemini-3-pro-preview' or 'claude-opus-4-5'."
        ),
        examples=["gemini-3-pro-preview", "claude-opus-4-5"],
    )
    test_mode: bool = Field(
        False,
        description="If true, runs in test mode with mock data (no LLM/GPU usage).",
    )
    credentials: Optional[UserCredentials] = Field(
        None,
        description="User-provided API credentials. If not provided, falls back to server environment.",
    )


class OrchestratorExperimentRequest(BaseModel):
    """
    Request body for running the multi-agent orchestrator.

    This maps to:
        python main.py "<task>" --mode orchestrator \\
            --num-agents N --max-rounds R --max-parallel P [--gpu GPU] [--model MODEL]
    """

    task: str = Field(
        ...,
        description=(
            "High-level research question for the orchestrator to investigate. "
            "The orchestrator will decompose this into multiple hypotheses and "
            "launch single-agent experiments as needed."
        ),
        examples=[
            "Characterize the scaling behaviour of depth vs width in small transformers."
        ],
    )
    gpu: Optional[str] = Field(
        None,
        description=(
            "Default GPU hint for experiments spawned by the orchestrator "
            "(e.g. 'T4', 'A10G', 'A100', 'any'). If omitted, falls back to the "
            "host default (CPU-only or configured GPU)."
        ),
        examples=["A10G"],
    )
    model: str = Field(
        "gemini-3-pro-preview",
        description=(
            "The LLM model to use for the experiment. "
            "Options: 'gemini-3-pro-preview' or 'claude-opus-4-5'."
        ),
        examples=["gemini-3-pro-preview", "claude-opus-4-5"],
    )
    num_agents: int = Field(
        3,
        ge=1,
        le=16,
        description=(
            "How many distinct single-agent researchers to launch in the first wave. "
            "This is passed as --num-agents."
        ),
    )
    max_rounds: int = Field(
        3,
        ge=1,
        le=10,
        description=(
            "Maximum number of orchestration rounds (waves of experiments). "
            "This is passed as --max-rounds."
        ),
    )
    max_parallel: int = Field(
        2,
        ge=1,
        le=16,
        description=(
            "Maximum number of experiments to run in parallel in a single wave. "
            "This is passed as --max-parallel."
        ),
    )
    test_mode: bool = Field(
        False,
        description="If true, runs in test mode with mock data (no LLM/GPU usage).",
    )
    credentials: Optional[UserCredentials] = Field(
        None,
        description="User-provided API credentials. If not provided, falls back to server environment.",
    )


class SummaryHistoryItem(BaseModel):
    """Minimal view of an agent step for sidebar summarization."""

    type: Literal["thought", "code", "result", "text"] = Field(
        ...,
        description="Type of step (kept small to control context size).",
    )
    content: str = Field(
        ..., description="Truncated text content of the step (agent thought or tool output)."
    )


class AgentSummaryRequest(BaseModel):
    """Request body for Gemini-lite sidebar summaries."""

    agent_id: str = Field(..., description="Sub-agent identifier")
    history: List[SummaryHistoryItem] = Field(
        ..., description="Last ~5 steps for this agent (already truncated on client)."
    )


class AgentSummaryResponse(BaseModel):
    """Shape returned to the frontend for sidebar rendering."""

    summary: str = Field(..., description="Short markdown-friendly finding")
    chart: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Optional chart spec with keys: title, type(line|bar), labels, series. "
            "Omitted if no obvious numeric progression."
        ),
    )


class CredentialStatus(BaseModel):
    """Report which required API credentials are present."""

    has_google_api_key: bool = Field(
        ...,
        description="True when GOOGLE_API_KEY is set to a non-placeholder value.",
    )
    has_anthropic_api_key: bool = Field(
        ...,
        description="True when ANTHROPIC_API_KEY is set to a non-placeholder value.",
    )
    has_modal_token: bool = Field(
        ...,
        description="True when both MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are set.",
    )


class CredentialUpdateRequest(BaseModel):
    """Payload for setting API credentials via the UI."""

    google_api_key: Optional[str] = Field(
        None, description="Full Google API key from AI Studio"
    )
    anthropic_api_key: Optional[str] = Field(
        None, description="Anthropic API key from https://console.anthropic.com"
    )
    modal_token_id: Optional[str] = Field(
        None, description="Modal token ID from https://modal.com/settings/tokens"
    )
    modal_token_secret: Optional[str] = Field(
        None, description="Modal token secret from https://modal.com/settings/tokens"
    )


class ProcessSummary(BaseModel):
    """
    Structured view of a completed CLI run.

    The stdout/stderr fields preserve all Rich formatting escape codes,
    while the *_plain variants provide the same content with ANSI codes stripped
    for easy rendering in front-ends.
    """

    mode: Literal["single", "orchestrator"] = Field(
        ...,
        description="Which execution mode was used.",
    )
    task: str = Field(
        ...,
        description="Original task/hypothesis passed to main.py.",
    )
    gpu: Optional[str] = Field(
        None,
        description="GPU hint passed to main.py (if any).",
    )
    command: List[str] = Field(
        ...,
        description="Exact command invoked by the API to run the experiment.",
    )
    started_at: datetime = Field(
        ...,
        description="UTC timestamp when the subprocess started.",
    )
    finished_at: datetime = Field(
        ...,
        description="UTC timestamp when the subprocess finished.",
    )
    duration_seconds: float = Field(
        ...,
        description="Wall-clock run duration in seconds.",
    )
    exit_code: int = Field(
        ...,
        description="Subprocess exit code. Zero usually indicates success.",
    )
    stdout: str = Field(
        ...,
        description="Raw stdout produced by main.py (including ANSI colour codes).",
    )
    stderr: str = Field(
        ...,
        description="Raw stderr produced by main.py (including ANSI colour codes).",
    )
    stdout_plain: str = Field(
        ...,
        description="Stdout with ANSI escape codes stripped for simple rendering.",
    )
    stderr_plain: str = Field(
        ...,
        description="Stderr with ANSI escape codes stripped for simple rendering.",
    )


app = FastAPI(
    title="AI Researcher API",
    description=(
        "Thin HTTP wrapper around the existing CLI-based AI Researcher agents.\n\n"
        "The API does not reimplement any research logic; it simply shells out "
        "to `main.py` and returns everything the CLI prints so that a front-end "
        "can visualise it in rich ways."
    ),
    version="0.1.0",
)

from fastapi.middleware.cors import CORSMiddleware

# Optional, lightweight summarizer (kept outside agent logic)
from insights import summarize_agent_findings

# For Railway/production: allow all origins, or be more specific
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files in production
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

def _env_value_present(value: Optional[str]) -> bool:
    """Treat empty or placeholder values as missing."""
    if value is None:
        return False
    cleaned = value.strip()
    if not cleaned:
        return False

    lower = cleaned.lower()
    # Ignore common placeholder patterns (e.g., "your_google_api_key_here").
    if lower.startswith("your_") or lower.endswith("_here"):
        return False
    if lower in {"changeme", "example"}:
        return False
    return True


def _persist_env(key: str, value: str) -> None:
    """Persist a credential to both the running process and the .env file."""
    os.environ[key] = value
    set_key(str(ENV_PATH), key, value)


def _credential_status() -> CredentialStatus:
    """Summarize which credentials are available."""
    has_google = _env_value_present(os.environ.get("GOOGLE_API_KEY"))
    has_anthropic = _env_value_present(os.environ.get("ANTHROPIC_API_KEY"))
    has_modal_id = _env_value_present(os.environ.get("MODAL_TOKEN_ID"))
    has_modal_secret = _env_value_present(os.environ.get("MODAL_TOKEN_SECRET"))
    return CredentialStatus(
        has_google_api_key=has_google,
        has_anthropic_api_key=has_anthropic,
        has_modal_token=has_modal_id and has_modal_secret,
    )


def _ensure_main_exists() -> None:
    """
    Verify that main.py exists at the expected location.

    If not, raise an HTTPException so callers get a clear error.
    """
    if not MAIN_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"main.py not found at expected path: {MAIN_PATH}",
        )


def _build_single_command(req: SingleExperimentRequest) -> List[str]:
    """
    Build the command to run a single-agent experiment via main.py.
    """
    cmd: List[str] = [
        sys.executable,
        str(MAIN_PATH),
        req.task,
        "--mode",
        "single",
        "--model",
        req.model,
    ]
    if req.gpu:
        cmd.extend(["--gpu", req.gpu])
    if req.test_mode:
        cmd.append("--test-mode")
    return cmd


def _build_orchestrator_command(req: OrchestratorExperimentRequest) -> List[str]:
    """
    Build the command to run the orchestrator via main.py.
    """
    cmd: List[str] = [
        sys.executable,
        str(MAIN_PATH),
        req.task,
        "--mode",
        "orchestrator",
        "--model",
        req.model,
        "--num-agents",
        str(req.num_agents),
        "--max-rounds",
        str(req.max_rounds),
        "--max-parallel",
        str(req.max_parallel),
    ]
    if req.gpu:
        cmd.extend(["--gpu", req.gpu])
    if req.test_mode:
        cmd.append("--test-mode")
    return cmd


def _run_and_capture(
    cmd: List[str],
    *,
    mode: Literal["single", "orchestrator"],
    task: str,
    gpu: Optional[str],
) -> ProcessSummary:
    """
    Run `main.py` as a subprocess and capture all of its stdout/stderr.

    This leaves the underlying CLI behaviour untouched and simply wraps it.
    """
    _ensure_main_exists()

    started_at = datetime.now(timezone.utc)

    # Enable structured event emission in the child process so the frontend
    # can consume ::EVENT::-prefixed messages.
    env = dict(os.environ)
    env["AI_RESEARCHER_ENABLE_EVENTS"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []

    def _reader(stream, chunks: List[str]) -> None:
        for line in stream:
            chunks.append(line)

    t_out = threading.Thread(
        target=_reader, args=(proc.stdout, stdout_chunks), daemon=True
    )
    t_err = threading.Thread(
        target=_reader, args=(proc.stderr, stderr_chunks), daemon=True
    )

    t_out.start()
    t_err.start()

    exit_code = proc.wait()
    t_out.join()
    t_err.join()

    finished_at = datetime.now(timezone.utc)

    stdout_text = "".join(stdout_chunks)
    stderr_text = "".join(stderr_chunks)

    return ProcessSummary(
        mode=mode,
        task=task,
        gpu=gpu,
        command=cmd,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=(finished_at - started_at).total_seconds(),
        exit_code=exit_code,
        stdout=stdout_text,
        stderr=stderr_text,
        stdout_plain=strip_ansi(stdout_text),
        stderr_plain=strip_ansi(stderr_text),
    )


def _stream_subprocess(
    cmd: List[str],
    *,
    meta: Dict[str, Any],
    credentials: Optional[UserCredentials] = None,
) -> StreamingResponse:
    """
    Stream stdout/stderr from `main.py` as newline-delimited JSON (NDJSON).

    Each line of output is sent as:
        {
          "type": "line",
          "stream": "stdout" | "stderr",
          "timestamp": "<ISO-8601>",
          "raw": "<raw line>",
          "plain": "<line without ANSI>",
          ...meta
        }

    When the process finishes, a final summary event is sent:
        {
          "type": "summary",
          "exit_code": <int>,
          "started_at": "<ISO-8601>",
          "finished_at": "<ISO-8601>",
          "duration_seconds": <float>,
          ...meta
        }
    """
    _ensure_main_exists()

    started_at = datetime.now(timezone.utc)

    # Enable structured event emission in the child process so the frontend
    # can consume ::EVENT::-prefixed messages.
    env = dict(os.environ)
    env["AI_RESEARCHER_ENABLE_EVENTS"] = "1"

    # Override with user-provided credentials if present
    if credentials:
        if credentials.google_api_key:
            env["GOOGLE_API_KEY"] = credentials.google_api_key
            print(f"[DEBUG] Set GOOGLE_API_KEY from request (len={len(credentials.google_api_key)})", file=sys.stderr)
        if credentials.anthropic_api_key:
            env["ANTHROPIC_API_KEY"] = credentials.anthropic_api_key
            print(f"[DEBUG] Set ANTHROPIC_API_KEY from request (len={len(credentials.anthropic_api_key)})", file=sys.stderr)
        if credentials.modal_token_id:
            env["MODAL_TOKEN_ID"] = credentials.modal_token_id
            print(f"[DEBUG] Set MODAL_TOKEN_ID from request (len={len(credentials.modal_token_id)})", file=sys.stderr)
        if credentials.modal_token_secret:
            env["MODAL_TOKEN_SECRET"] = credentials.modal_token_secret
            print(f"[DEBUG] Set MODAL_TOKEN_SECRET from request (len={len(credentials.modal_token_secret)})", file=sys.stderr)
    else:
        print("[DEBUG] No credentials in request, using environment", file=sys.stderr)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    # Queue is used to multiplex stdout and stderr into a single ordered stream.
    q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()

    def _push_line(stream, stream_name: str) -> None:
        for line in stream:
            event: Dict[str, Any] = {
                "type": "line",
                "stream": stream_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw": line,
                "plain": strip_ansi(line),
            }
            event.update(meta)
            q.put(event)

    def _wait_for_exit() -> None:
        exit_code = proc.wait()
        finished_at = datetime.now(timezone.utc)
        summary: Dict[str, Any] = {
            "type": "summary",
            "timestamp": finished_at.isoformat(),
            "exit_code": exit_code,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": (finished_at - started_at).total_seconds(),
        }
        summary.update(meta)
        q.put(summary)
        # Sentinel to tell the iterator to stop.
        q.put(None)

    threading.Thread(
        target=_push_line, args=(proc.stdout, "stdout"), daemon=True
    ).start()
    threading.Thread(
        target=_push_line, args=(proc.stderr, "stderr"), daemon=True
    ).start()
    threading.Thread(target=_wait_for_exit, daemon=True).start()

    def event_iterator():
        while True:
            item = q.get()
            if item is None:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_iterator(),
        media_type="application/x-ndjson",
    )


@app.get("/api/health", summary="Simple health probe")
def health_check() -> Dict[str, Any]:
    """
    Lightweight health check.

    This does not call Gemini or Modal; it only verifies that `main.py`
    exists on disk and that the API process can see it.
    """
    exists = MAIN_PATH.exists()
    return {
        "status": "ok" if exists else "error",
        "main_py": str(MAIN_PATH),
        "main_py_exists": exists,
    }


@app.get("/api/state", summary="Get global system state")
def get_state() -> Dict[str, Any]:
    """
    Return the current global state of the orchestrator.
    This is primarily to support legacy clients or status monitors.
    """
    # Import here to avoid circular imports if orchestrator imports api_server
    try:
        from orchestrator import _experiment_counter, _default_gpu
        return {
            "status": "active",
            "experiments_run": _experiment_counter,
            "default_gpu": _default_gpu,
        }
    except ImportError:
        return {
            "status": "active",
            "info": "Orchestrator module not loaded",
        }


@app.get(
    "/api/credentials/status",
    response_model=CredentialStatus,
    summary="Check whether required API keys are set",
)
def credentials_status() -> CredentialStatus:
    """
    Return a minimal view of credential readiness.

    Used by the frontend to gate runs and prompt users for missing keys.
    """
    return _credential_status()


@app.post(
    "/api/credentials",
    response_model=CredentialStatus,
    summary="Set Google/Modal credentials (persists to .env)",
)
def update_credentials(req: CredentialUpdateRequest) -> CredentialStatus:
    """
    Allow the UI to persist credentials locally.

    Keys are written to the running process and the .env file so subsequent
    subprocesses inherit them.
    """
    try:
        if req.google_api_key and req.google_api_key.strip():
            _persist_env("GOOGLE_API_KEY", req.google_api_key.strip())
        if req.anthropic_api_key and req.anthropic_api_key.strip():
            _persist_env("ANTHROPIC_API_KEY", req.anthropic_api_key.strip())
        if req.modal_token_id and req.modal_token_id.strip():
            _persist_env("MODAL_TOKEN_ID", req.modal_token_id.strip())
        if req.modal_token_secret and req.modal_token_secret.strip():
            _persist_env("MODAL_TOKEN_SECRET", req.modal_token_secret.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist credentials: {e}") from e

    return _credential_status()


@app.post(
    "/api/experiments/single",
    response_model=ProcessSummary,
    summary="Run a single-agent experiment (blocking)",
)
def run_single_experiment(req: SingleExperimentRequest) -> ProcessSummary:
    """
    Run the original single-agent researcher and wait for it to finish.

    This endpoint blocks until the underlying CLI command exits, then returns
    a fully structured `ProcessSummary` containing all CLI output.
    """
    cmd = _build_single_command(req)
    return _run_and_capture(
        cmd,
        mode="single",
        task=req.task,
        gpu=req.gpu,
    )


@app.post(
    "/api/experiments/orchestrator",
    response_model=ProcessSummary,
    summary="Run the multi-agent orchestrator (blocking)",
)
def run_orchestrator_experiment(req: OrchestratorExperimentRequest) -> ProcessSummary:
    """
    Run the orchestrator mode end-to-end and wait for it to finish.

    The returned `ProcessSummary` includes all orchestrator logs, experiment
    transcripts, and the final paper generated at the end of the run.
    """
    cmd = _build_orchestrator_command(req)
    return _run_and_capture(
        cmd,
        mode="orchestrator",
        task=req.task,
        gpu=req.gpu,
    )


@app.post(
    "/api/experiments/single/stream",
    summary="Stream a single-agent experiment as newline-delimited JSON",
)
def stream_single_experiment(req: SingleExperimentRequest) -> StreamingResponse:
    """
    Run the single-agent researcher and stream all logs as NDJSON.

    This is ideal for front-ends that want to show real-time logs or
    progressively render the final report as it is produced.
    """
    cmd = _build_single_command(req)
    meta = {
        "mode": "single",
        "task": req.task,
        "gpu": req.gpu,
        "command": cmd,
    }
    return _stream_subprocess(cmd, meta=meta, credentials=req.credentials)


@app.post(
    "/api/experiments/orchestrator/stream",
    summary="Stream the orchestrator as newline-delimited JSON",
)
def stream_orchestrator_experiment(
    req: OrchestratorExperimentRequest,
) -> StreamingResponse:
    """
    Run the orchestrator mode and stream all logs as NDJSON.

    The stream includes orchestrator thinking, tool calls, and nested
    single-agent transcripts exactly as printed by the CLI.
    """
    cmd = _build_orchestrator_command(req)
    meta = {
        "mode": "orchestrator",
        "task": req.task,
        "gpu": req.gpu,
        "command": cmd,
    }
    return _stream_subprocess(cmd, meta=meta, credentials=req.credentials)


@app.post(
    "/api/agents/summarize",
    response_model=AgentSummaryResponse,
    summary="Summarize the last few sub-agent turns for the sidebar",
)
def summarize_agent(req: AgentSummaryRequest) -> AgentSummaryResponse:
    """Run a cheap Gemini call that condenses recent agent thoughts/tool outputs."""

    try:
        result = summarize_agent_findings(req.agent_id, [item.model_dump() for item in req.history])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return AgentSummaryResponse(**result)


@app.get("/api/files/list", summary="List files in a directory")
def list_files(path: str = ".") -> List[Dict[str, Any]]:
    """
    List files and directories in the given path relative to the project root.
    """
    # Security check: ensure path resolves to inside BASE_DIR
    # We strip any leading slashes/dots to prevent traversing up from root initially
    clean_path = path.strip("/")
    if clean_path == "." or clean_path == "":
        target_path = BASE_DIR
    else:
        target_path = (BASE_DIR / clean_path).resolve()

    if not str(target_path).startswith(str(BASE_DIR)):
         raise HTTPException(status_code=403, detail="Access denied: Cannot traverse outside project root.")

    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Path not found.")

    if not target_path.is_dir():
         raise HTTPException(status_code=400, detail="Path is not a directory.")

    results = []
    try:
        for item in target_path.iterdir():
            # Skip hidden files/dirs and .git, and __pycache__
            if item.name.startswith(".") or item.name == "__pycache__":
                continue

            stats = item.stat()
            results.append({
                "name": item.name,
                "path": str(item.relative_to(BASE_DIR)).replace("\\", "/"),
                "type": "directory" if item.is_dir() else "file",
                "size": stats.st_size,
                "modified": datetime.fromtimestamp(stats.st_mtime, timezone.utc).isoformat(),
            })
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied.")

    # Sort: directories first, then files
    results.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))
    return results


@app.get("/api/files/read", summary="Read a file content")
def read_file(path: str) -> Dict[str, str]:
    """
    Read the content of a text file.
    """
    clean_path = path.strip("/")
    target_path = (BASE_DIR / clean_path).resolve()

    if not str(target_path).startswith(str(BASE_DIR)):
         raise HTTPException(status_code=403, detail="Access denied.")

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        content = target_path.read_text(encoding="utf-8")
        return {"content": content}
    except UnicodeDecodeError:
         raise HTTPException(status_code=400, detail="Binary file not supported for text view.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files/download", summary="Download a file")
def download_file(path: str):
    """
    Download a single file.
    """
    clean_path = path.strip("/")
    target_path = (BASE_DIR / clean_path).resolve()

    if not str(target_path).startswith(str(BASE_DIR)):
         raise HTTPException(status_code=403, detail="Access denied.")

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(target_path, filename=target_path.name)


@app.get("/api/files/download_zip", summary="Download directory as zip")
def download_zip(path: str = "."):
    """
    Download a directory as a zip file.
    """
    clean_path = path.strip("/")
    if clean_path == "." or clean_path == "":
        target_path = BASE_DIR
    else:
        target_path = (BASE_DIR / clean_path).resolve()

    if not str(target_path).startswith(str(BASE_DIR)):
         raise HTTPException(status_code=403, detail="Access denied.")

    if not target_path.exists() or not target_path.is_dir():
         raise HTTPException(status_code=404, detail="Directory not found.")

    # Create a zip file in memory
    import io
    import zipfile

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(target_path):
            # exclude hidden folders
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "node_modules"]

            for file in files:
                if file.startswith("."):
                    continue
                file_path = Path(root) / file
                arcname = file_path.relative_to(target_path)
                zip_file.write(file_path, arcname)

    zip_buffer.seek(0)
    filename = target_path.name if (path != "." and path != "") else "project"
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}.zip"}
    )


# ---------------------------------------------------------------------------
# Static file serving for production (Railway, etc.)
# ---------------------------------------------------------------------------
# Mount static assets if the frontend dist folder exists
if FRONTEND_DIST.exists() and (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="static-assets")


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """
    Catch-all route to serve the frontend SPA.
    API routes are defined above, so they take precedence.
    """
    # If frontend dist exists, serve it
    if FRONTEND_DIST.exists():
        # Check if the requested file exists
        file_path = FRONTEND_DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        # Otherwise serve index.html for SPA routing
        index_path = FRONTEND_DIST / "index.html"
        if index_path.exists():
            return FileResponse(index_path)

    # Fallback: return a simple message if no frontend
    return {"message": "AI Researcher API is running. Frontend not built.", "docs": "/docs"}


if __name__ == "__main__":
    # Convenience entrypoint so you can run:
    #   python api_server.py
    # during development instead of calling uvicorn manually.
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    reload_enabled = os.environ.get("RAILWAY_ENVIRONMENT") is None  # Disable reload in production

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=reload_enabled,
    )
