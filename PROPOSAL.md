# Artifact Management Proposal

## Current State
- **Logs:** `agent.log` is created in the root directory.
- **Final Paper:** Printed to stdout and emitted as an event; not saved to disk.
- **Transcripts:** Logged to `agent.log`; not saved individually.
- **Sandbox Artifacts:** Remain in the Modal sandbox unless printed.

## Proposed Solution

To organize artifacts better and prevent overwriting, we propose creating a structured directory for each run.

### Directory Structure
```
runs/
  └── <YYYYMMDD_HHMMSS>_<sanitized_task_name>/
      ├── agent.log              # Full execution log for this run
      ├── final_paper.md         # The generated research paper
      ├── metadata.json          # Run details (task, model, params)
      └── agents/
          ├── agent_1.txt        # Transcript for Agent 1
          ├── agent_2.txt        # Transcript for Agent 2
          └── ...
```

### Implementation Steps

1.  **Initialize Run Directory:**
    *   In `main.py` (or `orchestrator.py` / `agent.py`), before starting the experiment:
        *   Generate a timestamp (e.g., `20231027_103000`).
        *   Sanitize the task name (take first 30 chars, replace non-alphanumeric with `_`).
        *   Create the directory `runs/<timestamp>_<task_slug>/`.

2.  **Update Logging:**
    *   Modify `logger.py` to accept a `log_file_path` argument in `setup_logging`.
    *   Initialize the logger to write to the new run directory's `agent.log`.

3.  **Save Final Paper:**
    *   In `orchestrator.py`, after the final paper is generated:
        *   Write the content to `runs/.../final_paper.md`.

4.  **Save Transcripts:**
    *   In `orchestrator.py`, when a sub-agent completes:
        *   Write its transcript to `runs/.../agents/agent_<id>.txt`.

5.  **Metadata:**
    *   Save run arguments (task, model, gpu, etc.) to `runs/.../metadata.json` for reproducibility.
