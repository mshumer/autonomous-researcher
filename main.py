import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from agent import run_experiment_loop
from logger import print_status, set_log_file


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Debug: show which credentials are available
    import sys
    google_key = os.environ.get("GOOGLE_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    modal_id = os.environ.get("MODAL_TOKEN_ID", "")
    modal_secret = os.environ.get("MODAL_TOKEN_SECRET", "")
    print(f"[DEBUG] Credentials check: GOOGLE_API_KEY={'set' if google_key else 'missing'} (len={len(google_key)}), "
          f"ANTHROPIC_API_KEY={'set' if anthropic_key else 'missing'}, "
          f"MODAL_TOKEN={'set' if modal_id and modal_secret else 'missing'}", file=sys.stderr)

    parser = argparse.ArgumentParser(
        description="AI Experiment Agent CLI (single-agent and orchestrator modes)"
    )
    parser.add_argument(
        "task",
        type=str,
        help=(
            "In 'single' mode: the hypothesis to verify.\n"
            "In 'orchestrator' mode: the high-level research task to investigate."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory to store run artifacts (agent.log, final paper, etc.).",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU type to request (e.g., 'T4', 'A10G', 'A100', 'any').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "orchestrator"],
        default="single",
        help=(
            "Execution mode: "
            "'single' runs a single-researcher agent (original behavior); "
            "'orchestrator' runs the higher-level multi-agent orchestrator."
        ),
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=3,
        help="(orchestrator) Number of initial single-researcher agents to launch.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="(orchestrator) Maximum number of orchestration rounds.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help=(
            "(orchestrator) Maximum number of experiments to run in parallel "
            "in a single wave of tool calls."
        ),
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with mock data (no LLM/GPU usage).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gemini-3-pro-preview", "claude-opus-4-5"],
        default="gemini-3-pro-preview",
        help=(
            "LLM model to use: "
            "'gemini-3-pro-preview' (default) or 'claude-opus-4-5'."
        ),
    )

    args = parser.parse_args()

    # Determine/Create Run Directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        # If passed by parent process, assume created. If not, create it.
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Generate new run directory for top-level process
        task_slug = "".join(x for x in args.task if x.isalnum() or x in " -_").strip().replace(" ", "_")[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Change naming convention to <task>_<timestamp> as requested
        run_dir = Path("runs") / f"{task_slug}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    # Configure Logger to write to this run directory
    set_log_file(run_dir / "agent.log")

    # Save metadata for top-level runs
    if not args.run_dir:
        try:
            with open(run_dir / "metadata.json", "w") as f:
                json.dump(vars(args), f, indent=2, default=str)
        except Exception as e:
            print(f"[WARNING] Failed to save metadata: {e}", file=sys.stderr)

    # Preserve existing behavior by default: single agent mode.
    if args.mode == "single":
        print_status("Initializing Single Researcher Agent...", "bold cyan")

        try:
            # Record GPU preference globally for sandbox creation
            import agent as agent_module

            agent_module._selected_gpu = args.gpu
            run_experiment_loop(args.task, test_mode=args.test_mode, model=args.model)
        except KeyboardInterrupt:
            print_status("\nExperiment interrupted by user.", "bold red")
            sys.exit(0)
        except Exception as e:
            import traceback
            print_status(f"\nFatal Error: {e}", "bold red")
            print(f"[ERROR] Fatal Error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
    else:
        # Multi-agent orchestrator mode.
        print_status("Initializing Orchestrator Agent...", "bold cyan")

        try:
            from orchestrator import run_orchestrator_loop

            run_orchestrator_loop(
                research_task=args.task,
                num_initial_agents=args.num_agents,
                max_rounds=args.max_rounds,
                default_gpu=args.gpu,
                max_parallel_experiments=args.max_parallel,
                test_mode=args.test_mode,
                model=args.model,
                run_dir=run_dir,
            )
        except KeyboardInterrupt:
            print_status("\nOrchestrated experiment interrupted by user.", "bold red")
            sys.exit(0)
        except Exception as e:
            import traceback
            print_status(f"\nFatal Error (orchestrator mode): {e}", "bold red")
            print(f"[ERROR] Fatal Error (orchestrator): {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
