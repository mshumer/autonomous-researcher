# AI Researcher
[![Twitter Follow](https://img.shields.io/twitter/follow/mattshumer_?style=social)](https://twitter.com/mattshumer_)

[Be the first to know when I publish new AI builds + demos!](https://tally.so/r/w2M17p)

An autonomous AI researcher. It takes a research objective, breaks it into experiments, spins up separate agents with access to their own GPUs to run these experiments, and delivers a paper-style writeup with findings.

## How it Works
- Decomposes your prompt into experiments and assigns them to specialist researcher agents.
- Each agent can launch GPU-enabled sandboxes to train models/run inference/etc., evaluate, and collect evidence.
- Based on the results of these experiments, the orchestrator can decide to finalize, or run more experiments.
- The orchestrator goes over all of the results and turns them into a coherent "paper".

## Run it (web notebook, one command)
The fastest way to use it:
```
python run_app.py
```
This installs missing deps, starts the API + frontend, and opens the notebook. If Google/Modal keys aren’t set, the UI will prompt you and save them locally before the run starts.

## Keys Needed
- **LLM key** (at least one):
  - Google AI Studio: `GOOGLE_API_KEY` (for Gemini 3 Pro)
  - Anthropic: `ANTHROPIC_API_KEY` (for Claude Opus 4.5)
- **Modal tokens**: `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (for GPU sandboxes)
- Add them to `.env` in the repo root, or paste them into the web prompt when asked.

## Model Selection
Choose between **Gemini 3 Pro** and **Claude Opus 4.5** from the dropdown in the web UI, or via CLI with `--model`.

## Optional CLI
Prefer the terminal?
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py "Does label smoothing improve ViT-Base on CIFAR-10?" --mode single --gpu any --model gemini-3-pro-preview
```
Orchestrator (multi-agent):
```
python main.py "Characterize scaling laws for sparse attention transformers" \
  --mode orchestrator --num-agents 3 --max-rounds 3 --max-parallel 2 --gpu any
```
Dry run:
```
python main.py "Sanity check the pipeline" --mode orchestrator --test-mode
```

## Deploy to Railway
One-click deploy to Railway:

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/new?template=https://github.com/mattshumer/ai-researcher)

Or manually:
1. Fork this repo
2. Create a new Railway project from GitHub
3. Add environment variables: `GOOGLE_API_KEY` or `ANTHROPIC_API_KEY`, plus `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`
4. Deploy — Railway will build the frontend and start the API automatically

## Status/Contribution
This is a super-early, experimental harness. There are a number of improvements to be worked out (i.e. dataset sharing between agents, key management, etc.), literature search, that would make this way more capable. If anyone wants to add these in, feel free!