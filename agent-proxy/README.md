# agent-proxy: the sleep loop at the memory layer

This is the negative half of [the project](../README.md). It implements consolidation, dreaming, and pruning as *system behaviour* over a two-tier memory store, with a hosted LLM and no gradients.

It is not an implementation of the paper's mechanism. The paper updates weights through custom low-rank experts, GKD, and RL. Two of its three stages happen to be pure orchestration, which is what this directory reproduces, and the measured answer is that a good retriever plus versioned facts beats the loop on recall.

| condition | all | direct | derived | ctx tokens/query |
|---|---|---|---|---|
| `no_memory` | 0% | 0% | 0% | 0 |
| `transcript_rag` | 95.5% | 100% | 83.3% | 227 |
| `chunk_rag` | 100% | 100% | 100% | 77 |
| `sleep` | 90.9% | 100% | 66.7% | 361 |

Where it does win is interference: when facts are revised, the arbiter retires superseded nodes.

| condition | interference accuracy |
|---|---|
| `chunk_rag` | 85.7% |
| `sleep` | 100% |

## Setup

```bash
pip install -r requirements.txt
export SLEEP_BASE_URL=https://your-endpoint/v1
export SLEEP_API_KEY=...
export SLEEP_MODEL=your-model
```

Any OpenAI-compatible chat endpoint works. If you run Hermes, the factory also falls back to `~/.hermes/config.yaml` so no separate key setup is needed. Credentials are never printed.

## Run

```bash
python demo.py                     # three sessions, one sleep, then recall
python evals/eval_retention.py     # four conditions, writes evals/results.json
python evals/eval_updates.py       # revised facts, writes evals/results_updates.json
python evals/export_sft.py         # memory -> SFT JSONL, feeds slm-test
```

## Files

- `sleep_agent/memory.py` — two-tier store, links, supersession, kind-aware retrieval
- `sleep_agent/phases.py` — consolidate, dream, prune, and the arbiter
- `sleep_agent/framework.py` — the wake/sleep orchestration
- `sleep_agent/llm.py` — OpenAI-compatible factory with JSON extraction
- `data/probe_corpus.json` — the "Vantar Labs" facts used for retention
- `data/update_stream.json` — sequential fact revisions used for interference

Built By [Harish Kotra](https://harishkotra.me) · [Checkout my other builds](https://dailybuild.xyz)
