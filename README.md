# PaperLens

An MCP server for **research engineering**: going from a paper to a working understanding of how its ideas became real code.

Not a search wrapper. PaperLens does the deterministic work that language models do badly — flattening LaTeX, replaying equation counters, mining implementation URLs, indexing repositories, matching concrete anchors — and hands the calling agent precisely-scoped evidence with stable addresses. The agent does the language work, then writes its conclusions back into a graph that validates them.

**Status: Phase 2 of 8.** Paper ingestion works end to end. Implementation discovery, code intelligence and correlation are not built yet. See [ARCHITECTURE.md](ARCHITECTURE.md) §5 for the plan and [RESEARCH.md](RESEARCH.md) for why it is designed this way.

---

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
```

Register with an MCP client (Claude Code / Claude Desktop):

```json
{
  "mcpServers": {
    "paperlens": {
      "command": "/absolute/path/to/PaperLens/.venv/bin/python",
      "args": ["-m", "paperlens.server"]
    }
  }
}
```

State lives in `~/.paperlens/` (`graph.db` plus cached arXiv sources). Override with `PAPERLENS_HOME`.

## Use

```
resolve_paper("Learning Transferable Visual Models…")   → candidates + arXiv ids
ingest_paper("2103.00020")                              → structure counts + URIs
get_paper_skeleton("2103.00020")                        → every addressable anchor
```

Then read only what you need:

```
paperlens://paper/2103.00020
paperlens://paper/2103.00020/section/2.3
paperlens://paper/2103.00020/equation/{content_hash}
paperlens://paper/2103.00020/component/{slug}
paperlens://paper/2103.00020/algorithm/{slug}
paperlens://paper/2103.00020/values
paperlens://paper/2103.00020/urls
```

`paperlens_fetch(uri)` returns identical content, for clients without resource support.

## Three things worth knowing

**LaTeX, not PDF.** arXiv e-print source preserves section structure, exact math and the authors' own `\url{}` links. PDF extraction destroys all three. Papers without source are marked `UNAVAILABLE` rather than silently degraded.

**Equations are addressed by content hash, not number.** Equation numbers do not exist in LaTeX source — they are assigned by a counter at compile time, and authors almost never `\label` them. *Attention Is All You Need* numbers three equations and labels none; its only `\label{eq:attention}` is commented out. PaperLens replays the counter, so `…/equation/1` works, but the number carries its own confidence and is never better than `LIKELY` without checking the compiled PDF.

**`UNKNOWN` is a normal answer.** A tool that cannot say "I could not establish this" will invent mappings. Asking CLIP for an equation returns an error explaining it has none — not a guess.

## Develop

```bash
.venv/bin/python -m pytest tests -q          # unit tests, no network
.venv/bin/python -m pytest tests -q -m network   # against real arXiv papers
```

Test corpus ground truth was verified by hand in Phase 0:

| Paper | Why it is in the corpus |
|---|---|
| `1706.03762` Transformer | Three numbered equations, zero labels, nine `\input` files — the equation-identity case. |
| `2103.00020` CLIP | Zero equations; core algorithm is a figure image; official repo lacks the paper's central method — the adversarial case. |

## Docs

- [RESEARCH.md](RESEARCH.md) — what already exists, what died with Papers With Code, and what is genuinely new
- [ARCHITECTURE.md](ARCHITECTURE.md) — decisions, schema, tool surface, build plan
- [DECISIONS.md](DECISIONS.md) — what was chosen, what was rejected, why
