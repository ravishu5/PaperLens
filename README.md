# PaperLens

An MCP server for **research engineering**: going from a paper to a working understanding of how its ideas became real code.

Not a search wrapper. PaperLens does the deterministic work that language models do badly — flattening LaTeX, replaying equation counters, mining implementation URLs, indexing repositories, matching concrete anchors — and hands the calling agent precisely-scoped evidence with stable addresses. The agent does the language work, then writes its conclusions back into a graph that validates them.
**Status: Phase 7 of 8.** Ingestion, implementation discovery, code intelligence, paper↔code mapping, comparison and research lineage all work end to end. Reproduction planning is not built yet.

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

GitHub access is optional but strongly recommended: PaperLens uses `GITHUB_TOKEN`/`GH_TOKEN` if set, otherwise the `gh` CLI's token, otherwise anonymous access at 60 requests/hour.

An optional `S2_API_KEY` improves forward lineage: Semantic Scholar supplies the sentences in which later papers cite this one, and unauthenticated access shares one heavily-contended pool. Without it, PaperLens falls back to OpenAlex for structure and says that citing sentences are unavailable.

Code intelligence uses [jcodemunch](https://github.com/jgravelle/jcodemunch-mcp) when it is reachable (70+ languages) and falls back to a built-in Python AST backend otherwise. Force one with `PAPERLENS_CODE_PROVIDER=python-ast|jcodemunch`.

## Use

```
resolve_paper("Learning Transferable Visual Models…")   → candidates + arXiv ids
ingest_paper("2103.00020")                              → structure counts + URIs
get_paper_skeleton("2103.00020")                        → every addressable anchor
find_implementations("2103.00020")                      → ranked repos + evidence
index_repository("openai/CLIP")                         → commit SHA + symbol counts
get_repo_outline("openai/CLIP")                         → files and symbols as URIs
search_code("openai/CLIP", "contrastive loss")          → symbols, or a citable absence
record_paper_analysis("2103.00020", {...})              → validated write-back
map_paper_to_code("2103.00020", "openai/CLIP")          → anchors ↔ symbols, with evidence
reverse_engineer_paper("2103.00020")                    → the assembled reconstruction
compare_paper_with_code("2103.00020", "openai/CLIP")    → differences, BLOCKING first
find_implementation_gaps("2103.00020", "openai/CLIP")   → what would block a reproduction
compare_implementations("2103.00020", [repo_a, repo_b]) → which implements more
trace_research_lineage("2103.00020")                    → predecessors and successors
trace_method("2103.00020", "temperature")               → one method through the literature
find_sota_successors("2103.00020")                      → later work, with citing sentences
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
paperlens://paper/2103.00020/implementations
paperlens://repo/openai/CLIP
paperlens://repo/openai/CLIP/file/clip/model.py
paperlens://repo/openai/CLIP/symbol/{percent-encoded symbol id}
paperlens://mapping/2103.00020/openai/CLIP
paperlens://difference/2103.00020/openai/CLIP
paperlens://paper/2103.00020/references
paperlens://lineage/2103.00020
```

`paperlens_fetch(uri)` returns identical content, for clients without resource support.

## Six things worth knowing

**LaTeX, not PDF.** arXiv e-print source preserves section structure, exact math and the authors' own `\url{}` links. PDF extraction destroys all three. Papers without source are marked `UNAVAILABLE` rather than silently degraded.

**Equations are addressed by content hash, not number.** Equation numbers do not exist in LaTeX source — they are assigned by a counter at compile time, and authors almost never `\label` them. *Attention Is All You Need* numbers three equations and labels none; its only `\label{eq:attention}` is commented out. PaperLens replays the counter, so `…/equation/1` works, but the number carries its own confidence and is never better than `LIKELY` without checking the compiled PDF.

**The mapping is the point.** `map_paper_to_code` joins paper anchors to code symbols with evidence on both sides. An exact constant match can reach `CONFIRMED` — the paper states τ = 0.07, and the identical literal sits at `clip/model.py:295` inside `CLIP.__init__`. A name resemblance cannot: `Image Encoder → CLIP.encode_image` is `POSSIBLE`. And the contrastive objective comes back `ABSENT` with a citable reference, because `openai/CLIP` does not contain it.

**Lineage comes from the paper itself, where it can.** Every arXiv submission ships its own bibliography, so predecessors are extracted from the paper's `.bbl` and the sentence that cites them — offline, never rate limited, and quotable. CLIP's τ traces to arXiv:1805.01978 with the exact sentence that says so. Successors need a citation index and are marked as the weaker evidence they are.

**Differences quote both sides.** `compare_paper_with_code` pairs an exact quote from the paper with an exact search over source. CLIP's paper calls the temperature clamp "necessary to prevent training instability"; `openai/CLIP` contains no clamp, so that surfaces as `BLOCKING`. The inference joining a paper term to a code term is a heuristic, so such differences are `LIKELY`, never `CONFIRMED`.

**Official is not the same as complete.** `find_implementations` ranks by evidential strength and reports content coverage for every candidate. `openai/CLIP` comes back first — `OFFICIAL`, `CONFIRMED`, cited to the URL in the paper's own abstract — with `missing: [dataset, training, inference]` on the face of the result, because the official repository ships inference weights and not the contrastive objective the paper is about.

**`UNKNOWN` is a normal answer, and absence is a finding.** A tool that cannot say "I could not establish this" will invent mappings. Asking CLIP for an equation returns an error explaining it has none. Searching `openai/CLIP` for its own contrastive loss returns `found: false` with a citable `absence_ref` and `confidence: CONFIRMED` — because establishing that something is missing is a result, not a failure.

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

