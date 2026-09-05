"""Run every PaperLens tool over each paper and save the outputs."""
from __future__ import annotations

import json, os, sys, time, traceback
from pathlib import Path

from paperlens import config as _cfg
_cfg.quiet_dependencies()

from paperlens.graph.store import Store
from paperlens.paper.ingest import ingest_paper
from paperlens.resources import resolve, paper_uri, encode_paper, ResourceNotFound
from paperlens.correlate.discover import find_implementations
from paperlens.correlate.mapper import map_paper_to_code
from paperlens.correlate.compare import (compare_paper_with_code,
                                         find_implementation_gaps,
                                         compare_implementations)
from paperlens.correlate.lineage import (trace_research_lineage,
                                         find_sota_successors, trace_method)
from paperlens.correlate.plan import build_reproduction_plan
from paperlens.code.indexer import index_repository
from paperlens.sources import arxiv

PAPERS = [
    ("01-VNet",         "1606.04797", "convolution"),
    ("02-SegResNet",    "1810.11654", "autoencoder"),
    ("03-nnUNet",       "1809.10486", "augmentation"),
    ("04-TransBTS",     "2103.04430", "transformer"),
    ("05-UNETR",        "2103.10504", "transformer"),
    ("06-nnFormer",     "2109.03201", "attention"),
    ("07-SwinUNETR",    "2201.01266", "swin"),
    ("08-RepUX-Net",    "2303.05785", "kernel"),
    ("09-UNesT",        "2209.14378", "transformer"),
    ("10-deformUX-Net", "2310.00199", "deformable"),
    # Supplied as "ULD-Net", but this identifier resolves to BANF. Named for
    # what it is rather than what it was labelled.
    ("11-arXiv-2404.13024-BANF", "2404.13024", "band-limited"),
    # Published in Biomedical Signal Processing and Control, not on arXiv.
    ("12-ULD-Net", "10.1016/j.bspc.2025.108746", "depthwise"),
]

OUT = Path("output")
MAX_REPO_FILES = 4000


def save(folder: Path, name: str, data) -> None:
    (folder / f"{name}.json").write_text(json.dumps(data, indent=2, default=str))


def step(folder: Path, name: str, fn):
    """Run one tool, save its result, and never let a failure abort the run."""
    t0 = time.time()
    try:
        out = fn()
    except Exception as exc:
        out = {"error": type(exc).__name__, "message": str(exc),
               "traceback": traceback.format_exc()[-1500:]}
        print(f"      ! {name}: {type(exc).__name__}: {str(exc)[:90]}")
    save(folder, name, out)
    print(f"      · {name} ({time.time() - t0:.1f}s)")
    return out


def resource(folder: Path, name: str, store, uri: str):
    return step(folder, name, lambda: resolve(store, uri))


def run(slug: str, arxiv_id: str, method_term: str) -> dict:
    folder = OUT / slug
    folder.mkdir(parents=True, exist_ok=True)
    store = Store()
    summary = {"paper": slug, "arxiv_id": arxiv_id, "steps": {}}
    print(f"\n=== {slug}  ({arxiv_id})")

    def _resolve():
        if arxiv.parse_arxiv_id(arxiv_id):
            m = arxiv.fetch_metadata(store, arxiv_id)
            return {"candidates": [{
                "arxiv_id": m.arxiv_id, "title": m.title,
                "authors": [a["name"] for a in m.authors[:8]],
                "published_at": m.published_at, "confidence": "CONFIRMED",
                "why": "explicit arXiv identifier"}]}
        from paperlens.sources import crossref
        w = crossref.work(store, crossref.parse_doi(arxiv_id))
        return {"candidates": [{
            "doi": w.doi, "title": w.title, "venue": w.venue, "year": w.year,
            "authors": [a["name"] for a in w.authors[:8]],
            "confidence": "CONFIRMED", "why": "explicit DOI"}]}

    step(folder, "01_resolve_paper", _resolve)

    ing = step(folder, "02_ingest_paper", lambda: _asdict(ingest_paper(store, arxiv_id)))
    summary["fidelity"] = ing.get("fidelity")
    if ing.get("error"):
        summary["steps"]["ingest"] = "FAILED"
        return summary

    skel = step(folder, "03_get_paper_skeleton", lambda: _skeleton(store, arxiv_id))
    resource(folder, "04_resource_paper", store, paper_uri(arxiv_id))
    resource(folder, "05_resource_values", store, paper_uri(arxiv_id, "values"))
    resource(folder, "06_resource_urls", store, paper_uri(arxiv_id, "urls"))
    resource(folder, "07_resource_references", store,
             paper_uri(arxiv_id, "references"))

    impl = step(folder, "08_find_implementations",
                lambda: find_implementations(store, arxiv_id))
    # Prefer an established implementation. A candidate whose only link to the
    # paper is a matching name is not one: comparing against it manufactures
    # absences that say nothing about the paper.
    repo = None
    usable = [c for c in (impl.get("candidates") or [])
              if not c.get("name_is_only_evidence")
              and c.get("relation") not in ("DERIVED", "UNRELATED")]
    for cand in usable:
        if cand["confidence"] in ("CONFIRMED", "LIKELY"):
            repo = cand["repo"]
            break
    if repo is None and usable:
        repo = usable[0]["repo"]
    summary["repo"] = repo

    lineage = step(folder, "12_trace_research_lineage",
                   lambda: trace_research_lineage(store, arxiv_id, limit=15))
    step(folder, "13_find_sota_successors",
         lambda: find_sota_successors(store, arxiv_id, limit=12))
    step(folder, "14_trace_method",
         lambda: trace_method(store, arxiv_id, method_term))
    resource(folder, "15_resource_lineage", store, f"paperlens://lineage/{encode_paper(arxiv_id)}")

    if repo:
        idx = step(folder, "09_index_repository",
                   lambda: _asdict(index_repository(store, repo)))
        if not idx.get("error"):
            step(folder, "10_map_paper_to_code",
                 lambda: map_paper_to_code(store, arxiv_id, repo))
            step(folder, "11_compare_paper_with_code",
                 lambda: compare_paper_with_code(store, arxiv_id, repo))
            step(folder, "16_find_implementation_gaps",
                 lambda: find_implementation_gaps(store, arxiv_id, repo))
            resource(folder, "17_resource_mapping", store,
                     f"paperlens://mapping/{encode_paper(arxiv_id)}/{repo}")
            resource(folder, "18_resource_differences", store,
                     f"paperlens://difference/{encode_paper(arxiv_id)}/{repo}")
            step(folder, "19_build_reproduction_plan",
                 lambda: build_reproduction_plan(store, arxiv_id, repo))
            others = [c["repo"] for c in (impl.get("candidates") or [])[:3]
                      if c["repo"] != repo][:1]
            if others:
                def _cmp():
                    for o in others:
                        index_repository(store, o)
                    return compare_implementations(store, arxiv_id, [repo] + others)
                step(folder, "20_compare_implementations", _cmp)
    else:
        save(folder, "09_index_repository", {
            "skipped": "No implementation candidate was established. Candidates "
                       "whose only link to the paper is a matching repository "
                       "name are not compared against, because that manufactures "
                       "absences which say nothing about the paper.",
            "rejected_candidates": [
                {"repo": c["repo"], "relation": c["relation"],
                 "confidence": c["confidence"],
                 "name_is_only_evidence": c.get("name_is_only_evidence")}
                for c in (impl.get("candidates") or [])],
        })

    step(folder, "21_reverse_engineer_paper", lambda: _reverse(store, arxiv_id))
    summary["counts"] = {k: v for k, v in ing.items()
                         if k in ("sections", "equations", "stated_values",
                                  "declared_urls", "components", "algorithms")}
    save(folder, "00_summary", summary)
    return summary


def _asdict(obj):
    import dataclasses
    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj


def _skeleton(store, arxiv_id):
    from paperlens.server import get_paper_skeleton, _store
    import paperlens.server as srv
    srv._store = store
    return get_paper_skeleton(arxiv_id)


def _reverse(store, arxiv_id):
    import paperlens.server as srv
    srv._store = store
    return srv.reverse_engineer_paper(arxiv_id)


if __name__ == "__main__":
    only = sys.argv[1:] or None
    OUT.mkdir(exist_ok=True)
    results = []
    for slug, aid, term in PAPERS:
        if only and not any(o in slug for o in only):
            continue
        try:
            results.append(run(slug, aid, term))
        except Exception:
            print(f"  !! {slug} aborted\n{traceback.format_exc()[-1200:]}")
            results.append({"paper": slug, "arxiv_id": aid, "aborted": True})
    (OUT / "_batch_summary.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nDONE")
