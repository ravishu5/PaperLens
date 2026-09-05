"""Generate output/README.md: what each paper's run produced, and how good it is."""
from __future__ import annotations
import json, pathlib

OUT = pathlib.Path("output")
NAMES = {"01-VNet":"V-Net","02-SegResNet":"SegResNet","03-nnUNet":"nnU-Net",
         "04-TransBTS":"TransBTS","05-UNETR":"UNETR","06-nnFormer":"nnFormer",
         "07-SwinUNETR":"Swin UNETR","08-RepUX-Net":"RepUX-Net","09-UNesT":"UNesT",
         "10-deformUX-Net":"DeformUX-Net",
         "11-arXiv-2404.13024-BANF":"BANF (supplied as ULD-Net)"}


def j(d, n, default=None):
    f = d / f"{n}.json"
    return json.load(open(f)) if f.exists() else (default if default is not None else {})


def rows():
    for d in sorted(OUT.glob("[0-9][0-9]-*")):
        if d.name == "11-ULD-Net":
            continue
        ing, impl = j(d, "02_ingest_paper"), j(d, "08_find_implementations")
        cands = impl.get("candidates") or []
        top = cands[0] if cands else {}
        yield d, {
            "name": NAMES.get(d.name, d.name), "arxiv": ing.get("arxiv_id", "?"),
            "title": ing.get("title", ""), "fid": ing.get("fidelity", "-"),
            "sec": ing.get("sections", 0), "eq": ing.get("equations", 0),
            "val": ing.get("stated_values", 0),
            "refs": j(d, "07_resource_references").get("count", 0),
            "repo": top.get("repo", "—"), "rel": top.get("relation", "—"),
            "conf": top.get("confidence", "—"),
            "map": j(d, "10_map_paper_to_code").get("summary") or {},
            "diff": j(d, "11_compare_paper_with_code").get("summary") or {},
            "gaps": j(d, "16_find_implementation_gaps").get("summary") or {},
            "lin": j(d, "12_trace_research_lineage").get("counts") or {},
            "plan": j(d, "19_build_reproduction_plan").get("readiness") or {},
            "analysis": len((j(d, "22_record_paper_analysis").get("accepted") or {}).get("components", [])),
            "files": sorted(p.name for p in d.glob("*.json")),
        }


def main() -> None:
    data = list(rows())
    L = []
    L.append("# PaperLens run — 11 volumetric segmentation architectures\n")
    L.append("One folder per paper. Every file is a tool's raw JSON output, named for\n"
             "the tool that produced it, in the order the pipeline runs them.\n")
    L.append(f"Papers processed: **{len(data)}**.\n")
    L.append("> **On entry 11.** The identifier supplied for *ULD-Net* — "
             "`2404.13024` — resolves to *BANF: Band-limited Neural Fields for "
             "Levels of Detail Reconstruction*, a neural-fields paper, not a "
             "volumetric segmentation architecture. It was run and is filed "
             "under its real title rather than the label it arrived with. "
             "`ULD-Net` itself remains unidentified: arXiv full-text search "
             "returns nothing for the name. See "
             "[11-ULD-Net](11-ULD-Net/00_UNRESOLVED.md).\n")

    L.append("\n## Resolution\n")
    L.append("| # | Architecture | arXiv | Title |")
    L.append("|---|---|---|---|")
    for i, (d, r) in enumerate(data, 1):
        L.append(f"| {i} | {r['name']} | [`{r['arxiv']}`](https://arxiv.org/abs/{r['arxiv']}) | {r['title'][:70]} |")
    L.append("| — | ULD-Net | — | **not identified** — see [11-ULD-Net](11-ULD-Net/00_UNRESOLVED.md) |")

    L.append("\n## What was extracted\n")
    L.append("| Architecture | Fidelity | Sections | Equations | Values | References | Components recorded |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for d, r in data:
        L.append(f"| {r['name']} | `{r['fid']}` | {r['sec']} | {r['eq']} | {r['val']} | {r['refs']} | {r['analysis']} |")

    L.append("\n## Implementation found\n")
    L.append("| Architecture | Top candidate | Relation | Confidence |")
    L.append("|---|---|---|---|")
    for d, r in data:
        L.append(f"| {r['name']} | `{r['repo']}` | {r['rel']} | {r['conf']} |")

    L.append("\n## Correlation and comparison\n")
    L.append("| Architecture | Matched | Absent | Ambiguous | Unknown | Blocking | Significant | Gaps | Predecessors |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d, r in data:
        m, dd, g, l = r["map"], r["diff"], r["gaps"], r["lin"]
        L.append(f"| {r['name']} | {m.get('MATCHED',0)} | {m.get('ABSENT',0)} | "
                 f"{m.get('AMBIGUOUS',0)} | {m.get('UNKNOWN',0)} | {dd.get('BLOCKING',0)} | "
                 f"{dd.get('SIGNIFICANT',0)} | {sum(g.values())} | {l.get('predecessors',0)} |")

    L.append("\n## Files in each folder\n")
    L.append("```")
    for n in (data[0][1]["files"] if data else []):
        L.append(n)
    L.append("```")

    L.append("""
## How to read these

**Confidence is not decoration.** `CONFIRMED` means machine-checkable — an exact
numeric literal found in source, a URL the authors printed, or an established
absence. `LIKELY` and `POSSIBLE` mean corroborating signals agreed. `UNKNOWN`
means the evidence did not reach, and it is a normal outcome rather than a
failure.

**An absent component is a finding, not an error.** `map_paper_to_code` reports
`ABSENT` only when the component has a distinctive name that occurs nowhere in
the repository's source.

**Where no official implementation was established**, the paper declared no
repository URL and the candidates come from searching GitHub. Those runs say so
in their notes, and their comparisons should be read as "against this
third-party repository", not "against the paper's own code".
""")
    (OUT / "README.md").write_text("\n".join(L) + "\n")
    print(f"wrote output/README.md ({len(data)} papers)")


if __name__ == "__main__":
    main()
