# ULD-Net — not resolved

> **Update.** An identifier was supplied for this entry: `arXiv:2404.13024`.
> That resolves to **"BANF: Band-limited Neural Fields for Levels of Detail
> Reconstruction"** (Shabanov et al., 2024-04-19) — a neural-fields paper about
> filtering, not a volumetric segmentation architecture. It was run in full and
> filed under its own title at
> [`11-arXiv-2404.13024-BANF`](../11-arXiv-2404.13024-BANF/), rather than being
> labelled ULD-Net.
>
> Checked alongside it: arXiv full-text search for `"ULD-Net"` returns **zero**
> results, and the adjacent identifiers `2404.13023`, `2404.13025`, `2404.03024`,
> `2404.13204` and `2402.13024` are all unrelated papers (nuclear clocks,
> gravitational waves, effect modelling, neuroimaging regression, smart
> environments). So this is not a near-miss transcription of a neighbouring ID.
>
> **ULD-Net itself remains unidentified.**

**No PaperLens tools were run for this entry, because the paper could not be
identified.** Substituting a plausible-looking paper would produce an entire
folder of confident, wrong output — the failure mode this project exists to
prevent.

## What was searched

| Query | Result |
|---|---|
| `ULD-Net` (exact phrase, arXiv) | no results |
| `ULD-Net medical image segmentation` | no results |
| `ULD-Net universal lesion detection` | no results |
| `ULD Net 3D medical image segmentation network` | unrelated (MVP-Net, universal lesion detection) |
| `unsupervised lesion detection network segmentation` | unrelated (USL-Net, ASC-Net, Patch2Loc) |
| GitHub `ULD-Net in:name,description` | 4 repositories, none a medical segmentation architecture |

The two real GitHub hits are:

- `xiexi51/ULD-Net` — "Ultra-Low-Degree Fully Polynomial…", private-inference
  networks, unrelated to medical imaging.
- `caoxin918/ULD-Net-3D-Unsupervised-Learning-by-Dense-Similarity-Learning-with-Equivariant-Crop`
  — 3D **point-cloud** self-supervised learning, not volumetric medical
  segmentation.

Neither belongs in a comparison table alongside V-Net, UNETR and Swin UNETR.

## What would resolve it

Any one of:

- the arXiv identifier, or
- the full paper title, or
- the venue and year, or
- the GitHub repository the authors released.

Then run:

```bash
.venv/bin/python _run_batch.py 11-ULD-Net
```

after adding the identifier to `PAPERS` in `_run_batch.py`.

## Note on the other ten

The remaining ten entries all resolved to a paper whose title contains the
architecture name, which is recorded per paper in `01_resolve_paper.json` with
the evidence for the match. `output/_resolution.json` holds the full candidate
lists considered.
