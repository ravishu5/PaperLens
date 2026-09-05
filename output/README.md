# PaperLens run — 11 volumetric segmentation architectures

One folder per paper. Every file is a tool's raw JSON output, named for
the tool that produced it, in the order the pipeline runs them.

Papers processed: **12**.

> **Two notes on identification.** *ULD-Net* is not on arXiv — it is published in *Biomedical Signal Processing and Control* (DOI `10.1016/j.bspc.2025.108746`), which is why the earlier searches found nothing. It is entry 12, ingested from a PDF with its reference list recovered from the publisher.
>
> The identifier first supplied for it, `2404.13024`, resolves to *BANF: Band-limited Neural Fields*, an unrelated neural-fields paper. It was run too, and is filed under its own title as entry 11 rather than under the label it arrived with.


## Resolution

| # | Architecture | Identifier | Title |
|---|---|---|---|
| 1 | V-Net | [`1606.04797`](https://arxiv.org/abs/1606.04797) | V-Net: Fully Convolutional Neural Networks for Volumetric Medical Imag |
| 2 | SegResNet | [`1810.11654`](https://arxiv.org/abs/1810.11654) | 3D MRI brain tumor segmentation using autoencoder regularization |
| 3 | nnU-Net | [`1809.10486`](https://arxiv.org/abs/1809.10486) | nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segment |
| 4 | TransBTS | [`2103.04430`](https://arxiv.org/abs/2103.04430) | TransBTS: Multimodal Brain Tumor Segmentation Using Transformer |
| 5 | UNETR | [`2103.10504`](https://arxiv.org/abs/2103.10504) | UNETR: Transformers for 3D Medical Image Segmentation |
| 6 | nnFormer | [`2109.03201`](https://arxiv.org/abs/2109.03201) | nnFormer: Interleaved Transformer for Volumetric Segmentation |
| 7 | Swin UNETR | [`2201.01266`](https://arxiv.org/abs/2201.01266) | Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumor |
| 8 | RepUX-Net | [`2303.05785`](https://arxiv.org/abs/2303.05785) | Scaling Up 3D Kernels with Bayesian Frequency Re-parameterization for  |
| 9 | UNesT | [`2209.14378`](https://arxiv.org/abs/2209.14378) | UNesT: Local Spatial Representation Learning with Hierarchical Transfo |
| 10 | DeformUX-Net | [`2310.00199`](https://arxiv.org/abs/2310.00199) | DeformUX-Net: Exploring a 3D Foundation Backbone for Medical Image Seg |
| 11 | BANF (supplied as ULD-Net) | [`2404.13024`](https://arxiv.org/abs/2404.13024) | BANF: Band-limited Neural Fields for Levels of Detail Reconstruction |
| 12 | ULD-Net | [`10.1016/j.bspc.2025.108746`](https://doi.org/10.1016/j.bspc.2025.108746) | ULD-Net: A U-shaped branch large kernel depthwise convolution volume n |

## What was extracted

| Architecture | Fidelity | Sections | Equations | Values | References | Components recorded |
|---|---|---:|---:|---:|---:|---:|
| V-Net | `LATEX_EXACT` | 8 | 2 | 1 | 20 | 4 |
| SegResNet | `LATEX_EXACT` | 12 | 5 | 4 | 23 | 6 |
| nnU-Net | `LATEX_EXACT` | 19 | 2 | 1 | 16 | 5 |
| TransBTS | `LATEX_EXACT` | 12 | 3 | 7 | 24 | 3 |
| UNETR | `LATEX_EXACT` | 19 | 9 | 11 | 55 | 4 |
| nnFormer | `LATEX_EXACT` | 14 | 12 | 4 | 40 | 4 |
| Swin UNETR | `LATEX_EXACT` | 10 | 3 | 8 | 43 | 5 |
| RepUX-Net | `LATEX_EXACT` | 11 | 9 | 5 | 23 | 3 |
| UNesT | `LATEX_EXACT` | 40 | 2 | 14 | 71 | 5 |
| DeformUX-Net | `LATEX_EXACT` | 18 | 3 | 6 | 32 | 3 |
| BANF (supplied as ULD-Net) | `LATEX_EXACT` | 19 | 22 | 1 | 53 | 4 |
| ULD-Net | `PDF_DERIVED` | 19 | 4 | 12 | 53 | 0 |

## Implementation found

| Architecture | Top candidate | Relation | Confidence |
|---|---|---|---|
| V-Net | `faustomilletari/VNet` | OFFICIAL | CONFIRMED |
| SegResNet | `sinclairjang/3D-MRI-brain-tumor-segmentation-using-autoencoder-regularization` | REPRODUCTION | LIKELY |
| nnU-Net | `MIC-DKFZ/nnUNet` | THIRD_PARTY | POSSIBLE |
| TransBTS | `Rubics-Xuan/TransBTS` | OFFICIAL | CONFIRMED |
| UNETR | `mkara44/unetr_pytorch` | REPRODUCTION | LIKELY |
| nnFormer | `282857341/nnFormer` | THIRD_PARTY | LIKELY |
| Swin UNETR | `hxhxhx33/SwinUNETR` | THIRD_PARTY | LIKELY |
| RepUX-Net | `MASILab/RepUX-Net` | OFFICIAL | CONFIRMED |
| UNesT | `MASILab/UNesT` | OFFICIAL | CONFIRMED |
| DeformUX-Net | `MASILab/deform-uxnet` | OFFICIAL | CONFIRMED |
| BANF (supplied as ULD-Net) | `theialab/banf` | THIRD_PARTY | LIKELY |
| ULD-Net | `caoxin918/ULD-Net-3D-Unsupervised-Learning-by-Dense-Similarity-Learning-with-Equivariant-Crop` | THIRD_PARTY | POSSIBLE |

## Correlation and comparison

| Architecture | Matched | Absent | Ambiguous | Unknown | Blocking | Significant | Gaps | Predecessors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V-Net | 1 | 0 | 6 | 1 | 0 | 0 | 2 | 15 |
| SegResNet | 1 | 0 | 6 | 6 | 0 | 0 | 0 | 15 |
| nnU-Net | 0 | 0 | 5 | 4 | 0 | 0 | 2 | 15 |
| TransBTS | 0 | 1 | 10 | 5 | 0 | 1 | 1 | 15 |
| UNETR | 2 | 0 | 9 | 11 | 0 | 0 | 0 | 15 |
| nnFormer | 2 | 0 | 6 | 4 | 0 | 0 | 2 | 15 |
| Swin UNETR | 1 | 1 | 8 | 4 | 0 | 2 | 0 | 15 |
| RepUX-Net | 0 | 0 | 6 | 3 | 0 | 0 | 4 | 15 |
| UNesT | 2 | 0 | 15 | 12 | 0 | 0 | 3 | 15 |
| DeformUX-Net | 0 | 0 | 8 | 8 | 0 | 0 | 5 | 15 |
| BANF (supplied as ULD-Net) | 0 | 0 | 4 | 7 | 0 | 0 | 4 | 15 |
| ULD-Net | 1 | 11 | 3 | 9 | 0 | 10 | 1 | 15 |

## Files in each folder

```
00_summary.json
01_resolve_paper.json
02_ingest_paper.json
03_get_paper_skeleton.json
04_resource_paper.json
05_resource_values.json
06_resource_urls.json
07_resource_references.json
08_find_implementations.json
09_index_repository.json
10_map_paper_to_code.json
11_compare_paper_with_code.json
12_trace_research_lineage.json
13_find_sota_successors.json
14_trace_method.json
15_resource_lineage.json
16_find_implementation_gaps.json
17_resource_mapping.json
18_resource_differences.json
19_build_reproduction_plan.json
20_compare_implementations.json
21_reverse_engineer_paper.json
22_record_paper_analysis.json
```

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

