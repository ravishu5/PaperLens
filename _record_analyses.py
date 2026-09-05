"""Record a method analysis per paper, then re-run the mapping-dependent tools.

This is the agent half of the write-back design (D-009): the server supplies the
skeleton, the agent identifies the method components, and the server validates
that every component cites a section that actually exists before storing it.
"""
from __future__ import annotations

import dataclasses, json, sys, traceback
from pathlib import Path

from paperlens import config as _cfg
_cfg.quiet_dependencies()

from paperlens.code.indexer import index_repository
from paperlens.correlate.analysis import record_paper_analysis
from paperlens.correlate.compare import (compare_paper_with_code,
                                         find_implementation_gaps)
from paperlens.correlate.mapper import map_paper_to_code
from paperlens.correlate.plan import build_reproduction_plan
from paperlens.graph.store import Store
from paperlens.resources import resolve

OUT = Path("output")

# component = (name, kind, section_path, description)
ANALYSES: dict[str, tuple[str, list[tuple[str, str, str, str]]]] = {
  "01-VNet": ("1606.04797", [
    ("V-Net Architecture", "architecture", "2", "Fully convolutional volumetric encoder-decoder with residual stages."),
    ("Dice Loss Layer", "loss", "3", "Differentiable Dice coefficient used directly as the objective."),
    ("Training Procedure", "training", "3.1", "Volumetric training with elastic deformation augmentation."),
    ("Inference", "inference", "3.2", "Whole-volume prediction at test time."),
  ]),
  "02-SegResNet": ("1810.11654", [
    ("Encoder", "architecture", "3.1", "ResNet-style encoder with group normalisation."),
    ("Decoder", "architecture", "3.2", "Decoder producing the segmentation map."),
    ("Variational Autoencoder Branch", "module", "3.3", "VAE branch regularising the shared encoder."),
    ("Loss", "loss", "3.4", "Dice loss combined with VAE reconstruction and KL terms."),
    ("Optimization", "optimizer", "3.5", "Adam with a decaying learning-rate schedule."),
    ("Data Preprocessing and Augmentation", "preprocessing", "3.7", "Intensity normalisation and random augmentation."),
  ]),
  "03-nnUNet": ("1809.10486", [
    ("Network Architectures", "architecture", "2.1", "2D, 3D and cascaded U-Net variants configured per dataset."),
    ("Preprocessing", "preprocessing", "2.2", "Resampling and intensity normalisation derived from the dataset fingerprint."),
    ("Training Procedure", "training", "2.3", "Fixed training schedule with on-the-fly augmentation."),
    ("Inference", "inference", "2.4", "Sliding-window prediction with test-time augmentation."),
    ("Postprocessing", "inference", "2.5", "Connected-component postprocessing chosen by cross-validation."),
  ]),
  "04-TransBTS": ("2103.04430", [
    ("Overall Architecture", "architecture", "2.1", "3D CNN encoder, transformer bottleneck, CNN decoder."),
    ("Network Encoder", "architecture", "2.2", "3D convolutional encoder feeding transformer tokens."),
    ("Network Decoder", "architecture", "2.3", "Progressive upsampling decoder with skip connections."),
  ]),
  "05-UNETR": ("2103.10504", [
    ("Architecture", "architecture", "3.1", "Pure transformer encoder with a CNN decoder via skip connections."),
    ("Loss Function", "loss", "3.2", "Combined soft Dice and cross-entropy loss."),
    ("Datasets", "dataset", "4.1", "BTCV and MSD volumetric segmentation datasets."),
    ("Implementation Details", "training", "4.3", "Training schedule, optimiser and augmentation settings."),
  ]),
  "06-nnFormer": ("2109.03201", [
    ("Encoder", "architecture", "3.2", "Interleaved convolution and self-attention encoder."),
    ("Bottleneck", "module", "3.3", "Transformer bottleneck between encoder and decoder."),
    ("Decoder", "architecture", "3.4", "Symmetric decoder with skip connections."),
    ("Implementation details", "training", "4.1", "Optimiser, schedule and augmentation settings."),
  ]),
  "07-SwinUNETR": ("2201.01266", [
    ("Encoder", "architecture", "3.1", "Swin transformer encoder over 3D patches with shifted windows."),
    ("Decoder", "architecture", "3.2", "CNN decoder connected by skip connections at each resolution."),
    ("Loss Function", "loss", "3.3", "Soft Dice loss over the segmentation output."),
    ("Implementation Details", "training", "3.4", "Training schedule, optimiser and augmentation."),
    ("Dataset and Model Ensembling", "dataset", "3.5", "BraTS data and ensembling across folds."),
  ]),
  "08-RepUX-Net": ("2303.05785", [
    ("Bayesian Frequency Re-parameterization", "module", "3.2", "Re-parameterises large 3D kernels with a Bayesian frequency prior."),
    ("Model Architecture", "architecture", "3.3", "UX-Net backbone using the re-parameterised large kernels."),
    ("Experimental Setup", "training", "4", "Datasets, optimiser and training schedule."),
  ]),
  "09-UNesT": ("2209.14378", [
    ("Hierarchical Transformer Encoder", "architecture", "3.1", "Hierarchical transformer encoder over local 3D blocks."),
    ("3D Block Aggregation", "module", "3.2", "Block aggregation module coupling neighbouring local volumes."),
    ("Decoder", "architecture", "3.3", "Convolutional decoder producing the segmentation."),
    ("Dataset", "dataset", "4.1", "Whole-brain, renal and multi-organ datasets."),
    ("Implementation Details", "training", "4.2", "Optimiser, schedule and augmentation."),
  ]),
  "10-deformUX-Net": ("2310.00199", [
    ("Depthwise Deformable Convolution", "module", "4.1", "Depthwise deformable convolution with tri-planar offsets."),
    ("Complete Architecture", "architecture", "4.2", "Full DeformUX-Net backbone."),
    ("Experimental Setup", "training", "5", "Datasets, optimiser and training schedule."),
  ]),
}


def record(slug: str) -> dict:
    arxiv_id, comps = ANALYSES[slug]
    folder = OUT / slug
    folder.mkdir(parents=True, exist_ok=True)
    store = Store()
    analysis = {"components": [
        {"name": n, "kind": k, "description": d,
         "evidence": [f"paperlens://paper/{arxiv_id}/section/{sec}"]}
        for n, k, sec, d in comps]}
    res = record_paper_analysis(store, arxiv_id, analysis, author="agent:batch")
    (folder / "22_record_paper_analysis.json").write_text(
        json.dumps(res, indent=2, default=str))
    print(f"  {slug}: accepted {len(res['accepted']['components'])}, "
          f"rejected {len(res['rejected'])}")
    for r in res["rejected"]:
        print(f"      rejected {r.get('name')}: {r['reason'][:80]}")

    summary = json.load(open(folder / "00_summary.json"))
    repo = summary.get("repo")
    if not repo:
        return res
    try:
        index_repository(store, repo)
        for name, fn in [
            ("10_map_paper_to_code", lambda: map_paper_to_code(store, arxiv_id, repo)),
            ("11_compare_paper_with_code", lambda: compare_paper_with_code(store, arxiv_id, repo)),
            ("16_find_implementation_gaps", lambda: find_implementation_gaps(store, arxiv_id, repo)),
            ("19_build_reproduction_plan", lambda: build_reproduction_plan(store, arxiv_id, repo)),
            ("17_resource_mapping", lambda: resolve(store, f"paperlens://mapping/{arxiv_id}/{repo}")),
            ("18_resource_differences", lambda: resolve(store, f"paperlens://difference/{arxiv_id}/{repo}")),
            ("21_reverse_engineer_paper", lambda: _reverse(store, arxiv_id)),
        ]:
            (folder / f"{name}.json").write_text(json.dumps(fn(), indent=2, default=str))
    except Exception:
        print(f"      ! re-run failed:\n{traceback.format_exc()[-600:]}")
    return res


def _reverse(store, arxiv_id):
    import paperlens.server as srv
    srv._store = store
    return srv.reverse_engineer_paper(arxiv_id)


if __name__ == "__main__":
    only = sys.argv[1:] or list(ANALYSES)
    for slug in only:
        if slug in ANALYSES:
            record(slug)
    print("DONE")
