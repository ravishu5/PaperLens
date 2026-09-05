"""Resolve the 11 architectures to arXiv papers, and verify each match."""
import json, os, sys
from paperlens.graph.store import Store
from paperlens.server import store as _s
from paperlens.sources import arxiv

TARGETS = [
    ("VNet",         "V-Net Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation", ["v-net"]),
    ("SegResNet",    "3D MRI brain tumor segmentation using autoencoder regularization", ["autoencoder regularization", "segresnet"]),
    ("nnUNet",       "nnU-Net self-configuring method for deep learning-based biomedical image segmentation", ["nnu-net", "nn-unet"]),
    ("TransBTS",     "TransBTS Multimodal Brain Tumor Segmentation Using Transformer", ["transbts"]),
    ("UNETR",        "UNETR Transformers for 3D Medical Image Segmentation", ["unetr"]),
    ("nnFormer",     "nnFormer Interleaved Transformer for Volumetric Segmentation", ["nnformer"]),
    ("Swin UNETR",   "Swin UNETR Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images", ["swin unetr"]),
    ("RepUX-Net",    "Scaling Up 3D Kernels with Bayesian Frequency Re-parameterization for Medical Image Segmentation", ["repux", "re-parameterization"]),
    ("UNesT",        "UNesT Local Spatial Representation Learning with Hierarchical Transformer for Efficient Medical Segmentation", ["unest", "unest"]),
    ("deformUX-Net", "Deformable Large Kernel Attention 3D UX-Net medical image segmentation", ["deform", "ux-net"]),
    ("ULD-Net",      "ULD-Net medical image segmentation", ["uld-net", "uld"]),
]

store = Store()
out = []
for slug, query, must in TARGETS:
    rec = {"name": slug, "query": query, "candidates": [], "chosen": None,
           "match_confidence": "UNKNOWN", "why": ""}
    try:
        hits = arxiv.search(store, query, max_results=8)
    except Exception as e:
        rec["why"] = f"search failed: {e}"
        out.append(rec); continue
    for h in hits:
        rec["candidates"].append({"arxiv_id": h.arxiv_id, "title": h.title,
                                  "year": (h.published_at or "")[:4]})
    # Choose only when the title actually contains a distinguishing token.
    for h in hits:
        low = h.title.lower()
        if any(m in low for m in must):
            rec["chosen"] = h.arxiv_id
            rec["chosen_title"] = h.title
            rec["match_confidence"] = "LIKELY"
            rec["why"] = f"title contains {[m for m in must if m in low]}"
            break
    if not rec["chosen"] and hits:
        rec["why"] = "no candidate title contained a distinguishing token"
    out.append(rec)
    print(f"{slug:<14} -> {rec['chosen'] or 'UNRESOLVED':<12} {rec.get('chosen_title','')[:62]}")

json.dump(out, open("output/_resolution.json", "w"), indent=2)
print("\nresolved:", sum(1 for r in out if r["chosen"]), "/", len(out))
