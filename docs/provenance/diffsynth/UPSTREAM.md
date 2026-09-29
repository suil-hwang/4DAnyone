# DiffSynth-Studio provenance

This directory preserves the provenance of 4DAnyone's inference-only extract from [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), licensed under Apache-2.0. The retained source is now in `fdanyone/model/dit.py` and `fdanyone/model/pose_encoder.py`; its license is copied at `third_party/licenses/DIFFSYNTH_LICENSE`.

- Public base revision: `04e39f7de53df7276a7b40ca1791c2a393e05ff3`
- Research fork revision used by the original experiment: `c00782d90c872c97bda4745a9e6a41a0a4a7c4db`
- `UPSTREAM.patch` SHA-256 (LF line endings): `178e6035e451f94a2122fa4d2c876a488546964768b90275549cbf609f97daba`
- Extracted: 2026-07-16

The research fork revision was not anonymously reachable when the release contract was audited. `UPSTREAM.patch` therefore records the exact binary-safe diff from the public base to the research revision for the Wan/SpaTem/scheduler sources from which the reader graph was derived. Unrelated research-fork changes are deliberately excluded. `VENDORED_FILES.txt` records the historical extract and its current destinations.

The VAE now comes from the external `diffsynth==2.1.8` package (`diffsynth.models.wan_video_vae.WanVideoVAE38`). Its source and the generic flow-matching scheduler are no longer bundled. The fixed inference schedule and Euler update are implemented by the existing model loading and denoising code. The custom DiT and PoseEncoder were moved without changing their arithmetic or checkpoint keys. This migration does not change the original attribution of those model implementations.

4DAnyone subsequently reduced the generic Wan implementation to the one checkpoint-key-compatible reader graph: video attention, MVS attention in every block, ViewPack source tokens, fixed prompt context, and precomputed RGB-pose features. Registry, downloader, text encoder/tokenizer, generic pipeline, training adapters, camera/control modules, VRAM wrappers, and alternate schedulers are not part of the reader closure. The retained automatic attention fallback remains observable in run metadata.

The reader inference path bounds the temporary FP64 RoPE workspace and reuses PoseEncoder/FFN activation storage when gradients are disabled. These release-specific memory-lifetime changes preserve checkpoint parameter keys and the full-group attention and FFN matrix-multiplication shapes. The offline UMT5 conversion implementation used to generate the frozen prompt asset is development-only and not part of the reader distribution; it retains this license and provenance.

The conditioning builder prepares PoseEncoder's contiguous temporal-prefix input directly. Inference DiT blocks own and update one residual stream; target and null pose banks are consumed before those blocks. The former local VAE also used compact temporal caches and reused completed branch storage; these local VAE memory-lifetime changes are not part of the external package. The package migration preserves the normalization convention and temporal chunk boundaries.

To reproduce the retained research sources before pruning:

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
git -C DiffSynth-Studio checkout 04e39f7de53df7276a7b40ca1791c2a393e05ff3
git -C DiffSynth-Studio apply --check /path/to/UPSTREAM.patch
git -C DiffSynth-Studio apply /path/to/UPSTREAM.patch
```

The patch contains research-code additions and is itself source code. It remains covered by `third_party/licenses/DIFFSYNTH_LICENSE` and this attribution.
