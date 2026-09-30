"""Per-view VAE execution and publication for generation stages."""

from __future__ import annotations

import gc
import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from fdanyone.config import INFERENCE
from fdanyone.model.loader import load_vae
from fdanyone.video import write_video

if TYPE_CHECKING:
    from diffsynth.models.wan_video_vae import WanVideoVAE38

LOGGER = logging.getLogger("fdanyone")


class VaeExecutor:
    """Run independent views on an ordered CUDA device pool."""

    def __init__(self, model: WanVideoVAE38, devices: tuple[str, ...]) -> None:
        self.devices = devices
        self._models = [model]
        self.last_peak_vram_bytes: dict[str, dict[str, int]] = {}

    @classmethod
    def load(cls, path: str | Path, devices: tuple[str, ...]) -> VaeExecutor:
        return cls(load_vae(path), devices)

    @property
    def latent_channels(self) -> int:
        return int(self._models[0].model.z_dim)

    @property
    def upsampling_factor(self) -> int:
        return int(self._models[0].upsampling_factor)

    def _run_workers(
        self,
        inputs: Tensor,
        publication: tuple[Path, Fraction, ThreadPoolExecutor] | None = None,
    ) -> tuple:
        num_views = len(inputs)
        worker_count = min(num_views, len(self.devices))
        while len(self._models) < worker_count:
            self._models.append(deepcopy(self._models[0]))
        stopped = Event()
        outputs = {}
        peaks: dict[str, dict[str, int]] = {}

        pool = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="vae")
        try:
            futures = [
                pool.submit(self._run_worker, index, worker_count, inputs, stopped, publication)
                for index in range(worker_count)
            ]
            for future in as_completed(futures):
                device, peak, results = future.result()
                peaks[device] = peak
                outputs.update(results)
        finally:
            stopped.set()
            pool.shutdown(wait=True, cancel_futures=True)
            # Join every worker before offloading models or releasing CUDA mappings.
            for index in range(worker_count):
                torch.cuda.set_device(self.devices[index])
                self._models[index].to("cpu")
                torch.cuda.empty_cache()
        self.last_peak_vram_bytes = peaks
        return tuple(outputs[index] for index in range(num_views))

    def _run_worker(
        self,
        index: int,
        worker_count: int,
        inputs: Tensor,
        stopped: Event,
        publication: tuple[Path, Fraction, ThreadPoolExecutor] | None,
    ) -> tuple[str, dict[str, int], dict]:
        device = self.devices[index]
        model = self._models[index]
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        model.to(device)
        outputs = {}
        pending: Future[Path] | None = None

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for view_index in range(index, len(inputs), worker_count):
                if stopped.is_set():
                    break
                batched = inputs[view_index].unsqueeze(0).to(dtype=torch.bfloat16, device="cpu" if INFERENCE.tiled_vae else device)
                if publication is None:
                    if INFERENCE.tiled_vae:
                        tile_size = tuple(size * model.upsampling_factor for size in INFERENCE.vae_tile_size)
                        tile_stride = tuple(stride * model.upsampling_factor for stride in INFERENCE.vae_tile_stride)
                        encoded = model.tiled_encode(batched, device, tile_size, tile_stride)
                    else:
                        encoded = model.single_encode(batched, device)
                    del batched
                    outputs[view_index] = encoded[0].detach().to(device="cpu").contiguous()
                    del encoded
                    continue

                video_root, fps, codec_pool = publication
                if INFERENCE.tiled_vae:
                    decoded = model.tiled_decode(batched, device, INFERENCE.vae_tile_size, INFERENCE.vae_tile_stride)
                else:
                    decoded = model.single_decode(batched, device)
                del batched
                # Preserve FP32 scaling, then transfer packed uint8 frames to the CPU.
                scaled = decoded[0].detach().float()
                scaled.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)
                frames = scaled.permute(1, 2, 3, 0).to(dtype=torch.uint8, memory_format=torch.contiguous_format)
                rgb_frames = tuple(frames.cpu().numpy())
                del decoded, scaled, frames

                # Decode while the previous video is encoded; queue at most one per worker.
                if pending is not None:
                    pending.result()
                if stopped.is_set():
                    break
                LOGGER.info("Publishing target camera %02d", view_index)
                pending = codec_pool.submit(
                    write_video, iter(rgb_frames), video_root / f"{view_index:02d}.mp4", fps,
                    crf=INFERENCE.target_h264_crf, preset=INFERENCE.h264_preset,
                )
                outputs[view_index] = pending
                del rgb_frames
        if publication is not None:
            outputs = {index: future.result() for index, future in outputs.items()}
        torch.cuda.synchronize(device)
        peak = {
            "allocated": int(torch.cuda.max_memory_allocated(device)),
            "reserved": int(torch.cuda.max_memory_reserved(device)),
        }
        return device, peak, outputs

    def encode(self, videos: Tensor) -> Tensor:
        """Encode CPU-resident views and gather latents by input index."""

        return torch.stack(self._run_workers(videos))

    def publish_targets(self, latents: Tensor, output_dir: Path, fps: Fraction) -> tuple[Path, ...]:
        video_root = output_dir / "videos"
        video_root.mkdir(parents=True, exist_ok=False)

        worker_count = min(len(latents), len(self.devices))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="video-codec") as codec_pool:
            return self._run_workers(latents, (video_root, fps, codec_pool))

    def release_replicas(self) -> None:
        """Keep one CPU model while releasing stage-local parallel replicas."""

        del self._models[1:]
        gc.collect()

    def close(self) -> None:
        """Release all CPU replicas after the final VAE stage."""

        self._models.clear()
        gc.collect()
