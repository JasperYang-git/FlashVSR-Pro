#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 FlashVSR-Pro 的实时直播视频超分脚本

整体链路（推荐示例）：

1. 主播端（OBS）：
   - 推流地址：rtmp://你的服务器IP:1935/live/src
   - 分辨率建议：1280x720 或 1920x1080，编码 H.264 + AAC

2. 服务器端（已启动 SRS）：
   - 本脚本从 SRS 拉取低清流：  --input-rtmp  rtmp://服务器IP:1935/live/src
   - 使用 FlashVSR-Pro 进行超分：--mode tiny-long --tile-dit --tile-vae
   - 将高清结果推回 SRS：        --output-rtmp rtmp://服务器IP:1935/live/sr
   - 当指定 --keep-audio 时，会从 input-rtmp 流中复制音频，保证声音不变。

3. 观众端：
   - 直接播放 SRS 的高清流：rtmp://服务器IP:1935/live/sr（或对应的 HLS 地址）

用法示例（2 倍超分，保持音频）：

python livestream_infer.py \
  --input-rtmp  rtmp://127.0.0.1:1935/live/src \
  --output-rtmp rtmp://127.0.0.1:1935/live/sr  \
  --input-width 1280 --input-height 720        \
  --mode tiny-long --tile-dit --tile-vae      \
  --scale 2.0 --keep-audio
"""

import os
import sys
import time
import argparse
import threading
import queue
import select
from collections import deque
from typing import List, Optional, Union

import numpy as np
from PIL import Image

import subprocess

import torch

# 确保可以导入同目录下的 infer.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from infer import (  # type: ignore
    init_pipeline,
    compute_scaled_and_target_dims,
    process_batch_gpu,
    NVENC_AVAILABLE,
    TILE_AVAILABLE,
    apply_tiled_inference_simple,
)


class StreamBuffer:
    """简单的线程安全帧缓冲队列"""

    def __init__(self, max_size: int = 300):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.frame_count = 0

    def put(self, frame: Image.Image) -> None:
        with self.lock:
            self.buffer.append(frame)
            self.frame_count += 1

    def get_batch(self, batch_size: int) -> Optional[List[Image.Image]]:
        with self.lock:
            if len(self.buffer) >= batch_size:
                batch = [self.buffer.popleft() for _ in range(batch_size)]
                return batch
        return None

    def size(self) -> int:
        with self.lock:
            return len(self.buffer)


OutputFrame = Union[Image.Image, np.ndarray]


def tensor2video_fast(frames: torch.Tensor) -> List[np.ndarray]:
    """
    Convert output tensor to uint8 numpy frames efficiently.

    Key optimization for livestream:
    quantize to uint8 on GPU first, then transfer to CPU once, which avoids
    sending float32 frames over PCIe and avoids a large CPU-side astype().
    """
    if frames.ndim == 5:
        frames = frames.squeeze(0)

    frames = (
        frames.permute(1, 2, 3, 0)
        .add(1.0)
        .mul(127.5)
        .clamp(0, 255)
        .to(torch.uint8)
        .contiguous()
        .cpu()
        .numpy()
    )
    return [frame for frame in frames]


class FlashVSRRealtime:
    """
    基于 infer.py 的实时 FlashVSR 推理器

    这里不重新实现模型加载逻辑，而是直接复用 infer.py 中的 init_pipeline、
    compute_scaled_and_target_dims、process_batch_gpu 等函数，以保证行为一致。
    """

    def __init__(
        self,
        mode: str = "tiny-long",
        tile_dit: bool = False,
        tile_vae: bool = False,
        tile_size: int = 256,
        overlap: int = 24,
        device: str = "cuda",
        dtype: str = "bf16",
        scale: float = 2.0,
        input_width: int = 1280,
        input_height: int = 720,
        sparse_ratio: float = 2.0,
        kv_ratio: float = 3.0,
        local_range: int = 11,
        seed: int = 0,
        warmup: bool = True,
        warmup_frames: int = 9,
        lq_bootstrap_windows: int = 3,
        color_fix: bool = False,
        low_latency: bool = True,
        profile_batches: int = 0,
        warmup_on_init: bool = True,
        thread_warmup_passes: int = 0,
    ):
        self.mode = mode
        self.tile_dit = tile_dit
        self.tile_vae = tile_vae
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = device
        self.dtype_str = dtype
        self.scale = scale
        self.in_w = input_width
        self.in_h = input_height
        self.sparse_ratio = sparse_ratio
        self.kv_ratio = kv_ratio
        self.local_range = local_range
        self.seed = seed
        self.warmup_enabled = warmup
        self.warmup_frames = warmup_frames
        self._warmup_done = False
        self.lq_bootstrap_windows = lq_bootstrap_windows
        self.color_fix = bool(color_fix)
        self.low_latency = low_latency
        self.profile_batches = int(profile_batches) if profile_batches else 0
        self._batch_idx = 0
        self.warmup_on_init = bool(warmup_on_init)
        self.thread_warmup_passes = max(0, int(thread_warmup_passes))

        # dtype 映射
        if self.dtype_str == "fp16":
            self.dtype_torch = torch.float16
        elif self.dtype_str == "bf16":
            self.dtype_torch = torch.bfloat16
        else:
            self.dtype_torch = torch.float32

        # 启用 TF32 & cudnn benchmark，和 infer.py 保持一致
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # 计算放大后的尺寸以及对齐到 128 倍数的推理尺寸
        self.sW, self.sH, self.tW, self.tH = compute_scaled_and_target_dims(
            self.in_w, self.in_h, scale=self.scale, multiple=128
        )
        self.exact_w = self.sW
        self.exact_h = self.sH

        print(
            f"[FlashVSRRealtime] Input {self.in_w}x{self.in_h}, "
            f"scale {self.scale}x -> target {self.exact_w}x{self.exact_h}, "
            f"padded to {self.tW}x{self.tH}"
        )

        # 构造一个简单的 Namespace 交给 infer.init_pipeline 复用加载逻辑
        args = argparse.Namespace()
        args.mode = self.mode
        args.device = self.device
        args.dtype = self.dtype_str
        args.tile_vae = self.tile_vae
        args.tile_size = self.tile_size
        args.overlap = self.overlap
        # Hint for infer.init_pipeline(): keep models on GPU, avoid CPU offload.
        args.realtime_low_latency = bool(self.low_latency)

        # init_pipeline 内部会根据 mode 选择 VAE / TCDecoder 并加载 DiT
        self.pipe, self.vae_instance = init_pipeline(args)

        if self.device == "cuda":
            # Reduce first-run jitter by keeping allocator behavior stable
            try:
                torch.cuda.set_per_process_memory_fraction(1.0)
            except Exception:
                pass

        # Main-thread warmup: warms CUDA context on THIS thread only.
        # Actual inference runs on process_thread → first real batch can still pay ~10s "cold start"
        # unless thread_warmup_passes > 0 (see process_thread).
        if self.warmup_enabled and self.warmup_on_init:
            try:
                self.warmup(self.warmup_frames)
            except Exception as e:
                print(f"[FlashVSRRealtime] Warmup failed (will continue): {e}")

    def warmup(self, num_frames: int) -> None:
        """
        Multi-pass warmup to eliminate cold-start latency:
        - Pass 1: CUDA kernel compilation + memory allocation (slow)
        - Pass 2: PyTorch graph optimization (may still be slow)
        - Pass 3+: Verify stable performance
        """
        if self._warmup_done:
            return
        if num_frames < 5:
            num_frames = 5

        print(f"[FlashVSRRealtime] Warmup started (frames={num_frames}, passes=3, lq_bootstrap_windows={self.lq_bootstrap_windows})")
        dummy = [Image.new("RGB", (self.in_w, self.in_h), (0, 0, 0)) for _ in range(num_frames)]

        # Run 3 passes to ensure all compilation/optimization is done
        total_start = time.time()
        for pass_idx in range(3):
            pass_start = time.time()
            _ = self.process_batch(dummy)
            if self.device == "cuda":
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            pass_elapsed = time.time() - pass_start
            print(f"[FlashVSRRealtime] Warmup pass {pass_idx+1}/3: {pass_elapsed:.2f}s")

        total_elapsed = time.time() - total_start
        self._warmup_done = True
        print(f"[FlashVSRRealtime] Warmup complete in {total_elapsed:.2f}s total")

    def process_batch(self, frames: List[Image.Image]) -> List[OutputFrame]:
        """
        处理一批帧，返回超分后的帧列表（保持帧数一致）。
        """
        def sync_cuda() -> None:
            if self.device == "cuda":
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass

        if not frames:
            return frames

        num_frames = len(frames)
        if num_frames < 5:
            # 帧数太少时直接透传，避免模型对极短序列不稳定
            print(f"[FlashVSRRealtime] Batch too small ({num_frames}), bypass SR")
            return frames

        # ---- lightweight timing (optional) ----
        self._batch_idx += 1
        do_profile = self.profile_batches > 0 and self._batch_idx <= self.profile_batches
        sync_cuda()
        t0 = time.perf_counter()

        # 转为 numpy 数组列表，形状 (H, W, C)，uint8
        batch_arr = [np.array(f.convert("RGB"), dtype=np.uint8) for f in frames]

        # 利用 infer.py 中的 GPU 预处理逻辑：resize + pad + 归一化
        # 输出形状 (B, C, H, W)，位于 GPU 上
        lq_batch = process_batch_gpu(
            batch_arr,
            sH=self.sH,
            sW=self.sW,
            tH=self.tH,
            tW=self.tW,
            dtype=self.dtype_torch,
            device=self.device,
        )
        sync_cuda()
        t_pre = time.perf_counter()

        # 变换为 FlashVSR 需要的形状：1, C, T, H, W
        LQ = lq_batch.permute(1, 0, 2, 3).unsqueeze(0)

        # 组装与 infer.py 一致的 pipeline 参数
        pipeline_kwargs = {
            "prompt": "",
            "negative_prompt": "",
            "cfg_scale": 1.0,
            "num_inference_steps": 1,
            "seed": self.seed,
            "LQ_video": LQ,
            "num_frames": num_frames,
            "height": self.tH,
            "width": self.tW,
            "is_full_block": False,
            "if_buffer": True,
            "topk_ratio": self.sparse_ratio * 768 * 1280 / (self.tH * self.tW),
            "kv_ratio": self.kv_ratio,
            "local_range": self.local_range,
            "color_fix": self.color_fix,
            "lq_bootstrap_windows": self.lq_bootstrap_windows,
        }

        # VAE 分块参数（仅在 tile-vae 时生效）
        if self.tile_vae:
            vae_tile_size_latent = max(32, self.tile_size // 8)
            vae_overlap_latent = max(4, self.overlap // 8)
            pipeline_kwargs["tiled"] = True
            pipeline_kwargs["tile_size"] = (vae_tile_size_latent, vae_tile_size_latent)
            pipeline_kwargs["tile_stride"] = (
                vae_tile_size_latent - vae_overlap_latent,
                vae_tile_size_latent - vae_overlap_latent,
            )
            print(
                f"[FlashVSRRealtime] VAE tiling: size={pipeline_kwargs['tile_size']}, "
                f"stride={pipeline_kwargs['tile_stride']}"
            )

        # 执行推理
        start = time.perf_counter()
        with torch.inference_mode():
            if self.tile_dit and TILE_AVAILABLE:
                print(
                    f"[FlashVSRRealtime] DiT tiled inference, tile_size={self.tile_size}, overlap={self.overlap}"
                )
                tile_kwargs = dict(pipeline_kwargs)
                tile_kwargs.pop("LQ_video", None)
                vae_tile_size_tuple = tile_kwargs.pop("tile_size", None)

                video = apply_tiled_inference_simple(
                    self.pipe,
                    LQ,
                    tile_size=self.tile_size,
                    overlap=self.overlap,
                    tile_size_vae=vae_tile_size_tuple,
                    **tile_kwargs,
                )
            else:
                video = self.pipe(**pipeline_kwargs)
        sync_cuda()

        pipeline_elapsed = time.perf_counter() - start
        t_dit = time.perf_counter()

        # 输出形状：1, C, T, H, W，需要裁剪回精确分辨率 self.exact_h, self.exact_w
        if video.shape[-2] != self.exact_h or video.shape[-1] != self.exact_w:
            curr_h, curr_w = video.shape[-2], video.shape[-1]
            pad_h = curr_h - self.exact_h
            pad_w = curr_w - self.exact_w
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            video = video[
                ...,
                pad_top : pad_top + self.exact_h,
                pad_left : pad_left + self.exact_w,
            ]

        # 转回 numpy 帧（列表）
        to_video_start = time.perf_counter()
        frames_np = tensor2video_fast(video)
        sync_cuda()
        to_video_elapsed = time.perf_counter() - to_video_start
        t_decode = time.perf_counter()

        # 保证输出帧数与输入一致
        if len(frames_np) > num_frames:
            frames_np = frames_np[:num_frames]
        elif len(frames_np) < num_frames:
            frames_np.extend([frames_np[-1]] * (num_frames - len(frames_np)))

        # Keep SR output as numpy arrays to avoid PIL -> numpy roundtrips in push_thread.
        out_frames = [np.ascontiguousarray(f) for f in frames_np]
        total_elapsed = time.perf_counter() - t0
        print(
            f"[FlashVSRRealtime] Batch {num_frames} frames: "
            f"pipeline={pipeline_elapsed:.2f}s to_video={to_video_elapsed:.2f}s "
            f"total={total_elapsed:.2f}s "
            f"ready_fps={num_frames / max(total_elapsed, 1e-6):.1f}"
        )
        if do_profile:
            # Note: t_dit includes DiT+decode inside pipeline; t_decode mostly covers tensor->cpu/uint8 conversion.
            print(
                "[FlashVSRRealtime][Profile] "
                f"batch={self._batch_idx} frames={num_frames} "
                f"preprocess={(t_pre - t0):.3f}s "
                f"pipeline={(t_dit - t_pre):.3f}s "
                f"to_video={(t_decode - t_dit):.3f}s "
                f"total={total_elapsed:.3f}s"
            )
        return out_frames


class RTMPCapture:
    """通过 ffmpeg+pipe 从 SRS/RTMP 拉取原始帧"""

    def __init__(self, rtmp_url: str, fps: int, width: int, height: int):
        self.rtmp_url = rtmp_url
        self.fps = fps
        self.width = width
        self.height = height
        self.process: Optional[subprocess.Popen] = None
        self._log_thread: Optional[threading.Thread] = None
        self._restart_lock = threading.Lock()
        self._partial_frame_buf = bytearray()

    def start(self) -> None:
        print(f"[RTMPCapture] Connecting to {self.rtmp_url} ...")
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-nostats",
            "-i",
            self.rtmp_url,
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            # Keep input low-latency-ish. (Doesn't guarantee perfection, but reduces buffering.)
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-f",
            "rawvideo",
            "-",
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            # Important: ffmpeg stderr isn't read anywhere in this script.
            # If left as PIPE, it can eventually fill up and block ffmpeg,
            # causing stdout to stop and live SR to "freeze".
            stderr=subprocess.DEVNULL,
            bufsize=10 * self.width * self.height * 3,
        )

        # Dump ffmpeg stderr asynchronously (useful to see if ffmpeg exits or errors).
        # def _log_ffmpeg() -> None:
        #     assert self.process is not None
        #     try:
        #         for line in iter(self.process.stderr.readline, b""):
        #             if not line:
        #                 break
        #             print(f"[RTMPCapture ffmpeg] {line.decode(errors='ignore').strip()}")
        #     except Exception:
        #         pass

        # self._log_thread = threading.Thread(target=_log_ffmpeg, daemon=True)
        # self._log_thread.start()
        print("[RTMPCapture] Started.")

    def read_frame(self) -> Optional[Image.Image]:
        if self.process is None or self.process.stdout is None:
            return None
        if self.process.poll() is not None:
            return None
        frame_size = self.width * self.height * 3
        # Avoid blocking forever on pipe reads when ffmpeg or upstream stalls.
        timeout_s = 1.0
        stdout = self.process.stdout
        while len(self._partial_frame_buf) < frame_size:
            if self.process.poll() is not None:
                return None
            try:
                ready, _, _ = select.select([stdout], [], [], timeout_s)
            except Exception:
                return None
            if not ready:
                # Let capture_thread handle restart logic on repeated timeouts.
                return None
            need = frame_size - len(self._partial_frame_buf)
            try:
                chunk = os.read(stdout.fileno(), need)
            except Exception:
                return None
            if not chunk:
                return None
            self._partial_frame_buf.extend(chunk)

        data = bytes(self._partial_frame_buf[:frame_size])
        del self._partial_frame_buf[:frame_size]
        if len(data) != frame_size:
            return None
        frame = np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3)
        return Image.fromarray(frame, "RGB")

    def stop(self) -> None:
        if self.process:
            try:
                if self.process.stdout:
                    self.process.stdout.close()
            except Exception:
                pass
            try:
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
                try:
                    self.process.wait(timeout=2.0)
                except Exception:
                    pass
            self.process = None

    def restart(self, backoff_s: float = 1.0) -> None:
        with self._restart_lock:
            try:
                self.stop()
            except Exception:
                pass
            if backoff_s > 0:
                time.sleep(backoff_s)
            self.start()


class RTMPPush:
    """
    将处理后帧通过 ffmpeg 推回 SRS。

    当 keep_audio=True 且提供 audio_source_url 时，ffmpeg 会从该 RTMP 流复制音频，
    从而实现“保留原始音频”的效果，等价于 infer.py 的 --keep-audio 语义。
    """

    def __init__(
        self,
        rtmp_url: str,
        fps: int,
        width: int,
        height: int,
        keep_audio: bool = False,
        audio_source_url: Optional[str] = None,
    ):
        self.rtmp_url = rtmp_url
        self.fps = fps
        self.width = width
        self.height = height
        self.keep_audio = keep_audio
        self.audio_source_url = audio_source_url
        self.process: Optional[subprocess.Popen] = None
        self._log_thread: Optional[threading.Thread] = None
        self._restart_lock = threading.Lock()

    def start(self) -> None:
        print(f"[RTMPPush] Pushing to {self.rtmp_url} ...")
        gop = max(int(self.fps), 1)

        if self.keep_audio and self.audio_source_url:
            # 双输入：0 为超分后原始帧，1 为原始 RTMP（只取音频）
            cmd = [
                "ffmpeg",
                "-loglevel",
                "error",
                "-nostats",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{self.width}x{self.height}",
                "-framerate",
                str(self.fps),
                "-i",
                "pipe:",
                "-i",
                self.audio_source_url,
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-g",
                str(gop),
                "-keyint_min",
                str(gop),
                "-sc_threshold",
                "0",
                "-bf",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-b:v",
                "6000k",
                "-maxrate",
                "6000k",
                "-bufsize",
                "3000k",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                "-flvflags",
                "no_duration_filesize",
                "-flush_packets",
                "1",
                "-f",
                "flv",
                self.rtmp_url,
            ]
        else:
            # 仅视频
            cmd = [
                "ffmpeg",
                "-loglevel",
                "error",
                "-nostats",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{self.width}x{self.height}",
                "-framerate",
                str(self.fps),
                "-i",
                "pipe:",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-g",
                str(gop),
                "-keyint_min",
                str(gop),
                "-sc_threshold",
                "0",
                "-bf",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-b:v",
                "6000k",
                "-maxrate",
                "6000k",
                "-bufsize",
                "3000k",
                "-flvflags",
                "no_duration_filesize",
                "-flush_packets",
                "1",
                "-f",
                "flv",
                self.rtmp_url,
            ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            # ffmpeg 一般不会往 stdout 输出有效内容；保留 PIPE 容易因缓冲区满导致异常行为。
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=10 * self.width * self.height * 3,
        )

        def _log_ffmpeg() -> None:
            proc = self.process
            if proc is None or proc.stderr is None:
                return
            try:
                for line in iter(proc.stderr.readline, b""):
                    if not line:
                        break
                    msg = line.decode(errors="ignore").strip()
                    if msg:
                        print(f"[RTMPPush ffmpeg] {msg}")
            except Exception:
                pass

        self._log_thread = threading.Thread(target=_log_ffmpeg, daemon=True)
        self._log_thread.start()
        print("[RTMPPush] Started.")

    def is_alive(self) -> bool:
        return bool(self.process is not None and self.process.poll() is None and self.process.stdin is not None)

    def restart(self, backoff_s: float = 1.0) -> None:
        with self._restart_lock:
            try:
                self.stop()
            except Exception:
                pass
            if backoff_s > 0:
                time.sleep(backoff_s)
            self.start()

    def write_frame(self, frame: OutputFrame) -> bool:
        if not self.process or not self.process.stdin:
            return False
        if self.process.poll() is not None:
            return False
        try:
            if isinstance(frame, Image.Image):
                arr = np.asarray(frame.convert("RGB"), dtype=np.uint8)
            else:
                arr = np.asarray(frame, dtype=np.uint8)
                if arr.ndim != 3 or arr.shape[2] != 3:
                    raise ValueError(f"Unexpected frame shape: {arr.shape}")
            arr = np.ascontiguousarray(arr)
            self.process.stdin.write(arr.tobytes())
            return True
        except Exception as e:
            print(f"[RTMPPush] write_frame error: {e}")
            return False

    def stop(self) -> None:
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
            except Exception:
                pass
            try:
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
                try:
                    self.process.wait(timeout=2.0)
                except Exception:
                    pass
            self.process = None


def capture_thread(capture: RTMPCapture, buffer: StreamBuffer) -> None:
    try:
        consecutive_none = 0
        while True:
            frame = capture.read_frame()
            if frame is None:
                consecutive_none += 1
                if consecutive_none == 1:
                    print("[CaptureThread] read_frame returned None (starting restart loop) ...")
                elif consecutive_none % 5 == 0:
                    print(f"[CaptureThread] read_frame None x{consecutive_none}, restarting capture ...")
                # Restart ffmpeg when we can't decode frames for a while.
                if consecutive_none >= 5:
                    consecutive_none = 0
                    try:
                        # Drop incomplete bytes from old pipe before restart.
                        capture._partial_frame_buf.clear()
                        capture.restart(backoff_s=1.0)
                    except Exception as e:
                        print(f"[CaptureThread] capture.restart failed: {e}")
                        time.sleep(2.0)
                else:
                    time.sleep(0.05)
                continue

            consecutive_none = 0
            buffer.put(frame)
            if buffer.frame_count % 30 == 0:
                print(
                    f"[CaptureThread] Captured {buffer.frame_count} frames, "
                    f"buffer size={buffer.size()}"
                )
    except Exception as e:
        print(f"[CaptureThread] Error: {e}")


def _cuda_set_device_for_thread(device: str) -> None:
    """PyTorch creates/initializes the CUDA context on first use per thread; set device explicitly."""
    if device == "cpu":
        return
    if device == "cuda":
        torch.cuda.set_device(0)
        return
    if device.startswith("cuda:"):
        try:
            torch.cuda.set_device(int(device.split(":")[-1]))
        except Exception:
            torch.cuda.set_device(0)


def _stage_prefix(boot_t0: Optional[float]) -> str:
    if boot_t0 is None:
        return "[StageTiming]"
    return f"[StageTiming +{time.monotonic() - boot_t0:.3f}s]"


def _safe_queue_qsize(q: "queue.Queue[Image.Image]") -> int:
    try:
        return int(q.qsize())
    except Exception:
        return -1


def process_thread(
    flashvsr: FlashVSRRealtime,
    buffer: StreamBuffer,
    output_queue: "queue.Queue[OutputFrame]",
    batch_size: int,
    bootstrap_batch_size: Optional[int] = None,
    boot_t0: Optional[float] = None,
) -> None:
    try:
        first = True
        first_real_batch_started = False
        first_real_batch_done = False
        first_output_enqueued = False
        bs0 = int(bootstrap_batch_size) if bootstrap_batch_size is not None else int(batch_size)
        if bs0 < 5:
            bs0 = 5

        def enqueue_output_frames(
            out_frames: List[OutputFrame], source: str, input_frames: int
        ) -> None:
            nonlocal first_output_enqueued
            for f in out_frames:
                enqueued = False
                try:
                    output_queue.put_nowait(f)
                    enqueued = True
                except queue.Full:
                    try:
                        _ = output_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        output_queue.put_nowait(f)
                        enqueued = True
                    except queue.Full:
                        pass

                if enqueued and not first_output_enqueued:
                    first_output_enqueued = True
                    print(
                        f"{_stage_prefix(boot_t0)} first_output_enqueued "
                        f"source={source} input_frames={input_frames} "
                        f"output_queue={_safe_queue_qsize(output_queue)}"
                    )

        def run_real_batch(batch: List[Image.Image], source: str) -> None:
            nonlocal first, first_real_batch_started, first_real_batch_done, dyn_last_no_batch_ts
            input_frames = len(batch)
            if not first_real_batch_started:
                first_real_batch_started = True
                print(
                    f"{_stage_prefix(boot_t0)} first_real_batch_start "
                    f"source={source} input_frames={input_frames} "
                    f"buffer_remaining={buffer.size()}"
                )

            batch_t0 = time.monotonic()
            out_frames = flashvsr.process_batch(batch)
            batch_elapsed = time.monotonic() - batch_t0
            if not first_real_batch_done:
                first_real_batch_done = True
                print(
                    f"{_stage_prefix(boot_t0)} first_real_batch_done "
                    f"source={source} input_frames={input_frames} "
                    f"output_frames={len(out_frames)} elapsed={batch_elapsed:.3f}s"
                )

            enqueue_output_frames(out_frames, source=source, input_frames=input_frames)
            first = False
            dyn_last_no_batch_ts = time.monotonic()

        # --- Inference-thread CUDA warmup (critical for live latency) ---
        # Main thread warmup does NOT eliminate the first ~10s JIT/cudnn cost on this worker thread.
        tw = getattr(flashvsr, "thread_warmup_passes", 0)
        if tw > 0 and getattr(flashvsr, "warmup_enabled", False):
            _cuda_set_device_for_thread(flashvsr.device)
            nf = max(5, int(bs0))
            dummy = [
                Image.new("RGB", (flashvsr.in_w, flashvsr.in_h), (0, 0, 0))
                for _ in range(nf)
            ]
            print(
                f"[ProcessThread] CUDA warmup on inference thread: frames={nf}, passes={tw} "
                f"(matches first bootstrap batch shape)"
            )
            warmup_t0 = time.monotonic()
            print(
                f"{_stage_prefix(boot_t0)} warmup_start "
                f"scope=infer_thread frames={nf} passes={tw}"
            )
            for pass_idx in range(tw):
                t0 = time.time()
                _ = flashvsr.process_batch(dummy)
                if flashvsr.device != "cpu":
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                print(
                    f"[ProcessThread] Thread warmup pass {pass_idx + 1}/{tw}: "
                    f"{time.time() - t0:.2f}s"
                )
            print(
                f"{_stage_prefix(boot_t0)} warmup_done "
                f"scope=infer_thread frames={nf} passes={tw} "
                f"elapsed={time.monotonic() - warmup_t0:.3f}s"
            )

        # If capture hiccups, the buffer may not reach the requested batch_size.
        # In that case, we wait a short time and then process a smaller valid batch (4n+1),
        # so realtime pushing doesn't freeze for long.
        dyn_wait_s = 0.9
        dyn_last_no_batch_ts = time.monotonic()
        last_wait_log_ts = time.monotonic()

        while True:
            cur_bs = bs0 if first else int(batch_size)
            batch = buffer.get_batch(cur_bs)
            if batch is not None:
                run_real_batch(batch, source="regular")
            else:
                # buffer 不足 cur_bs，尝试动态小 batch 兜底
                now = time.monotonic()
                if now - dyn_last_no_batch_ts >= dyn_wait_s:
                    avail = buffer.size()
                    # choose largest <= avail that satisfies (n % 4 == 1)
                    if avail >= 5:
                        dyn_bs = avail - ((avail - 1) % 4)
                        dyn_bs = max(5, dyn_bs)
                        if dyn_bs <= avail:
                            dyn_last_no_batch_ts = time.monotonic()
                            try:
                                dyn_batch = buffer.get_batch(dyn_bs)
                                if dyn_batch is not None:
                                    print(
                                        f"[ProcessThread] Dynamic batch: requested {cur_bs}, avail {avail}, use {dyn_bs}"
                                    )
                                    run_real_batch(dyn_batch, source="dynamic")
                                    continue
                            except Exception as e:
                                print(f"[ProcessThread] Dynamic batch failed: {e}")
                    elif avail > 0:
                        # 为了避免 avail=1..4 时无 batch 可处理导致画面长时间重复上一帧，
                        # 这里把已有帧 padding 到 5（满足 n%4==1），仍然走完整 SR pipeline。
                        dyn_last_no_batch_ts = time.monotonic()
                        try:
                            dyn_take = avail
                            dyn_batch = buffer.get_batch(dyn_take)
                            if dyn_batch is not None and len(dyn_batch) > 0:
                                if len(dyn_batch) < 5:
                                    pad_last = dyn_batch[-1]
                                    dyn_batch = dyn_batch + [pad_last] * (5 - len(dyn_batch))
                                dyn_bs = len(dyn_batch)
                                print(
                                    f"[ProcessThread] Padded batch: requested {cur_bs}, avail {avail}, use {dyn_bs}"
                                )
                                run_real_batch(dyn_batch, source="padded")
                                continue
                        except Exception as e:
                            print(f"[ProcessThread] Padded batch failed: {e}")
                # 心跳日志：帮助你区分“buffer 没新帧” vs “process_thread 卡在 process_batch”
                now2 = time.monotonic()
                if now2 - last_wait_log_ts >= 2.0:
                    try:
                        qsz = output_queue.qsize()
                        qmax = getattr(output_queue, "maxsize", None)
                    except Exception:
                        qsz = -1
                        qmax = None
                    print(
                        f"[ProcessThread] Waiting: buffer_avail={buffer.size()} need={cur_bs} "
                        f"output_queue={qsz}/{qmax}"
                    )
                    last_wait_log_ts = now2
                time.sleep(0.05)
    except Exception as e:
        print(f"[ProcessThread] Error: {e}")


def push_thread(
    pusher: RTMPPush,
    output_queue: "queue.Queue[OutputFrame]",
    boot_t0: Optional[float] = None,
) -> None:
    """
    推送线程：当队列短暂为空时，重复上一帧，避免 ffmpeg 因断流输出噪点。
    """
    last_frame: Optional[OutputFrame] = None
    pushed = 0
    first_real_output_pushed = False
    pusher_started = False
    try:
        # 保持稳定输入节奏，避免 RTMP 链路因短暂停止而断开。
        # 在拿到第一帧真实 SR 输出之前，不启动 ffmpeg，避免提前推黑帧导致假在线/异常断链。
        # 启动后如果 output_queue 暂时为空，再按 fps 持续向 ffmpeg 写上一帧。
        frame_interval = 1.0 / max(float(getattr(pusher, "fps", 10.0)), 1e-6)
        next_t = time.monotonic()
        print("[PushThread] Waiting for first SR frame before starting RTMP push ...")
        while True:
            if not pusher_started:
                try:
                    frame = output_queue.get(timeout=0.5)
                    last_frame = frame
                except queue.Empty:
                    continue
                try:
                    pusher.start()
                    pusher_started = True
                    print(f"{_stage_prefix(boot_t0)} pusher_start first_sr_frame_ready=1")
                except Exception as e:
                    print(f"[PushThread] Initial start failed: {e}")
                    time.sleep(1.0)
                    continue
                next_t = time.monotonic()
                got_real_frame = True
            else:
                # 推流进程可能因 RTMP 断开而退出；这里做自动重连
                if not pusher.is_alive():
                    print("[PushThread] Pusher not alive, restarting ...")
                    try:
                        pusher.restart(backoff_s=1.0)
                    except Exception as e:
                        print(f"[PushThread] Restart failed: {e}")
                        time.sleep(1.0)
                        next_t = time.monotonic()
                        continue

                # 非阻塞地拿到最新帧；拿不到则重复上一帧
                got_real_frame = False
                try:
                    frame = output_queue.get_nowait()
                    last_frame = frame
                    got_real_frame = True
                except queue.Empty:
                    if last_frame is None:
                        time.sleep(0.01)
                        continue
                    frame = last_frame

            ok = pusher.write_frame(frame)
            if not ok:
                print("[PushThread] write_frame failed, restarting pusher ...")
                try:
                    pusher.restart(backoff_s=1.0)
                except Exception as e:
                    print(f"[PushThread] Restart failed: {e}")
                    time.sleep(1.0)
                next_t = time.monotonic()
                continue

            pushed += 1
            if got_real_frame and not first_real_output_pushed:
                first_real_output_pushed = True
                print(
                    f"{_stage_prefix(boot_t0)} first_output_pushed "
                    f"pushed_count={pushed} output_queue={_safe_queue_qsize(output_queue)}"
                )
            if pushed % 30 == 0:
                print(f"[PushThread] Pushed {pushed} frames")

            next_t += frame_interval
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
    except Exception as e:
        print(f"[PushThread] Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlashVSR-Pro Livestream Super-Resolution (RTMP → RTMP)"
    )
    parser.add_argument(
        "--input-rtmp",
        type=str,
        default="rtmp://47.79.124.13:30501/live/original_stream",
        help="输入 RTMP 地址（来自 OBS -> SRS），如 rtmp://127.0.0.1:1935/live/src",
    )
    parser.add_argument(
        "--output-rtmp",
        type=str,
        default="rtmp://47.79.124.13:30501/live/sr_stream",
        help="输出 RTMP 地址（推回 SRS），如 rtmp://127.0.0.1:1935/live/sr",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=640,
        help="输入流分辨率宽度（需与 OBS 设置一致）",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=360,
        help="输入流分辨率高度（需与 OBS 设置一致）",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="帧率（需与 OBS 设置大致一致）"
    )

    # FlashVSR 推理参数（对齐 infer.py）
    parser.add_argument(
        "--mode",
        type=str,
        default="tiny",
        choices=["full", "tiny", "tiny-long"],
        help="FlashVSR 模式，推荐 tiny-long",
    )
    parser.add_argument(
        "--tile-dit",
        action="store_true",
        help="对 DiT 启用分块推理，降低显存（推荐开启）",
    )
    parser.add_argument(
        "--tile-vae",
        action="store_true",
        help="对 VAE 解码启用分块（长视频 / 大分辨率时推荐开启）",
    )
    parser.add_argument(
        "--tile-size", type=int, default=256, help="DiT 分块大小（像素）"
    )
    parser.add_argument(
        "--overlap", type=int, default=24, help="分块重叠大小（像素）"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="超分倍率（理论上 4.0 质量最佳，2.0 更易实时）",
    )
    parser.add_argument(
        "--sparse-ratio",
        type=float,
        default=2.0,
        help="稀疏注意力比例，越小越快但略降质",
    )
    parser.add_argument(
        "--kv-ratio",
        type=float,
        default=3.0,
        help="KV cache 比例",
    )
    parser.add_argument(
        "--local-range",
        type=int,
        default=11,
        help="局部注意力范围",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="随机种子（可固定风格）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda 或 cpu，推荐 cuda",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="计算精度，推荐 bf16",
    )

    # 直播相关参数
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="每次送入 FlashVSR 的帧数（8n+1 更稳定，如 9, 17）",
    )
    parser.add_argument(
        "--bootstrap-batch-size",
        type=int,
        default=0,
        help="首段用于尽快出画面的 batch-size（0 表示与 batch-size 相同；推荐 9/13/17）",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=300,
        help="输入帧缓冲最大长度",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="保留原始 RTMP 音频（由 ffmpeg 从 input-rtmp 复制）",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="禁用启动预热（会增加首帧时延，但启动更快）",
    )
    parser.add_argument(
        "--warmup-thread-passes",
        type=int,
        default=1,
        help=(
            "在推理工作线程上跑 N 次 dummy 推理（默认 3）。"
            "主线程 warmup 无法消除工作线程首次 CUDA/cuDNN 冷启动，直播首批仍会慢 ~10s；"
            "在线程里预热后首批可与后续 batch 同量级。设为 0 则仅主线程 warmup（旧行为）。"
        ),
    )
    parser.add_argument(
        "--warmup-on-init",
        action="store_true",
        help="主线程也执行 warmup（默认关闭，避免与 --warmup-thread-passes 重复耗 ~15s）",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="预热用的帧数（默认=0 表示使用 bootstrap-batch-size 或 batch-size；建议与首批��一致）",
    )
    parser.add_argument(
        "--lq-bootstrap-windows",
        type=int,
        default=7,
        help="首段 LQ 特征预取窗口数（默认 3 适合实时场景；7 对应 25 帧但首帧慢）",
    )
    parser.add_argument(
        "--color-fix",
        action="store_true",
        help="启用颜色校正（提升观感但会增加时延；直播默认关闭）",
    )
    parser.add_argument(
        "--low-latency",
        action="store_true",
        help="低延迟模式：禁用 CPU offload/VRAM 管理，模型常驻 GPU（推荐直播开启）",
    )
    parser.add_argument(
        "--disable-low-latency",
        action="store_true",
        help="禁用低延迟模式（回退到省显存策略，可能增加 batch 时延）",
    )
    parser.add_argument(
        "--profile-batches",
        type=int,
        default=0,
        help="打印前 N 个 batch 的耗时拆分（预处理/推理/回传），用于分析首帧慢在哪里；0=关闭",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("FlashVSR-Pro Livestream Super-Resolution")
    print("=" * 80)
    print(f"Input RTMP : {args.input_rtmp}")
    print(f"Output RTMP: {args.output_rtmp}")
    print(f"Input Size : {args.input_width}x{args.input_height} @ {args.fps}fps")
    print(f"Mode       : {args.mode}")
    print(
        f"Scale      : {args.scale}x, tile-dit={args.tile_dit}, "
        f"tile-vae={args.tile_vae}, keep-audio={args.keep_audio}"
    )
    print("=" * 80)

    # 初始化缓冲与队列
    boot_t0 = time.monotonic()
    buffer = StreamBuffer(max_size=args.buffer_size)
    output_queue: "queue.Queue[OutputFrame]" = queue.Queue(maxsize=100)

    # 初始化 RTMP 拉流 / 推流
    capture = RTMPCapture(
        args.input_rtmp,
        fps=args.fps,
        width=args.input_width,
        height=args.input_height,
    )

    # 根据 scale 计算输出尺寸（对齐 infer.py 的对齐策略）
    sW, sH, tW, tH = compute_scaled_and_target_dims(
        args.input_width, args.input_height, scale=args.scale, multiple=128
    )
    out_w, out_h = sW, sH

    pusher = RTMPPush(
        args.output_rtmp,
        fps=args.fps,
        width=out_w,
        height=out_h,
        keep_audio=args.keep_audio,
        audio_source_url=args.input_rtmp if args.keep_audio else None,
    )

    # 初始化 FlashVSR 实时推理器
    try:
        # 确定 warmup 帧数：优先使用 warmup-frames，否则使用首批次大小
        bootstrap_bs = args.bootstrap_batch_size if args.bootstrap_batch_size > 0 else args.batch_size
        warmup_frames = args.warmup_frames if args.warmup_frames > 0 else bootstrap_bs

        if args.no_warmup:
            warmup_on_init = False
            thread_warmup_passes = 0
            print("Warmup     : off (--no-warmup)")
        else:
            thread_warmup_passes = max(0, int(args.warmup_thread_passes))
            warmup_on_init = bool(args.warmup_on_init)
            if thread_warmup_passes == 0:
                warmup_on_init = True  # legacy: only main-thread warmup
            print(
                f"Warmup     : main-thread={warmup_on_init}, "
                f"infer-thread-passes={thread_warmup_passes} "
                "(GPU 冷启动在工作线程；infer-thread>0 可避免首播 batch 再付 ~10s)"
            )

        flashvsr = FlashVSRRealtime(
            mode=args.mode,
            tile_dit=args.tile_dit,
            tile_vae=args.tile_vae,
            tile_size=args.tile_size,
            overlap=args.overlap,
            device=args.device,
            dtype=args.dtype,
            scale=args.scale,
            input_width=args.input_width,
            input_height=args.input_height,
            sparse_ratio=args.sparse_ratio,
            kv_ratio=args.kv_ratio,
            local_range=args.local_range,
            seed=args.seed,
            warmup=not args.no_warmup,
            warmup_frames=warmup_frames,
            lq_bootstrap_windows=args.lq_bootstrap_windows,
            color_fix=args.color_fix,
            low_latency=(not args.disable_low_latency),
            profile_batches=args.profile_batches,
            warmup_on_init=warmup_on_init,
            thread_warmup_passes=thread_warmup_passes,
        )
    except Exception as e:
        print(f"[Main] Failed to init FlashVSR: {e}")
        return

    # 启动 RTMP 拉流 / 推流
    try:
        capture.start()
    except Exception as e:
        print(f"[Main] Failed to start RTMP IO: {e}")
        return

    # 启动 3 个工作线程
    t_cap = threading.Thread(
        target=capture_thread, args=(capture, buffer), daemon=True
    )
    t_proc = threading.Thread(
        target=process_thread,
        args=(
            flashvsr,
            buffer,
            output_queue,
            args.batch_size,
            args.bootstrap_batch_size if args.bootstrap_batch_size > 0 else None,
            boot_t0,
        ),
        daemon=True,
    )
    t_push = threading.Thread(
        target=push_thread, args=(pusher, output_queue, boot_t0), daemon=True
    )

    t_cap.start()
    t_proc.start()
    t_push.start()

    # 主线程仅负责阻塞与 Ctrl-C 退出
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt, shutting down ...")
    finally:
        try:
            capture.stop()
        except Exception:
            pass
        try:
            pusher.stop()
        except Exception:
            pass
        try:
            flashvsr.vae_instance.clean_memory()
        except Exception:
            pass
        torch.cuda.empty_cache()
        print("[Main] Exit.")


if __name__ == "__main__":
    main()

# python livestream_infer.py \
#   --input-rtmp rtmp://localhost:1935/live/original_stream \
#   --output-rtmp rtmp://localhost:1935/live/sr_stream \
#   --input-width 640 --input-height 360 --fps 10 \
#   --mode tiny \
#   --batch-size 17 \
#   --bootstrap-batch-size 9 \
#   --lq-bootstrap-windows 3