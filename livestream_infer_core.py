#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlashVSR-Pro 实时直播超分核心脚本。

仅保留核心链路：
RTMP 拉流 -> FlashVSR 批量推理 -> RTMP 推流
"""

import os
import sys
import time
import argparse
import threading
import queue
import select
import subprocess
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from PIL import Image
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from infer import (  # type: ignore
    init_pipeline,
    compute_scaled_and_target_dims,
    process_batch_gpu,
    TILE_AVAILABLE,
    apply_tiled_inference_simple,
)


OutputFrame = Union[Image.Image, np.ndarray]


@dataclass
class FramePacket:
    frame_id: int
    frame: OutputFrame


class StreamBuffer:
    """简单线程安全帧缓冲。"""

    def __init__(self, max_size: int = 300):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.frame_count = 0

    def put(self, packet: FramePacket) -> None:
        with self.lock:
            self.buffer.append(packet)
            self.frame_count += 1

    def get_batch(self, batch_size: int) -> Optional[List[FramePacket]]:
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            return [self.buffer.popleft() for _ in range(batch_size)]

    def size(self) -> int:
        with self.lock:
            return len(self.buffer)


def tensor2video_fast(frames: torch.Tensor) -> List[np.ndarray]:
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
    return [np.ascontiguousarray(frame) for frame in frames]


class FlashVSRRealtime:
    """复用 `infer.py` 的模型加载和批量推理逻辑。"""

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
        lq_bootstrap_windows: int = 3,
        color_fix: bool = False,
        low_latency: bool = True,
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
        self.lq_bootstrap_windows = lq_bootstrap_windows
        self.color_fix = bool(color_fix)
        self.low_latency = low_latency

        if self.dtype_str == "fp16":
            self.dtype_torch = torch.float16
        elif self.dtype_str == "bf16":
            self.dtype_torch = torch.bfloat16
        else:
            self.dtype_torch = torch.float32

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        self.sW, self.sH, self.tW, self.tH = compute_scaled_and_target_dims(
            self.in_w, self.in_h, scale=self.scale, multiple=128
        )
        self.exact_w = self.sW
        self.exact_h = self.sH

        print(
            f"[FlashVSRRealtime] Input {self.in_w}x{self.in_h}, "
            f"scale {self.scale}x -> output {self.exact_w}x{self.exact_h}"
        )

        args = argparse.Namespace()
        args.mode = self.mode
        args.device = self.device
        args.dtype = self.dtype_str
        args.tile_vae = self.tile_vae
        args.tile_size = self.tile_size
        args.overlap = self.overlap
        args.realtime_low_latency = bool(self.low_latency)

        self.pipe, self.vae_instance = init_pipeline(args)

        if self.device == "cuda":
            try:
                torch.cuda.set_per_process_memory_fraction(1.0)
            except Exception:
                pass

    def process_batch(self, frames: List[Image.Image]) -> List[OutputFrame]:
        if not frames:
            return []

        num_frames = len(frames)
        if num_frames < 5:
            return frames

        batch_arr = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames]
        lq_batch = process_batch_gpu(
            batch_arr,
            sH=self.sH,
            sW=self.sW,
            tH=self.tH,
            tW=self.tW,
            dtype=self.dtype_torch,
            device=self.device,
        )
        lq_video = lq_batch.permute(1, 0, 2, 3).unsqueeze(0)

        pipeline_kwargs = {
            "prompt": "",
            "negative_prompt": "",
            "cfg_scale": 1.0,
            "num_inference_steps": 1,
            "seed": self.seed,
            "LQ_video": lq_video,
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

        if self.tile_vae:
            vae_tile_size_latent = max(32, self.tile_size // 8)
            vae_overlap_latent = max(4, self.overlap // 8)
            pipeline_kwargs["tiled"] = True
            pipeline_kwargs["tile_size"] = (
                vae_tile_size_latent,
                vae_tile_size_latent,
            )
            pipeline_kwargs["tile_stride"] = (
                vae_tile_size_latent - vae_overlap_latent,
                vae_tile_size_latent - vae_overlap_latent,
            )

        with torch.inference_mode():
            if self.tile_dit and TILE_AVAILABLE:
                tile_kwargs = dict(pipeline_kwargs)
                tile_kwargs.pop("LQ_video", None)
                vae_tile_size_tuple = tile_kwargs.pop("tile_size", None)
                video = apply_tiled_inference_simple(
                    self.pipe,
                    lq_video,
                    tile_size=self.tile_size,
                    overlap=self.overlap,
                    tile_size_vae=vae_tile_size_tuple,
                    **tile_kwargs,
                )
            else:
                video = self.pipe(**pipeline_kwargs)

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

        frames_np = tensor2video_fast(video)
        if len(frames_np) > num_frames:
            frames_np = frames_np[:num_frames]
        elif len(frames_np) < num_frames and frames_np:
            frames_np.extend([frames_np[-1]] * (num_frames - len(frames_np)))
        return frames_np


class RTMPCapture:
    """通过 ffmpeg 拉取原始帧。"""

    def __init__(self, rtmp_url: str, fps: int, width: int, height: int):
        self.rtmp_url = rtmp_url
        self.fps = fps
        self.width = width
        self.height = height
        self.process: Optional[subprocess.Popen] = None
        self._restart_lock = threading.Lock()
        self._partial_frame_buf = bytearray()
        self._frame_seq = 0

    def start(self) -> None:
        print(f"[RTMPCapture] Connecting to {self.rtmp_url}")
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
            "-f",
            "rawvideo",
            "-",
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10 * self.width * self.height * 3,
        )
        print("[RTMPCapture] Started")

    def read_frame(self) -> Optional[FramePacket]:
        if self.process is None or self.process.stdout is None:
            return None
        if self.process.poll() is not None:
            return None

        frame_size = self.width * self.height * 3
        stdout = self.process.stdout

        while len(self._partial_frame_buf) < frame_size:
            if self.process.poll() is not None:
                return None
            try:
                ready, _, _ = select.select([stdout], [], [], 1.0)
            except Exception:
                return None
            if not ready:
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
        image = Image.fromarray(frame, "RGB")
        self._frame_seq += 1
        return FramePacket(frame_id=self._frame_seq, frame=image)

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
    """通过 ffmpeg 将处理后帧推回 RTMP。"""

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
        print(f"[RTMPPush] Pushing to {self.rtmp_url}")
        gop = max(int(self.fps), 1)

        if self.keep_audio and self.audio_source_url:
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
        print("[RTMPPush] Started")

    def is_alive(self) -> bool:
        return bool(
            self.process is not None
            and self.process.poll() is None
            and self.process.stdin is not None
        )

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
            self.process.stdin.write(np.ascontiguousarray(arr).tobytes())
            return True
        except Exception as exc:
            print(f"[RTMPPush] write_frame error: {exc}")
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
            packet = capture.read_frame()
            if packet is None:
                consecutive_none += 1
                if consecutive_none >= 5:
                    consecutive_none = 0
                    print("[CaptureThread] Restarting capture")
                    try:
                        capture._partial_frame_buf.clear()
                        capture.restart(backoff_s=1.0)
                    except Exception as exc:
                        print(f"[CaptureThread] Restart failed: {exc}")
                        time.sleep(2.0)
                else:
                    time.sleep(0.05)
                continue

            consecutive_none = 0
            buffer.put(packet)
    except Exception as exc:
        print(f"[CaptureThread] Error: {exc}")


def _cuda_set_device_for_thread(device: str) -> None:
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


def process_thread(
    flashvsr: FlashVSRRealtime,
    buffer: StreamBuffer,
    output_queue: "queue.Queue[FramePacket]",
    batch_size: int,
    bootstrap_batch_size: Optional[int] = None,
    dyn_wait_s: float = 0.9,
) -> None:
    try:
        _cuda_set_device_for_thread(flashvsr.device)

        first = True
        bs0 = int(bootstrap_batch_size) if bootstrap_batch_size is not None else int(batch_size)
        bs0 = max(5, bs0)
        dyn_last_no_batch_ts = time.monotonic()

        dummy = [
            Image.new("RGB", (flashvsr.in_w, flashvsr.in_h), (0, 0, 0))
            for _ in range(bs0)
        ]
        print(
            f"[ProcessThread] Warmup on inference thread "
            f"({len(dummy)} frames, matches first batch shape)"
        )
        _ = flashvsr.process_batch(dummy)
        if flashvsr.device != "cpu":
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        print("[ProcessThread] Inference-thread warmup finished")

        while True:
            cur_bs = bs0 if first else int(batch_size)
            batch_packets = buffer.get_batch(cur_bs)

            if batch_packets is not None:
                batch = [packet.frame for packet in batch_packets]
                out_frames = flashvsr.process_batch(batch)
                output_count = len(out_frames)

                for idx, out_frame in enumerate(out_frames):
                    src_packet = batch_packets[min(idx, len(batch_packets) - 1)]
                    out_packet = FramePacket(frame_id=src_packet.frame_id, frame=out_frame)
                    try:
                        output_queue.put_nowait(out_packet)
                    except queue.Full:
                        try:
                            _ = output_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            output_queue.put_nowait(out_packet)
                        except queue.Full:
                            pass

                if first:
                    print(
                        f"[ProcessThread] First batch done: "
                        f"input={len(batch_packets)} output={output_count}"
                    )
                first = False
                dyn_last_no_batch_ts = time.monotonic()
                continue

            now = time.monotonic()
            if now - dyn_last_no_batch_ts >= dyn_wait_s:
                avail = buffer.size()
                if avail >= 5:
                    dyn_bs = avail - ((avail - 1) % 4)
                    dyn_bs = max(5, dyn_bs)
                    dyn_batch = buffer.get_batch(dyn_bs)
                    if dyn_batch is not None:
                        out_frames = flashvsr.process_batch([packet.frame for packet in dyn_batch])
                        for idx, out_frame in enumerate(out_frames):
                            src_packet = dyn_batch[min(idx, len(dyn_batch) - 1)]
                            out_packet = FramePacket(frame_id=src_packet.frame_id, frame=out_frame)
                            try:
                                output_queue.put_nowait(out_packet)
                            except queue.Full:
                                try:
                                    _ = output_queue.get_nowait()
                                except queue.Empty:
                                    pass
                                try:
                                    output_queue.put_nowait(out_packet)
                                except queue.Full:
                                    pass
                        dyn_last_no_batch_ts = time.monotonic()
                        continue

                if 0 < avail < 5:
                    dyn_batch = buffer.get_batch(avail)
                    if dyn_batch:
                        while len(dyn_batch) < 5:
                            dyn_batch.append(dyn_batch[-1])
                        out_frames = flashvsr.process_batch([packet.frame for packet in dyn_batch])
                        for idx, out_frame in enumerate(out_frames):
                            src_packet = dyn_batch[min(idx, len(dyn_batch) - 1)]
                            out_packet = FramePacket(frame_id=src_packet.frame_id, frame=out_frame)
                            try:
                                output_queue.put_nowait(out_packet)
                            except queue.Full:
                                try:
                                    _ = output_queue.get_nowait()
                                except queue.Empty:
                                    pass
                                try:
                                    output_queue.put_nowait(out_packet)
                                except queue.Full:
                                    pass
                        dyn_last_no_batch_ts = time.monotonic()
                        continue

            time.sleep(0.05)
    except Exception as exc:
        print(f"[ProcessThread] Error: {exc}")


def push_thread(pusher: RTMPPush, output_queue: "queue.Queue[FramePacket]") -> None:
    last_packet: Optional[FramePacket] = None
    pushed = 0
    pusher_started = False

    try:
        frame_interval = 1.0 / max(float(pusher.fps), 1e-6)
        next_t = time.monotonic()
        print("[PushThread] Waiting for first output frame")

        while True:
            if not pusher_started:
                try:
                    packet = output_queue.get(timeout=0.5)
                    last_packet = packet
                except queue.Empty:
                    continue
                try:
                    pusher.start()
                    pusher_started = True
                except Exception as exc:
                    print(f"[PushThread] Initial start failed: {exc}")
                    time.sleep(1.0)
                    continue
                next_t = time.monotonic()
            else:
                if not pusher.is_alive():
                    print("[PushThread] Restarting pusher")
                    try:
                        pusher.restart(backoff_s=1.0)
                    except Exception as exc:
                        print(f"[PushThread] Restart failed: {exc}")
                        time.sleep(1.0)
                        next_t = time.monotonic()
                        continue

                try:
                    last_packet = output_queue.get_nowait()
                except queue.Empty:
                    if last_packet is None:
                        time.sleep(0.01)
                        continue

            assert last_packet is not None
            if not pusher.write_frame(last_packet.frame):
                print("[PushThread] write_frame failed, restarting pusher")
                try:
                    pusher.restart(backoff_s=1.0)
                except Exception as exc:
                    print(f"[PushThread] Restart failed: {exc}")
                    time.sleep(1.0)
                next_t = time.monotonic()
                continue

            pushed += 1
            if pushed % 60 == 0:
                print(f"[PushThread] Pushed {pushed} frames")

            next_t += frame_interval
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
    except Exception as exc:
        print(f"[PushThread] Error: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlashVSR-Pro Livestream Super-Resolution Core (RTMP -> RTMP)"
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
    parser.add_argument("--input-width", type=int, default=640, help="输入宽度")
    parser.add_argument("--input-height", type=int, default=360, help="输入高度")
    parser.add_argument("--fps", type=int, default=10, help="输入/输出帧率")

    parser.add_argument(
        "--mode",
        type=str,
        default="tiny",
        choices=["full", "tiny", "tiny-long"],
        help="FlashVSR 模式",
    )
    parser.add_argument("--tile-dit", action="store_true", help="启用 DiT 分块")
    parser.add_argument("--tile-vae", action="store_true", help="启用 VAE 分块")
    parser.add_argument("--tile-size", type=int, default=256, help="分块大小")
    parser.add_argument("--overlap", type=int, default=24, help="分块重叠")

    parser.add_argument("--scale", type=float, default=2.0, help="超分倍率")
    parser.add_argument("--sparse-ratio", type=float, default=2.0, help="稀疏注意力比例")
    parser.add_argument("--kv-ratio", type=float, default=3.0, help="KV cache 比例")
    parser.add_argument("--local-range", type=int, default=11, help="局部注意力范围")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="计算精度",
    )

    parser.add_argument("--batch-size", type=int, default=25, help="常规 batch 大小")
    parser.add_argument(
        "--bootstrap-batch-size",
        type=int,
        default=0,
        help="首批 batch 大小，0 表示与 batch-size 相同",
    )
    parser.add_argument("--buffer-size", type=int, default=300, help="输入缓冲大小")
    parser.add_argument(
        "--dyn-wait-s",
        type=float,
        default=0.9,
        help="batch 不足时，等待多久后降级为动态 batch",
    )
    parser.add_argument("--keep-audio", action="store_true", help="保留原始音频")
    parser.add_argument(
        "--lq-bootstrap-windows",
        type=int,
        default=7,
        help="LQ 特征预取窗口数",
    )
    parser.add_argument("--color-fix", action="store_true", help="启用颜色校正")
    parser.add_argument(
        "--disable-low-latency",
        action="store_true",
        help="禁用低延迟常驻 GPU 策略",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bootstrap_bs = (
        args.bootstrap_batch_size if args.bootstrap_batch_size > 0 else args.batch_size
    )

    print("=" * 80)
    print("FlashVSR-Pro Livestream Core")
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

    buffer = StreamBuffer(max_size=args.buffer_size)
    output_queue: "queue.Queue[FramePacket]" = queue.Queue(maxsize=100)

    capture = RTMPCapture(
        args.input_rtmp,
        fps=args.fps,
        width=args.input_width,
        height=args.input_height,
    )

    out_w, out_h, _, _ = compute_scaled_and_target_dims(
        args.input_width, args.input_height, scale=args.scale, multiple=128
    )
    pusher = RTMPPush(
        args.output_rtmp,
        fps=args.fps,
        width=out_w,
        height=out_h,
        keep_audio=args.keep_audio,
        audio_source_url=args.input_rtmp if args.keep_audio else None,
    )

    try:
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
            lq_bootstrap_windows=args.lq_bootstrap_windows,
            color_fix=args.color_fix,
            low_latency=(not args.disable_low_latency),
        )
    except Exception as exc:
        print(f"[Main] Failed to init FlashVSR: {exc}")
        return

    try:
        capture.start()
    except Exception as exc:
        print(f"[Main] Failed to start capture: {exc}")
        return

    t_cap = threading.Thread(target=capture_thread, args=(capture, buffer), daemon=True)
    t_proc = threading.Thread(
        target=process_thread,
        args=(
            flashvsr,
            buffer,
            output_queue,
            args.batch_size,
            args.bootstrap_batch_size if args.bootstrap_batch_size > 0 else None,
            float(args.dyn_wait_s),
        ),
        daemon=True,
    )
    t_push = threading.Thread(
        target=push_thread,
        args=(pusher, output_queue),
        daemon=True,
    )

    t_cap.start()
    t_proc.start()
    t_push.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt, shutting down...")
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Main] Exit")


if __name__ == "__main__":
    main()
