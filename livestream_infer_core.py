#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core script for real-time video super-resolution using FlashVSR-Pro.

Only the core elements are retained.
RTMP stream pulling -> FlashVSR batch inference -> RTMP stream pushing
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


LOG_LEVEL = 1
CAPTURE_LOG_INTERVAL = 8
PUSH_LOG_INTERVAL = 8
EMPTY_OUT_QUEUE_WAIT_S = 0.1
OutputFrame = Union[Image.Image, np.ndarray]


def printlog(level, *args, **kwargs):
    if level > LOG_LEVEL:
        return
    print(*args, **kwargs)


def timestamp():
    return time.strftime("%H:%M:%S")

@dataclass
class FramePacket:
    frame_id: int
    frame: OutputFrame


class StreamBuffer:
    "Simple, thread safe frame buffer"

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
        color_fix: bool = False,
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
        self.color_fix = bool(color_fix)

        if self.dtype_str == "fp16":
            self.dtype_torch = torch.float16
        elif self.dtype_str == "bf16":
            self.dtype_torch = torch.bfloat16
        else:
            self.dtype_torch = torch.float32

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False

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
        args.realtime_low_latency = True

        self.pipe, self.vae_instance = init_pipeline(args)

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
            "streaming": True,
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


class RTMPBase:
    '''
    Base class for RTMPCapture and RTMPPush with generic functionality.
    Every subclass is expected to have a process and stream (stdin or stdout)
    attribute.
    '''
    def __init__(self):
        self.process = None
        self.stream = None
        self.name = type(self).__name__   # name of actual subclass
        self._restart_lock = threading.Lock()

    def is_alive(self) -> bool:
        return bool(
            self.process is not None
            and self.process.poll() is None
            and self.stream is not None
        )

    def stop(self) -> None:
        if not self.process or not self.stream:
            raise ValueError(f"[{self.name}] Unable to stop: Invalid state")
            return

        self.stream.close()
        try:
            self.process.terminate()
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            print(f"[{self.name}] Stop: Unable to terminante subprocess")
            try:
                self.process.kill()
            except Exception as exc:
                print(f"[{self.name}] Stop: Unable to kill subprocess:", exc)
        self.process = None

    def restart(self, backoff_s: float = 1.0) -> None:
        with self._restart_lock:
            self.stop()
            if backoff_s > 0:
                time.sleep(backoff_s)
            self.start()


class RTMPCapture(RTMPBase):
    "Manages an ffmpeg subprocess to capture frames into a buffer"

    def __init__(
            self,
            rtmp_url: str,
            fps: int,
            width: int,
            height: int,
            loop: bool,
            add_timestamp: bool
    ):
        super().__init__()
        self.rtmp_url = rtmp_url
        self.fps = fps
        self.width = width
        self.height = height
        self._partial_frame_buf = bytearray()
        self._frame_seq = 0
        self.loop_video = loop
        self.add_timestamp = add_timestamp

    def start(self) -> None:
        print(f"[RTMPCapture] Connecting to {self.rtmp_url}")
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-nostats",
        ]
        if self.loop_video:
            cmd.extend([
                "-stream_loop", "-1",
            ])
        cmd.extend([
            "-r", str(self.fps),
            "-re",
            "-i", self.rtmp_url,
            ])
        if self.add_timestamp:
            opt = " ".join([
                'drawtext=fontfile=DejaVuSans-Bold.ttf:',
                r"text='%{pts\:hms}':",
                "x=0: y=h-(2*lh):",
                "fontcolor=white:",
                "fontsize=45:",
                "box=1:",
                'boxcolor=0x00000000@0.5'
            ])
            cmd.extend([
                "-vf", opt
            ])
        cmd.extend([
            "-c:v", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-framerate", str(self.fps),
            "-f", "rawvideo",
            "-",
        ])

        # FIXME: can we check that the pipe doesn't get full and stalls?
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10 * self.width * self.height * 3,
        )
        self.stream = self.process.stdout
        print("[RTMPCapture] Started")

    def read_frame(self) -> Optional[FramePacket]:
        if not self.is_alive():
            return None

        frame_size = self.width * self.height * 3
        stdout = self.process.stdout

        while len(self._partial_frame_buf) < frame_size:
            if self.process.poll() is not None:
                print('[Read frame] Unable to read. Subprocess terminated')
                return None
            try:
                ready, _, _ = select.select([stdout], [], [], 1.0)
                if not ready:
                    return None
                need = frame_size - len(self._partial_frame_buf)
                chunk = os.read(stdout.fileno(), need)
                if not chunk:
                    return None
            except Exception as exc:
                print('[Read frame] Error while reading:', exc)
                return None
            self._partial_frame_buf.extend(chunk)
        data = bytes(self._partial_frame_buf[:frame_size])
        del self._partial_frame_buf[:frame_size]

        frame = np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3)
        image = Image.fromarray(frame, "RGB")
        self._frame_seq += 1
        return FramePacket(frame_id=self._frame_seq, frame=image)


class RTMPPush(RTMPBase):
    """通过 ffmpeg 将处理后帧推回 RTMP。"""

    def __init__(
        self,
        rtmp_url: str,
        fps: int,
        width: int,
        height: int,
        keep_audio: bool = False,
        audio_source_url: Optional[str] = None,
        add_timestamp: bool = False,
        trace: bool = False
    ):
        super().__init__()
        self.rtmp_url = rtmp_url
        self.fps = fps
        self.width = width
        self.height = height
        self.keep_audio = keep_audio
        self.audio_source_url = audio_source_url
        self.process = None
        self._log_thread = None
        self.add_timestamp = add_timestamp
        self.trace = trace

    def start(self) -> None:
        print(f"[RTMPPush] Pushing to {self.rtmp_url}")
        gop = max(int(self.fps), 1)
        use_audio = self.keep_audio and self.audio_source_url

        cmd = [
                "ffmpeg",
                "-loglevel", "error",
                "-nostats",
                "-f", "rawvideo",
                "-pixel_format", "rgb24",
                "-video_size", f"{self.width}x{self.height}",
                "-framerate", str(self.fps),
                "-i", "pipe:",
        ]
        if self.add_timestamp:
            opt = " ".join([
                'drawtext=fontfile=DejaVuSans-Bold.ttf:',
                r"text='%{pts\:hms}':",
                "x=(w-tw): y=h-(2*lh):",
                "fontcolor=yellow:",
                "fontsize=90:",
                "box=1:",
                'boxcolor=0x00000000@0.5'
            ])
            cmd.extend([
                "-vf", opt
            ])
        if use_audio:
            cmd.extend([
                "-i", self.audio_source_url,
                "-map", "0:v",
                "-map", "1:a",
            ])
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-g", str(gop),
            "-keyint_min", str(gop),
            "-sc_threshold", "0",
            "-bf", "0",
            "-pix_fmt", "yuv420p",
            "-b:v", "6000k",
            "-maxrate", "6000k",
            "-bufsize", "3000k",
        ])
        if use_audio:
            cmd.extend([
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
            ])
        cmd.extend([
            "-flush_packets", "1",
            "-f", "flv",
            "-y",
            self.rtmp_url,
        ])

        # FIXME: can we check that the pipe doesn't get full and stalls?
        env = os.environ
        if self.trace:
            env['FFREPORT'] = 'file=ffmpeg_out_trace.log:level=56'
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=10 * self.width * self.height * 3,
            env=env
        )
        self.stream = self.process.stdin

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
            except Exception as exc:
                print("[RTMPPush ffmpeg] Unable to log ffmpeg errors:", exc)
                pass

        self._log_thread = threading.Thread(target=_log_ffmpeg, daemon=True)
        self._log_thread.start()
        print("[RTMPPush] Started")

    def write_frame(self, frame: OutputFrame) -> bool:
        if not self.is_alive():
            return False

        if isinstance(frame, Image.Image):
            frame = frame.convert("RGB")
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Unexpected frame shape: {arr.shape}")
        self.process.stdin.write(np.ascontiguousarray(arr).tobytes())
        return True


def restart_capture(capture, restarts=5):
    for _ in range(restarts):
        try:
            capture._partial_frame_buf.clear()
            capture.restart(backoff_s=1.0)
        except Exception as exc:
            print("[CaptureThread] Restart failed:", exc)
            time.sleep(0.1)
        else:
            return True
    return False


def capture_thread(capture: RTMPCapture, buffer: StreamBuffer) -> None:
    consecutive_none = 0
    counter = 0
    timer = time.perf_counter()
    while True:
        packet = capture.read_frame()
        if packet is None:
            consecutive_none += 1
            if consecutive_none >= 5:
                print("[CaptureThread] Restarting capture")
                success = restart_capture(capture)
                if not success:
                    print("[CaptureThread] Unable to restart. Stopping thread")
                    break
                consecutive_none = 0
            else:
                time.sleep(1)
            continue

        consecutive_none = 0
        buffer.put(packet)
        counter += 1
        if counter % CAPTURE_LOG_INTERVAL == 0:
            printlog(1, "[CaptureThread] ({}) Captured {} frames in {:.3f} s. Total {} frames".format(
                timestamp(), CAPTURE_LOG_INTERVAL, time.perf_counter() - timer, counter))
            timer = time.perf_counter()


def capture_thread_safe(*args):
    "Safe version of capture_thread that prints exceptions"
    try:
        capture_thread(*args)
    except Exception as exc:
        print(f"[CaptureThread] Error: {exc}")


def process_thread(
    flashvsr: FlashVSRRealtime,
    buffer: StreamBuffer,
    output_queue: "queue.Queue[FramePacket]",
    batch_size: int,
) -> None:
    timer = time.perf_counter()
    counter = 0
    while True:
        batch_packets = buffer.get_batch(batch_size)
        if batch_packets is None:
            time.sleep(0.05)
            continue

        batch = [packet.frame for packet in batch_packets]
        out_frames = flashvsr.process_batch(batch)

        for idx, out_frame in enumerate(out_frames):
            src_packet = batch_packets[min(idx, len(batch_packets) - 1)]
            out_packet = FramePacket(frame_id=src_packet.frame_id, frame=out_frame)
            try:
                output_queue.put_nowait(out_packet)
            except queue.Full:
                print("[ProcessThread] Error: output queue full")

        if len(batch_packets) != len(out_frames):
            print("[ProcessThread] Error: Input/output frame mismatch")

        counter += len(out_frames)
        printlog(1, "[ProcessThread] ({}) Processed {}/{} frames in {:.3f} s. Total {} frames".format(
            timestamp(), len(batch_packets), len(out_frames), time.perf_counter() - timer, counter))
        timer = time.perf_counter()

def process_thread_safe(*args):
    "Safe version of process_thread that prints exceptions"
    try:
        process_thread(*args)
    except Exception as exc:
        print(f"[ProcessThread] Error: {exc}")


def restart_pusher(pusher, restarts=5):
    for _ in range(restarts):
        try:
            pusher.restart(backoff_s=1.0)
        except Exception as exc:
            print(f"[PushThread] Restart failed: {exc}")
            time.sleep(1.0)
        else:
            return True
    return False


def push_thread(pusher: RTMPPush, output_queue: "queue.Queue[FramePacket]") -> None:
    last_packet = None
    pushed = 0
    timer = time.perf_counter()
    frame_interval = 1.0 / max(float(pusher.fps), 1e-6)

    print("[PushThread] Waiting for first output frame")
    try:
        pusher.start()
    except Exception as exc:
        print(f"[PushThread] Initial start failed: {exc}")

    while True:
        next_t = time.monotonic() + frame_interval
        if not pusher.is_alive():
            print("[PushThread] Restarting pusher")
            success = restart_pusher(pusher)
            if not success:
                print("[PushThread] Unable to restart. Stopping thread")
                break
        try:
            last_packet = output_queue.get_nowait()
        except queue.Empty:
            # print("[PushThread] Empty output queue")
            time.sleep(EMPTY_OUT_QUEUE_WAIT_S)
            continue

        try:
            success = pusher.write_frame(last_packet.frame)
        except Exception as exc:
            print("[PushThread] Error while writing frame:", exc)
            continue

        pushed += 1
        if pushed % PUSH_LOG_INTERVAL == 0:
            printlog(1, "[PushThread] ({}) Pushed {} frames in {:.3f} s. Total {} frames".format(
                timestamp(), PUSH_LOG_INTERVAL, time.perf_counter() - timer, pushed))
            timer = time.perf_counter()

        sleep_s = next_t - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)


def push_tread_safe(*args):
    try:
        push_thread(*args)
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
        help="Input RTMP address (e.g. OBS -> SRS)",
    )
    parser.add_argument(
        "--output-rtmp",
        type=str,
        default="rtmp://47.79.124.13:30501/live/sr_stream",
        help="Output RTMP address (e.g. push back to SRS)",
    )
    parser.add_argument("--input-width", type=int, default=640, help="Input Width")
    parser.add_argument("--input-height", type=int, default=360, help="Ouput widht")
    parser.add_argument("--fps", type=int, default=10, help="Input/output frame rate")

    parser.add_argument("--mode", type=str, default="tiny",
                        choices=["full", "tiny", "tiny-long"], help="FlashVSR mode")
    parser.add_argument("--tile-dit", action="store_true",
                        help="Enable DiT tiling to reduce memory usage")
    parser.add_argument("--tile-vae", action="store_true",
                        help="Enable VAE tiling to reduce memory usage")
    parser.add_argument("--tile-size", type=int, default=256,
                        help="Tile size for DiT/VAE tiling")
    parser.add_argument("--overlap", type=int, default=24,
                        help="Amount of overlap in DiT/VAE tiling")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="Super resolution scaling factor")
    parser.add_argument("--sparse-ratio", type=float, default=2.0,
                        help="Sparsity ratio for attention layers")
    parser.add_argument("--kv-ratio", type=float, default=3.0,
                        help="KV cache ratio")
    parser.add_argument("--local-range", type=int, default=11,
                        help="Local attention range")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for random number generator")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Inference device")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"], help="Data type")
    parser.add_argument("--batch-size", type=int, default=24,
                        help="FlashVSR batch size, in frames. Needs to be multiple of 8.")
    parser.add_argument("--buffer-size", type=int, default=300,
                        help="Input buffer size")
    parser.add_argument("--keep-audio", action="store_true",
                        help="Retain original audio")
    parser.add_argument("--color-fix", action="store_true",
                        help="Enable color correction in FlashVSR")
    parser.add_argument("--loop", action="store_true", help='Loop input video')
    parser.add_argument("--itstamp", action="store_true",
                        help="Add timestamp to output video")
    parser.add_argument("--otstamp", action="store_true",
                        help="Add timestamp to output video")
    parser.add_argument("--otrace", action="store_true",
                        help="Save trace of output ffmpeg process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
    output_queue: "queue.Queue[FramePacket]" = queue.Queue(maxsize=1000)

    capture = RTMPCapture(
        args.input_rtmp,
        fps=args.fps,
        width=args.input_width,
        height=args.input_height,
        loop=args.loop,
        add_timestamp=args.itstamp
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
        add_timestamp=args.otstamp,
        trace=args.otrace
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
            color_fix=args.color_fix,
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
        target=process_thread_safe,
        args=(
            flashvsr,
            buffer,
            output_queue,
            args.batch_size,
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