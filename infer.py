#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import time
import argparse
import warnings
import shutil
import subprocess
from contextlib import redirect_stdout
import io

import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange

# Add project path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from diffsynth import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
from utils.utils import Causal_LQ4x_Proj
from utils.utils import Causal_LQ4x_Proj_simple as Causal_LQ4x_Proj
from utils.TCDecoder import build_tcdecoder
from utils import vae_manager

# Optional audio utilities
try:
    from utils.audio_utils import copy_video_with_audio, has_audio_stream
    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
    warnings.warn(f"Audio utilities not available: {e}")

    # Provide simple fallback functions
    def has_audio_stream(path):
        return False

    def copy_video_with_audio(original_video_path, processed_video_path, output_path):
        import shutil
        shutil.copy2(processed_video_path, output_path)
        return True

# Tile utilities
try:
    from utils.tile_utils import calculate_tile_coords, apply_tiled_inference_simple
    TILE_AVAILABLE = True
except ImportError as e:
    TILE_AVAILABLE = False
    warnings.warn(f"Tile utilities not available: {e}")

    # Provide simple fallback functions
    def calculate_tile_coords(height, width, tile_size, overlap):
        return [(0, 0, width, height)]

    def apply_tiled_inference_simple(pipeline, LQ_video, tile_size=256, overlap=24, **pipeline_kwargs):
        # No tiling, call pipeline directly
        return pipeline(**pipeline_kwargs)

def check_nvenc_support():
    """
    Check if NVIDIA NVENC encoder is available in BOTH:
    1. imageio_ffmpeg's bundled ffmpeg (used by imageio.get_writer)
    2. System ffmpeg (used by subprocess fallback)
    
    Only returns True if both support h264_nvenc.
    """
    try:
        # Check imageio_ffmpeg's bundled ffmpeg
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        result_imageio = subprocess.run(
            [ffmpeg_exe, '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=5
        )
        imageio_has_nvenc = 'h264_nvenc' in result_imageio.stdout
        
        # Check system ffmpeg
        result_system = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=5
        )
        system_has_nvenc = 'h264_nvenc' in result_system.stdout
        
        # Both must support NVENC for consistent behavior
        if imageio_has_nvenc and system_has_nvenc:
            return True
        elif system_has_nvenc and not imageio_has_nvenc:
            return False
        else:
            return False
    except Exception:
        return False

# Check NVENC support once at module level
# NVENC_AVAILABLE = check_nvenc_support()
NVENC_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description="FlashVSR-Pro Inference Script")
    
    # Basic parameters
    parser.add_argument("-i", "--input", type=str, default="./inputs/example0.mp4",
                       help="Path to input video file or folder of images")
    parser.add_argument("-o", "--output", type=str, default="./results",
                       help="Output directory or file path")
    parser.add_argument("--mode", type=str, default='tiny', 
                       choices=["full", "tiny", "tiny-long"],
                       help="Inference mode: full (with Wan VAE), tiny (with TCDecoder), tiny-long (for long videos)")
    
    # Tiling parameters
    parser.add_argument("--tile-dit", action="store_true",
                       help="Enable tiled inference for DiT (reduces VRAM usage)")
    parser.add_argument("--tile-vae", action="store_true",
                       help="Enable tiled decoding for VAE (only for full mode)")
    parser.add_argument("--tile-size", type=int, default=256,
                       help="Tile size for tiled inference")
    parser.add_argument("--overlap", type=int, default=24,
                       help="Overlap size between tiles")
    
    # Audio parameters
    parser.add_argument("--keep-audio", action="store_true",
                       help="Keep audio from input video (if exists)")
    
    # Inference parameters
    parser.add_argument("--scale", type=float, default=2.0,
                       help="Upscale factor")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--sparse-ratio", type=float, default=2.0,
                       help="Sparse attention ratio (1.5=faster, 2.0=stable)")
    parser.add_argument("--kv-ratio", type=float, default=3.0,
                       help="KV cache ratio")
    parser.add_argument("--local-range", type=int, default=11,
                       help="Local attention range (9=sharper, 11=stable)")
    parser.add_argument("--color-fix", action="store_true",
                       help="Apply color correction")
    parser.add_argument("--fps", type=float, default=None,
                       help="Output FPS (default: match input or 30 for images)")
    parser.add_argument("--quality", type=int, default=10,
                       help="Output video quality (0-10)")
    
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bf16",
                       choices=["fp32", "fp16", "bf16"],
                       help="Data type")
    parser.add_argument(
        "--lq-bootstrap-windows",
        type=int,
        default=4,
        help="首段 LQ 特征预取窗口数（7 对应 25 帧；实时可用 2~4 降低首帧尖峰并支持 batch<25）",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="在正式推理前先跑一轮同分辨率 dry-run 预热，降低首轮 CUDA/cuDNN 冷启动尖峰",
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="打印前处理、模型加载、warmup、LQ bootstrap、DiT、解码、保存等详细耗时",
    )
    
    return parser.parse_args()

def tensor2video(frames: torch.Tensor):
    """Convert tensor to list of Numpy Arrays (uint8)"""
    # Handle optional batch dimension
    if frames.ndim == 5:
        frames = frames.squeeze(0)
    
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    # Optimization: Return numpy arrays directly to avoid costly PIL conversion
    # Pipeline consumers (imageio, ffmpeg pipe) handle numpy arrays efficiently
    frames = [frame for frame in frames]
    return frames

def natural_key(name: str):
    """Natural sort key for filenames"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    """List image files with natural sorting"""
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def largest_8n1_leq(n):
    """Find largest 8n+1 <= n"""
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def is_video(path):
    """Check if path is a video file"""
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv','.webm'))


def save_video_with_audio_piped(frames, output_path, audio_source, fps=30, quality=10):
    """Save frames as video with audio directly using ffmpeg pipe"""
    if not frames: return False
    
    # Handle numpy arrays (H, W, C) vs PIL Images (W, H)
    if hasattr(frames[0], 'shape'):
        h, w = frames[0].shape[:2]
    else:
        w, h = frames[0].size

    # Approximation of quality to CRF/CQ: quality 10 -> crf 3, quality 5 -> crf 13, quality 0 -> crf 23
    # For NVENC, we use -cq (Constant Quality) and -rc vbr
    
    if NVENC_AVAILABLE:
        # NVENC settings - optimized for speed
        vcodec = 'h264_nvenc'
        # Map quality (0-10) to CQ (26-20): quality 10 -> CQ 20, quality 0 -> CQ 26
        cq = int(26 - quality * 0.6)
        encoding_args = [
            '-c:v', vcodec,
            '-preset', 'p1',      # p1 is fastest NVENC preset
            '-rc', 'vbr',
            '-cq', str(cq),
            '-b:v', '0',          # Let VBR handle bitrate
        ]
    else:
        # CPU x264 settings - optimized for speed and reasonable file size
        vcodec = 'libx264'
        # Map quality (0-10) to CRF (26-20): quality 10 -> CRF 20, quality 0 -> CRF 26
        # CRF 20-22 is visually lossless for most content
        crf = int(26 - quality * 0.6)
        encoding_args = [
            '-c:v', vcodec,
            '-preset', 'veryfast',   # Much faster than 'faster', minimal quality loss
            '-crf', str(crf),
            '-tune', 'film',         # Optimize for high-quality video content
        ]

    cmd = [
        'ffmpeg', '-y',
        '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',          # Input 0: stdin (video)
        '-i', audio_source, # Input 1: audio source file
        '-map', '0:v',      # Map video from input 0
        '-map', '1:a',      # Map audio from input 1
        '-pix_fmt', 'yuv420p', # Ensure compatibility
        *encoding_args,     # Inject codec specific args
        '-c:a', 'aac',      # Re-encode audio to aac for compatibility
        '-b:a', '192k',
        '-shortest',        # Finish when shortest stream ends
        output_path
    ]

    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("Error: ffmpeg not found for piping.")
        return False
        
    for frame in tqdm(frames, desc="Saving"):
        process.stdin.write(np.array(frame).tobytes())
        
    out, err = process.communicate()
    
    if process.returncode != 0:
        print(f"FFmpeg Error: {err.decode('utf-8', errors='ignore')}")
        return False
        
    return True


def save_video(frames, save_path, fps=30, quality=5):
    """Save frames as video"""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    ext = os.path.splitext(save_path)[-1].lower()
    
    # Map common video extensions to imageio formats
    format_map = {
        '.mp4': 'FFMPEG',
        '.avi': 'FFMPEG',
        '.mov': 'FFMPEG',
        '.mkv': 'FFMPEG',
        '.webm': 'FFMPEG',
        '.gif': 'GIF',
    }
    format_name = format_map.get(ext, 'FFMPEG')

    try:
        if NVENC_AVAILABLE and format_name == 'FFMPEG':
            # NVENC settings for imageio
            # Map quality (0-10) to QP (26-20): quality 10 -> QP 20, quality 0 -> QP 26
            qp = int(26 - quality * 0.6)
            w = imageio.get_writer(save_path, fps=fps, codec='h264_nvenc', quality=None,
                                 pixelformat='yuv420p', ffmpeg_params=['-preset', 'p1', '-qp', str(qp)])
        elif format_name == 'FFMPEG':
            # Map quality (0-10) to CRF (26-20): quality 10 -> CRF 20, quality 0 -> CRF 26
            crf = int(26 - quality * 0.6)
            w = imageio.get_writer(save_path, fps=fps, codec='libx264', quality=None,
                                 pixelformat='yuv420p', ffmpeg_params=['-preset', 'veryfast', '-tune', 'film', '-crf', str(crf)])
        elif format_name == 'GIF':
            w = imageio.get_writer(save_path, fps=fps, format='GIF')
        else:
            w = imageio.get_writer(save_path, fps=fps, quality=quality)

        for f in tqdm(frames, desc="Saving"):
            w.append_data(np.array(f))
        w.close()
    except Exception as e:
        print(f"Warning: imageio writer failed ({e}), falling back to direct ffmpeg pipe")
        
        # Fallback: Pipe directly to ffmpeg (much faster than saving frames to disk)
        import subprocess
        
        try:
            sample_frame = np.array(frames[0])
            height, width = sample_frame.shape[:2]
            
            encoding_args = []
            is_mp4 = save_path.lower().endswith('.mp4') or save_path.lower().endswith('.mov') or save_path.lower().endswith('.mkv')
            
            if NVENC_AVAILABLE and is_mp4:
                # Map quality (0-10) to QP (26-20): quality 10 -> QP 20, quality 0 -> QP 26
                qp = int(26 - quality * 0.6)
                encoding_args = ['-c:v', 'h264_nvenc', '-preset', 'p1', '-qp', str(qp), '-pix_fmt', 'yuv420p']
            elif is_mp4:
                # Map quality (0-10) to CRF (26-20): quality 10 -> CRF 20, quality 0 -> CRF 26
                crf = int(26 - quality * 0.6)
                encoding_args = ['-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'film', '-crf', str(crf), '-pix_fmt', 'yuv420p']
            else:
                crf = int(26 - quality * 0.6)
                encoding_args = ['-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'film', '-crf', str(crf), '-pix_fmt', 'yuv420p']

            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'rgb24',
                '-r', str(fps),
                '-i', '-', # Input from pipe
                '-an', # No audio
                *encoding_args,
                save_path
            ]
            
            print(f"Executing fallback FFmpeg command...")
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            
            for f in tqdm(frames, desc="Streaming to FFmpeg"):
                # Ensure frame is contiguous bytes
                img_data = np.array(f).tobytes()
                try:
                    process.stdin.write(img_data)
                except BrokenPipeError:
                    print("FFmpeg pipe broken during writing.")
                    break
            
            out, err = process.communicate()
            if process.returncode != 0:
                print(f"FFmpeg Error: {err.decode('utf-8', errors='ignore')}")
                raise subprocess.CalledProcessError(process.returncode, cmd)
                
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError, Exception) as e:
            print(f"Video saving failed: {e}")
            # Final fallback: just save as GIF
            print("Warning: ffmpeg not available, saving as GIF (This may be slow)")
            # Ensure we have a valid GIF path
            gif_path = save_path
            if not gif_path.lower().endswith('.gif'):
                root, _ = os.path.splitext(save_path)
                gif_path = root + ".gif"
            
            try:
                # Convert to PIL if necessary
                pil_frames = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in frames]
                pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                             duration=1000//fps, loop=0)
                print(f"Saved as GIF: {gif_path}")
            except Exception as gif_error:
                print(f"Failed to save GIF: {gif_error}")

def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 2.0, multiple: int = 128):
    """Compute scaled dimensions and target dimensions"""
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    # Pad to multiple instead of cropping (Round UP)
    # This prevents resolution loss (1480 -> 1408) by going 1480 -> 1536
    # The result will be cropped back to sW, sH after inference
    tW = ((sW + multiple - 1) // multiple) * multiple
    tH = ((sH + multiple - 1) // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(
            f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )

    return sW, sH, tW, tH

def process_frame_gpu(img_or_arr, sH: int, sW: int, tH: int, tW: int, dtype=torch.bfloat16, device='cuda'):
    # Kept for backward compatibility or single frame processing
    return process_batch_gpu([img_or_arr], sH, sW, tH, tW, dtype, device).squeeze(0)

def process_batch_gpu(batch_arr, sH: int, sW: int, tH: int, tW: int, dtype=torch.bfloat16, device='cuda'):
    """
    Process a batch of frames on GPU:
    1. Convert stacked numpy array/list to tensor
    2. Upscale (Bicubic) to (sH, sW)
    3. Center Crop to (tH, tW)
    4. Normalize to [-1, 1]
    """
    if isinstance(batch_arr, list):
         # Handle list of PIL images or numpy arrays
         if len(batch_arr) > 0 and isinstance(batch_arr[0], Image.Image):
             batch_arr = np.stack([np.array(img) for img in batch_arr])
         else:
             batch_arr = np.stack(batch_arr)
             
    # Input: (B, H, W, C) -> Output: (B, C, H, W)
    t = torch.from_numpy(batch_arr).to(device=device, dtype=dtype)
    t = t.permute(0, 3, 1, 2) # (B, C, H, W)
    
    # 2. Upscale
    if t.shape[2] != sH or t.shape[3] != sW:
        t = torch.nn.functional.interpolate(t, size=(sH, sW), mode='bicubic', align_corners=False)
    
    # 3. Pad to Target Size (tH, tW)
    # Target size is guaranteed to be >= sW/sH by compute_scaled_and_target_dims
    curr_h, curr_w = t.shape[2], t.shape[3]
    pad_h = tH - curr_h
    pad_w = tW - curr_w
    
    if pad_h > 0 or pad_w > 0:
        # Pad format: (left, right, top, bottom)
        # We pad right and bottom for simplicity, or center?
        # Center padding is better for symmetric context.
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        t = torch.nn.functional.pad(t, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    
    # 4. Normalize [0, 255] -> [-1, 1]
    t = t / 255.0 * 2.0 - 1.0
    
    return t # Output stays on GPU

def prepare_input_tensor(path: str, scale: float = 2, dtype=torch.bfloat16, device='cuda'):
    """Prepare input tensor from video or image sequence (GPU accelerated)"""
    if os.path.isdir(path):
        # Image sequence
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")

        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        N0 = len(paths0)
        print(f"Input: {w0}x{h0}, {N0} frames")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        scale_str = f"{int(scale)}X" if scale.is_integer() else f"{scale}X"
        print(f"{scale_str} Scaled: {sW}x{sH}")

        # Smart padding to next Frame count F such that (F-1)%8 == 0
        # This ensures we satisfy the model's 8n+1 requirement without dropping real frames
        k = (N0 - 1 + 7) // 8
        F = k * 8 + 1
        pad_len = F - N0
        paths = paths0 + [paths0[-1]] * pad_len
        
        # print(f"Frames: {N0} (padded to {F})")

        frames = []
        for p in paths:
            with Image.open(p).convert('RGB') as img:
                # Use GPU processing instead of CPU resize/crop
                frame_tensor = process_frame_gpu(img, sH, sW, tH, tW, dtype, device)
            frames.append(frame_tensor)
        
        # Stack all at once
        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)
        fps = 30
        return vid, tH, tW, F, fps, None, N0, sH, sW

    if is_video(path):
        # Video file
        rdr = imageio.get_reader(path)
        first = Image.fromarray(rdr.get_data(0)).convert('RGB')
        w0, h0 = first.size

        meta = {}
        try: 
            meta = rdr.get_meta_data()
        except Exception: 
            pass
        
        fps_val = meta.get('fps', 30)
        fps = float(fps_val) if isinstance(fps_val, (int, float)) else 30.0

        def count_frames(r):
            try:
                nf = meta.get('nframes', None)
                if isinstance(nf, int) and nf > 0: 
                    return nf
            except Exception: 
                pass
            try: 
                return r.count_frames()
            except Exception:
                n = 0
                try:
                    while True: 
                        r.get_data(n)
                        n += 1
                except Exception:
                    return n

        total = count_frames(rdr)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        # Enhance FPS precision: prefer Average FPS (Frames / Duration) to preserve total duration
        # This fixes issues where 89 frames @ 18fps = 4.94s (displayed as 4s) instead of 5.0s
        duration_sec = meta.get('duration', 0)
        if duration_sec > 0:
            calc_fps = total / duration_sec
            # If calculated FPS is within 10% of nominal FPS, accept it to ensure duration consistency
            if abs(calc_fps - fps) / (fps + 1e-6) < 0.1:
                # print(f"[Metadata] Refining FPS: {fps:.2f} -> {calc_fps:.3f} to match input duration ({duration_sec}s)")
                fps = calc_fps

        print(f"Input: {w0}x{h0}, {total} frames, {fps:.3f} FPS")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        scale_str = f"{int(scale)}X" if scale.is_integer() else f"{scale}X"
        print(f"{scale_str} Scaled: {sW}x{sH}")

        idx = list(range(total))
        # Smart padding to next Frame count F such that (F-1)%8 == 0
        # Pipeline tends to drop 4 frames at the end due to context window,
        # so we ensure F is large enough and prepare for the drop.
        # Original logic: k = (total - 1 + 7) // 8
        # Enhanced logic: Ensure we have enough padding to cover the pipeline's internal drop (4 frames)
        # If pipeline drops 4 frames, we want Output >= total.
        # But pipeline output size is based on input size.
        # Let's keep the standard block padding, and let the post-inference padding handle the rest.
        k = (total - 1 + 7) // 8
        F = k * 8 + 1
        pad_len = F - total
        idx = idx + [total-1] * pad_len
        
        # print(f"Frames: {total} (padded to {F})")

        frames = []
        batch_size = 32 # Process frames in batches to reduce GPU overhead
        batch_buffer = []
        
        try:
            for i in idx:
                # imageio returns numpy array (H,W,C)
                img_arr = rdr.get_data(i)
                batch_buffer.append(img_arr)
                
                if len(batch_buffer) >= batch_size:
                    batch_tensor = process_batch_gpu(batch_buffer, sH, sW, tH, tW, dtype, device)
                    frames.append(batch_tensor)
                    batch_buffer = []

            # Process remaining frames
            if batch_buffer:
                batch_tensor = process_batch_gpu(batch_buffer, sH, sW, tH, tW, dtype, device)
                frames.append(batch_tensor)
        finally:
            try: 
                rdr.close()
            except Exception: 
                pass

        vid = torch.cat(frames, dim=0).permute(1,0,2,3).unsqueeze(0)
        return vid, tH, tW, F, fps, path, total, sH, sW

    raise ValueError(f"Unsupported input: {path}")

def init_pipeline(args):
    """Initialize pipeline based on mode"""
    print(f"Device: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # Determine model path (relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.join(script_dir, "models/FlashVSR-v1.1")
    model_dir = os.getenv("FLASHVSR-Pro_MODEL_PATH", default_model_dir)
    print(f"Loading models from: {model_dir}")
    
    # Setup dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)
    
    # Initialize VAE Manager and load VAE
    # Determine vae_type automatically from mode
    vae_type = "wan2.1" if args.mode == 'full' else "tcd"
    
    # Suppress output from vae loading as we will control it
    with redirect_stdout(io.StringIO()):
        vae_system_instance = vae_manager.VAESystem(device=args.device, dtype=dtype)
        vae_model = vae_system_instance.load_vae(
            vae_type=vae_type,
            weight_path=None,  # No user custom path supported
            mode=args.mode,
            tile_vae=args.tile_vae,
            tile_size=args.tile_size,
            overlap=args.overlap,
            model_dir=model_dir
        )
    
    # Get VAE info for clean printing
    vae_info = vae_system_instance.get_current_vae_info()
    print(f"Loading VAE: {vae_type} ({vae_info.get('description', '')})")

    # Initialize model manager and load DiT model
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    dit_path = f"{model_dir}/diffusion_pytorch_model_streaming_dmd.safetensors"
    
    # Load DiT model (Removed silence to catch errors)
    # print(f"Loading DiT model from {dit_path}...")
    with redirect_stdout(io.StringIO()):
        mm.load_models([dit_path])
    
    # Basic validation
    if not hasattr(mm, 'state_dict') and not hasattr(mm, 'models'):
         # Diffsynth ModelManager structure varies, but let's check if we have components
         pass

    # Create pipeline based on mode
    with redirect_stdout(io.StringIO()):
        if args.mode == "full":
            pipe = FlashVSRFullPipeline.from_model_manager(mm, device=args.device)
            pipe.vae = vae_model
        else:  # tiny or tiny-long
            if args.mode == "tiny":
                pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=args.device)
            else:  # tiny-long
                pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=args.device)
            pipe.TCDecoder = vae_model
    
    # Load and setup LQ projector efficiently
    lq_proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
    lq_path = f"{model_dir}/LQ_proj_in.ckpt"
    if os.path.exists(lq_path):
        # print(f"Loading LQ Projector from {lq_path}...")
        lq_proj.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    else:
        print(f"[Warning] LQ Projector not found at {lq_path}")
    
    # Move to device and setup pipeline (combined operations)
    # NOTE: For livestream low-latency mode we should avoid CPU offload / VRAM management,
    # otherwise every batch may pay model move / paging costs.
    realtime_low_latency = bool(getattr(args, "realtime_low_latency", False))
    with redirect_stdout(io.StringIO()): # Suppress
        pipe.denoising_model().LQ_proj_in = lq_proj.to(args.device, dtype=dtype)
        pipe.to(args.device)
        if not realtime_low_latency:
            pipe.enable_vram_management(num_persistent_param_in_dit=None)
        pipe.init_cross_kv()
        pipe.load_models_to_device(["dit", "vae"])
    
    return pipe, vae_system_instance


def sync_device(device):
    """Synchronize CUDA work for accurate timing."""
    device_str = str(device)
    if device_str.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def build_pipeline_kwargs(args, LQ, num_frames, height, width):
    """Build pipeline kwargs shared by warmup and real inference."""
    pipeline_kwargs = {
        "prompt": "",
        "negative_prompt": "",
        "cfg_scale": 1.0,
        "num_inference_steps": 1,
        "seed": args.seed,
        "LQ_video": LQ,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "is_full_block": False,
        "if_buffer": True,
        "topk_ratio": args.sparse_ratio * 768 * 1280 / (height * width),
        "kv_ratio": args.kv_ratio,
        "local_range": args.local_range,
        "color_fix": args.color_fix,
        "lq_bootstrap_windows": args.lq_bootstrap_windows,
    }

    if args.tile_vae:
        pipeline_kwargs["tiled"] = True
        vae_tile_size_latent = max(32, args.tile_size // 8)
        vae_overlap_latent = max(4, args.overlap // 8)
        pipeline_kwargs["tile_size"] = (vae_tile_size_latent, vae_tile_size_latent)
        pipeline_kwargs["tile_stride"] = (
            vae_tile_size_latent - vae_overlap_latent,
            vae_tile_size_latent - vae_overlap_latent,
        )
        print(
            f"VAE Tiling Enabled: tile_size (latent)={pipeline_kwargs['tile_size']}, "
            f"stride={pipeline_kwargs['tile_stride']}"
        )

    return pipeline_kwargs


def execute_pipeline(pipe, args, LQ, pipeline_kwargs, announce=True):
    """Run the selected inference path without profiling wrappers."""
    if args.tile_dit:
        if announce:
            print(f"Tiled DiT: tile_size={args.tile_size}, overlap={args.overlap}")
        tile_kwargs = pipeline_kwargs.copy()
        tile_kwargs.pop("LQ_video", None)
        vae_tile_size_tuple = tile_kwargs.pop("tile_size", None)
        return apply_tiled_inference_simple(
            pipe,
            LQ,
            tile_size=args.tile_size,
            overlap=args.overlap,
            tile_size_vae=vae_tile_size_tuple,
            **tile_kwargs,
        )

    if announce:
        if args.mode == "tiny-long":
            msg = "Running inference (Streaming"
            if args.tile_vae:
                msg += " & VAE-tiled"
            msg += ")..."
            print(msg)
        else:
            print("Running inference...")

    return pipe(**pipeline_kwargs)


def run_pipeline_with_optional_profile(pipe, args, LQ, pipeline_kwargs, announce=True):
    """Run inference and optionally measure internal stage timings."""
    profile_stats = {
        "pipeline_total": 0.0,
        "lq_stream_forward_total": 0.0,
        "lq_stream_forward_first": 0.0,
        "lq_stream_forward_rest": 0.0,
        "lq_stream_forward_calls": 0,
        "dit_model_fn_total": 0.0,
        "dit_model_fn_calls": 0,
        "decoder_total": 0.0,
        "decoder_calls": 0,
        "color_fix_total": 0.0,
        "color_fix_calls": 0,
    }

    if not args.profile_timings:
        sync_device(args.device)
        t0 = time.perf_counter()
        video = execute_pipeline(pipe, args, LQ, pipeline_kwargs, announce=announce)
        sync_device(args.device)
        profile_stats["pipeline_total"] = time.perf_counter() - t0
        return video, profile_stats

    lq_module = getattr(pipe.denoising_model(), "LQ_proj_in", None)
    orig_stream_forward = getattr(lq_module, "stream_forward", None) if lq_module is not None else None

    profile_module = None
    if args.mode == "tiny":
        import diffsynth.pipelines.flashvsr_tiny as profile_module
    elif args.mode == "tiny-long":
        import diffsynth.pipelines.flashvsr_tiny_long as profile_module
    orig_model_fn = getattr(profile_module, "model_fn_wan_video", None) if profile_module is not None else None

    decoder_owner = getattr(pipe, "TCDecoder", None)
    decoder_attr = "decode_video"
    if decoder_owner is None or not hasattr(decoder_owner, decoder_attr):
        decoder_owner = getattr(pipe, "vae", None)
        decoder_attr = "decode"
    orig_decoder = getattr(decoder_owner, decoder_attr, None) if decoder_owner is not None else None

    color_owner = getattr(pipe, "ColorCorrector", None)
    orig_color_forward = getattr(color_owner, "forward", None) if color_owner is not None else None

    def timed_stream_forward(*a, **kw):
        sync_device(args.device)
        t0 = time.perf_counter()
        out = orig_stream_forward(*a, **kw)
        sync_device(args.device)
        dt = time.perf_counter() - t0
        if profile_stats["lq_stream_forward_calls"] == 0:
            profile_stats["lq_stream_forward_first"] += dt
        else:
            profile_stats["lq_stream_forward_rest"] += dt
        profile_stats["lq_stream_forward_total"] += dt
        profile_stats["lq_stream_forward_calls"] += 1
        return out

    def timed_model_fn(*a, **kw):
        sync_device(args.device)
        t0 = time.perf_counter()
        out = orig_model_fn(*a, **kw)
        sync_device(args.device)
        profile_stats["dit_model_fn_total"] += time.perf_counter() - t0
        profile_stats["dit_model_fn_calls"] += 1
        return out

    def timed_decoder(*a, **kw):
        sync_device(args.device)
        t0 = time.perf_counter()
        out = orig_decoder(*a, **kw)
        sync_device(args.device)
        profile_stats["decoder_total"] += time.perf_counter() - t0
        profile_stats["decoder_calls"] += 1
        return out

    def timed_color_forward(*a, **kw):
        sync_device(args.device)
        t0 = time.perf_counter()
        out = orig_color_forward(*a, **kw)
        sync_device(args.device)
        profile_stats["color_fix_total"] += time.perf_counter() - t0
        profile_stats["color_fix_calls"] += 1
        return out

    try:
        if orig_stream_forward is not None:
            lq_module.stream_forward = timed_stream_forward
        if orig_model_fn is not None:
            profile_module.model_fn_wan_video = timed_model_fn
        if orig_decoder is not None:
            setattr(decoder_owner, decoder_attr, timed_decoder)
        if orig_color_forward is not None:
            color_owner.forward = timed_color_forward

        sync_device(args.device)
        t0 = time.perf_counter()
        video = execute_pipeline(pipe, args, LQ, pipeline_kwargs, announce=announce)
        sync_device(args.device)
        profile_stats["pipeline_total"] = time.perf_counter() - t0
        return video, profile_stats
    finally:
        if orig_stream_forward is not None:
            lq_module.stream_forward = orig_stream_forward
        if orig_model_fn is not None:
            profile_module.model_fn_wan_video = orig_model_fn
        if orig_decoder is not None:
            setattr(decoder_owner, decoder_attr, orig_decoder)
        if orig_color_forward is not None:
            color_owner.forward = orig_color_forward


def build_warmup_kwargs(pipeline_kwargs):
    """Use a representative short clip to trigger first-run kernel setup."""
    warmup_frames = min(int(pipeline_kwargs["num_frames"]), 25)
    warmup_kwargs = dict(pipeline_kwargs)
    warmup_kwargs["LQ_video"] = pipeline_kwargs["LQ_video"][:, :, :warmup_frames, :, :]
    warmup_kwargs["num_frames"] = warmup_frames
    warmup_kwargs["color_fix"] = False
    return warmup_kwargs


def print_timing_breakdown(stage_timings, profile_stats=None):
    print("\n=== Timing Breakdown ===")
    ordered_keys = [
        ("prepare_input", "prepare_input_tensor"),
        ("model_init", "init_pipeline"),
        ("warmup", "warmup"),
        ("pipeline_total", "pipeline_total"),
        ("crop_output", "crop_output"),
        ("tensor_to_video", "tensor2video"),
        ("save_video", "save_video"),
        ("total", "total"),
    ]
    for key, label in ordered_keys:
        if key in stage_timings:
            print(f"{label:20s}: {stage_timings[key]:.3f}s")

    if not profile_stats:
        return

    print("--- Pipeline Internals ---")
    print(f"{'LQ_proj_total':20s}: {profile_stats['lq_stream_forward_total']:.3f}s")
    print(f"{'LQ_proj_first_call':20s}: {profile_stats['lq_stream_forward_first']:.3f}s")
    print(f"{'LQ_proj_rest_calls':20s}: {profile_stats['lq_stream_forward_rest']:.3f}s")
    print(f"{'LQ_proj_call_count':20s}: {profile_stats['lq_stream_forward_calls']}")
    print(f"{'DiT_model_fn':20s}: {profile_stats['dit_model_fn_total']:.3f}s")
    print(f"{'DiT_call_count':20s}: {profile_stats['dit_model_fn_calls']}")
    print(f"{'decoder_total':20s}: {profile_stats['decoder_total']:.3f}s")
    print(f"{'decoder_call_count':20s}: {profile_stats['decoder_calls']}")
    print(f"{'color_fix_total':20s}: {profile_stats['color_fix_total']:.3f}s")
    print(f"{'color_fix_call_count':20s}: {profile_stats['color_fix_calls']}")
    other = (
        profile_stats["pipeline_total"]
        - profile_stats["lq_stream_forward_total"]
        - profile_stats["dit_model_fn_total"]
        - profile_stats["decoder_total"]
        - profile_stats["color_fix_total"]
    )
    print(f"{'pipeline_other':20s}: {other:.3f}s")

def main():
    total_start_time = time.perf_counter()
    stage_timings = {}
    
    # Enable TF32 for faster matrix multiplications on Ampere/Hopper GPUs (A100/H100)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable cuDNN benchmark for constant input sizes (common in video processing)
    torch.backends.cudnn.benchmark = True
    
    # Display FlashVSR-Pro welcome banner at the very beginning
    print(r"""
███████╗██╗      █████╗ ███████╗██╗  ██╗██╗   ██╗███████╗█████╗           ██████╗ █████╗   ██████╗
██╔════╝██║     ██╔══██╗██╔════╝██║  ██║██║   ██║██╔════╝██╔══██╗         ██╔══██╗██╔══██╗██╔═══██╗
█████╗  ██║     ███████║███████╗███████║╚██╗ ██╔╝███████╗███████║ ██████╗ ██████╔╝███████║██║   ██║
██╔══╝  ██║     ██╔══██║╚════██║██╔══██║ ╚████╔╝ ╚════██║██╔═██║  ╚═════╝ ██╔═══╝ ██╔═██║ ██║   ██║
██║     ███████╗██║  ██║███████║██║  ██║  ╚██╔╝  ███████║██║  ██║         ██║     ██║  ██║╚██████╔╝
╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝         ╚═╝     ╚═╝  ╚═╝ ╚═════╝
                    ⚡FlashVSR-Pro: Enhanced Real-Time Video Super-Resolution
""")

    args = parse_args()
    
    # Check if mode was explicitly set
    mode_explicit = True
    if args.mode is None:
        args.mode = "tiny"
        mode_explicit = False

    # Store explicit arguments logic
    # Set default VAE based on mode if not specified
    if args.mode == "full":
        args.vae_type = "wan2.1"
    else:  # tiny or tiny-long
        args.vae_type = "tcd"

    # Validate VAE compatibility with mode (cache registry to avoid repeated imports)
    vae_registry = vae_manager.VAESystem.VAE_CONFIGS
    is_tcdecoder = vae_registry[args.vae_type]["is_tcdecoder"]
    
    # Combined device and memory checks
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("[WARNING] CUDA not available, falling back to CPU")
            args.device = "cpu"
        else:
            # Check GPU memory only if CUDA is available and not using tile-dit
            if not args.tile_dit:
                free_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                if free_mem < 8:  # less than 8GB
                    print(f"[WARNING] Low GPU memory ({free_mem:.1f}GB), consider using --tile-dit")
    
    # Check audio utility availability
    if args.keep_audio:
        if not AUDIO_AVAILABLE:
            print("[Warning] Audio utilities code not available.")
            args.keep_audio = False
        elif shutil.which('ffmpeg') is None:
            print("[Warning] 'ffmpeg' executable not found in PATH.")
            print("[Warning] Disabling audio preservation.")
            args.keep_audio = False
    
    # Check tile utility availability
    if (args.tile_dit or args.tile_vae) and not TILE_AVAILABLE:
        print("[Warning] Tiled inference requested but tile utilities not available.")
        print("[Warning] Continuing without tiling (may cause high VRAM usage).")
        args.tile_dit = False
        args.tile_vae = False
    
    # Automatically adjust too-small tile-size to avoid window attention errors
    if args.tile_dit and args.tile_size < 128:
        print(f"[WARNING] tile-size {args.tile_size} is too small and may cause window attention errors.")
        print(f"[WARNING] Setting tile-size to 128 (minimum supported value).")
        args.tile_size = 128

    # Create output directory
    if os.path.isdir(args.output):
        output_dir = args.output
    else:
        output_dir = os.path.dirname(args.output)
    # Always create output directory if it exists or not
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Mode: {args.mode}")
    # Prepare input
    print(f"Processing: {args.input}")
    
    # Set dtype
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    prepare_input_start = time.perf_counter()
    LQ, th, tw, F, fps, input_video_path, total_frames_orig, exact_h, exact_w = prepare_input_tensor(
        args.input, 
        scale=args.scale, 
        dtype=dtype,
        device=args.device
    )
    sync_device(args.device)
    stage_timings["prepare_input"] = time.perf_counter() - prepare_input_start
    
    # Override FPS if specified by user
    if args.fps is not None:
        fps = args.fps
    
    # Initialize pipeline with VAE manager
    model_init_start = time.perf_counter()
    pipe, vae_instance = init_pipeline(args)
    sync_device(args.device)
    stage_timings["model_init"] = time.perf_counter() - model_init_start
    
    # Ensure LQ is on the correct device (it should be already if prepare_input_tensor kept it there)
    # This is a no-op if already on device, but safe to keep.
    if LQ.device.type != args.device:
         print(f"Moving input to {args.device}...")
         LQ = LQ.to(args.device)
    
    # Determine output file name
    input_name = os.path.basename(args.input.rstrip('/')).split('.')[0]
    if os.path.isdir(args.output):
        # Generate unique output filename based on user provided parameters
        fn_parts = ["FlashVSR-Pro"]

        # Check if mode was explicitly provided in command line arguments
        mode_explicitly_set = any(arg.startswith("--mode") for arg in sys.argv)
        if mode_explicitly_set:
            fn_parts.append(args.mode)
        
        # Add scale
        fn_parts.append(f"scale{args.scale}")
        # Note: Seed is omitted as requested
        
        # Add optional parameter flags
        if args.keep_audio:
            fn_parts.append("audio")
        
        if args.fps is not None:
            fn_parts.append(f"fps{args.fps}")

        if args.color_fix:
            fn_parts.append("colorfix")
            
        if args.quality != 10:
            fn_parts.append(f"q{args.quality}")

        # Add advanced parameter flags if non-default
        if args.seed != 0:
            fn_parts.append(f"seed{args.seed}")
            
        if args.sparse_ratio != 2.0:
            fn_parts.append(f"sparse{args.sparse_ratio}")
            
        if args.kv_ratio != 3.0:
            fn_parts.append(f"kv{args.kv_ratio}")
            
        if args.local_range != 11:
            fn_parts.append(f"lr{args.local_range}")
            
        if args.dtype != "bf16":
            fn_parts.append(args.dtype)
            
        # Add tiling flags
        is_tiled = False
        if args.tile_dit:
            fn_parts.append("tile_dit")
            is_tiled = True
        
        if args.tile_vae:
            fn_parts.append("tile_vae")
            is_tiled = True
            
        # Append tile params only once if any tiling is enabled
        if is_tiled:
            # Append tile-size if non-default
            if args.tile_size != 256:
                fn_parts.append(f"ts{args.tile_size}")
            # Append overlap if non-default
            if args.overlap != 24:
                fn_parts.append(f"ol{args.overlap}")
            
        fn_parts.append(input_name)
        output_filename = "_".join(fn_parts) + ".mp4"
        output_path = os.path.join(args.output, output_filename)
    else:
        output_path = args.output
    
    # print(f"Output: {output_path}")
    
    pipeline_kwargs = build_pipeline_kwargs(args, LQ, F, th, tw)

    if args.warmup:
        warmup_kwargs = build_warmup_kwargs(pipeline_kwargs)
        print(
            f"Warmup enabled: frames={warmup_kwargs['num_frames']}, "
            f"color_fix={warmup_kwargs['color_fix']}"
        )
        print("Running warmup pass...")
        warmup_start = time.perf_counter()
        warmup_video, _ = run_pipeline_with_optional_profile(
            pipe,
            args,
            warmup_kwargs["LQ_video"],
            warmup_kwargs,
            announce=False,
        )
        sync_device(args.device)
        stage_timings["warmup"] = time.perf_counter() - warmup_start
        del warmup_video
        print(f"Warmup completed in {stage_timings['warmup']:.2f} seconds")

    # Run inference (tiled or standard)
    inference_start_time = time.perf_counter()
    video, profile_stats = run_pipeline_with_optional_profile(
        pipe,
        args,
        LQ,
        pipeline_kwargs,
        announce=True,
    )
    inference_end_time = time.perf_counter()
    inference_duration = inference_end_time - inference_start_time
    stage_timings["pipeline_total"] = inference_duration
    print(f"Inference completed in {inference_duration:.2f} seconds")
    
    if NVENC_AVAILABLE:
        print("NVENC hardware encoding detection passed (functional test).")
    else:
        print("NVENC hardware encoding not available or failed functional test. Using CPU encoding.")

    # Convert and save video
    # Crop back to exact requested resolution sH x sW (exact_h, exact_w)
    # The Model Output `video` is (B, C, T, H, W)
    crop_start = time.perf_counter()
    if video.shape[-2] != exact_h or video.shape[-1] != exact_w:
        # We padded center, so we crop center
        curr_h, curr_w = video.shape[-2], video.shape[-1]
        pad_h = curr_h - exact_h
        pad_w = curr_w - exact_w
        
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        
        video = video[..., pad_top:pad_top+exact_h, pad_left:pad_left+exact_w]
    sync_device(args.device)
    stage_timings["crop_output"] = time.perf_counter() - crop_start

    tensor_to_video_start = time.perf_counter()
    frames = tensor2video(video)
    stage_timings["tensor_to_video"] = time.perf_counter() - tensor_to_video_start

    # Ensure output Duration matches Input Duration (Remove padding / Fill missing)
    if len(frames) > total_frames_orig:
        frames = frames[:total_frames_orig]
    elif len(frames) < total_frames_orig:
        # print(f"[Warning] Output frames ({len(frames)}) < Input ({total_frames_orig}). Padding to restore duration.")
        frames.extend([frames[-1]] * (total_frames_orig - len(frames)))

    # Handle audio preservation
    if args.keep_audio and input_video_path and is_video(input_video_path) and has_audio_stream(input_video_path):
        # Optimized: piped saving (single pass)
        print("Preserving audio (streaming mode)...")
        save_start = time.perf_counter()
        success = save_video_with_audio_piped(frames, output_path, input_video_path, fps=fps, quality=args.quality)
        
        if not success:
            print("Piping failed, falling back to temp file method...")
            # Fallback: save silent video first, then merge audio
            temp_output = output_path.replace('.mp4', '_temp.mp4')
            save_video(frames, temp_output, fps=fps, quality=args.quality)
            print("Preserving audio (merge mode)...")
            copy_video_with_audio(input_video_path, temp_output, output_path)
            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
        stage_timings["save_video"] = time.perf_counter() - save_start
    else:
        # No audio to preserve: save directly to final output
        save_start = time.perf_counter()
        save_video(frames, output_path, fps=fps, quality=args.quality)
        stage_timings["save_video"] = time.perf_counter() - save_start

    print(f"Done!\nOutput: {output_path}")

    # Process total time
    total_end_time = time.perf_counter()
    total_duration = total_end_time - total_start_time
    stage_timings["total"] = total_duration
    print(f"Total processing time: {total_duration:.2f} seconds")
    if args.profile_timings:
        print_timing_breakdown(stage_timings, profile_stats)

    # Cleanup
    vae_instance.clean_memory()
    del pipe
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()


# ffmpeg -i inputs/example4.mp4 -vf "fps=10,scale=640:360" -frames:v 25 inputs/output_25_10fps_360p.mp4