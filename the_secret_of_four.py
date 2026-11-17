"""THE SECRET OF FOUR — Prime & Cousin Prime Spiral | 60-second vertical video.

Mathematics (radians): x(n) = n*cos(n), y(n) = n*sin(n).
A cousin-prime pair is (p, p+4), with both numbers prime.

Animation:
  0-3s   : full prime-spiral opening hook
  3-4s   : fade and rewind
  4-22s  : reveal all primes
  22-32s : highlight cousin primes; fade the other primes
  32-49s : reveal the two prime residue classes (modulo 6)
  49-60s : final reveal

The coordinates are exact. ONLY DOTS are drawn: no lines or curves,
including no connections between cousin-prime pairs.

Requirements: numpy, matplotlib, FFmpeg (only for MP4).
Works as a .py script or in a single Jupyter/Google Colab cell.
Use PREVIEW_ONLY=True to generate a PNG without rendering a video.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle


# ==================== SETTINGS ====================
MAX_N = 1_000_000              # Try 100_000, 1_000_000, 10_000_000
PREVIEW_ONLY = True             # True = PNG; False = full 60-second MP4
PREVIEW_SECOND = 55.0         # Try 2, 18, 28, 42, 55
DURATION = 60
FPS = 30
WIDTH, HEIGHT, DPI = 1080, 1920, 120

DOWNLOAD_IN_COLAB = True
SHOW_PREVIEW_IN_NOTEBOOK = True

# Timeline (seconds)
HOOK_END = 3.0
RESET_END = 4.0
PRIME_BUILD_END = 22.0
FILTER_END = 32.0
COLOR_END = 49.0

# Styling
BG = "#050916"
WHITE = "#F4F8FF"
CYAN = "#56DEFF"
GOLD = "#FFD66E"
PINK = "#FF5AC8"
MUTED = "#A7B6D5"
CAMERA_PADDING = 1.13
CAMERA_SMOOTHING = 0.15
MIN_HALF_HEIGHT = 45.0


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def sieve_of_eratosthenes(limit: int) -> np.ndarray:
    """Fast vectorized sieve; memory O(limit)."""
    if limit < 2:
        raise ValueError("MAX_N must be at least 2")
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[:2] = False
    is_prime[4::2] = False
    for p in range(3, math.isqrt(limit) + 1, 2):
        if is_prime[p]:
            is_prime[p * p :: 2 * p] = False
    return is_prime


def polar_points(numbers: np.ndarray) -> np.ndarray:
    """Exact mathematical mapping: r=n, theta=n radians."""
    n = np.asarray(numbers, dtype=np.float64)
    return np.column_stack((n * np.cos(n), n * np.sin(n)))


def build_data(max_n: int):
    sieve = sieve_of_eratosthenes(max_n)
    primes = np.flatnonzero(sieve).astype(np.int64)
    starts = primes[primes <= max_n - 4]
    starts = starts[sieve[starts + 4]]
    cousin = np.union1d(starts, starts + 4)

    # Independent sanity checks for known, small values.
    if max_n == 100:
        assert len(primes) == 25
        assert starts[:4].tolist() == [3, 7, 13, 19]
        assert len(starts) == 8
    if max_n == 1_000_000:
        assert len(primes) == 78_498
        assert len(starts) == 8_144
        assert len(cousin) == 16_287

    return primes, starts, cousin


def run(max_n=MAX_N, preview_only=PREVIEW_ONLY, preview_second=PREVIEW_SECOND):
    began = time.perf_counter()
    if max_n < 7:
        raise ValueError("Choose MAX_N >= 7 to include cousin-prime pairs")
    if max_n > 50_000_000:
        raise ValueError(
            "This version uses an in-memory sieve. Set MAX_N <= 50,000,000 "
            "or implement a segmented sieve for larger ranges."
        )
    if not (0 <= preview_second < DURATION):
        raise ValueError("PREVIEW_SECOND must be in [0, 60)")

    primes, starts, cousin = build_data(max_n)
    prime_xy = polar_points(primes)
    cousin_xy = polar_points(cousin)
    total_primes, total_pairs, total_cousin = len(primes), len(starts), len(cousin)

    print(f"Range: 2 to {max_n:,}")
    print(f"All primes: {total_primes:,}")
    print(f"Cousin-prime pairs (both endpoints <= MAX_N): {total_pairs:,}")
    print(f"Distinct primes belonging to a cousin-prime pair: {total_cousin:,}")
    print("Point-only visualization: NO connecting lines.", flush=True)

    # Color primes by their residue class modulo 6.
    # For cousin pairs (p, p+4) with p>3, endpoints are in OPPOSITE
    # residue classes 1 and 5 (mod 6). The pair (3,7) is exceptional.
    # Except for 2 and 3, primes are always 1 or 5 modulo 6.
    prime_colors = np.zeros((total_primes, 4), dtype=np.float32)
    prime_colors[primes % 6 == 1] = (0.28, 0.84, 1.0, 1.0)
    prime_colors[primes % 6 == 5] = (1.0, 0.74, 0.34, 1.0)
    prime_colors[(primes == 2) | (primes == 3)] = (1.0, 1.0, 1.0, 1.0)

    # Running bounds for a smooth camera that always contains all visible points.
    xs, ys = prime_xy.T
    lo_x = np.minimum.accumulate(np.minimum(xs, 0.0))
    hi_x = np.maximum.accumulate(np.maximum(xs, 0.0))
    lo_y = np.minimum.accumulate(np.minimum(ys, 0.0))
    hi_y = np.maximum.accumulate(np.maximum(ys, 0.0))

    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=BG)
    ax_left, ax_bottom, ax_width, ax_height = 0.045, 0.155, 0.91, 0.68
    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height], facecolor=BG)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    axes_aspect = (WIDTH * ax_width) / (HEIGHT * ax_height)

    def required_half(x0, x1, y0, y1, cx, cy, padding=CAMERA_PADDING):
        return max(
            MIN_HALF_HEIGHT,
            (y1 - cy) * padding,
            (cy - y0) * padding,
            (x1 - cx) * padding / axes_aspect,
            (cx - x0) * padding / axes_aspect,
        )

    full_cx = float((lo_x[-1] + hi_x[-1]) / 2)
    full_cy = float((lo_y[-1] + hi_y[-1]) / 2)
    full_half = required_half(lo_x[-1], hi_x[-1], lo_y[-1], hi_y[-1], full_cx, full_cy)

    # Clean editorial typography; no axes or grid.
    fig.text(0.5, 0.958, "THE SECRET OF FOUR", color=CYAN,
             fontsize=27, fontweight="bold", ha="center")
    fig.text(0.5, 0.930, "PRIME NUMBERS  →  COUSIN PRIMES", color=WHITE,
             fontsize=16, fontweight="bold", ha="center")
    fig.text(0.5, 0.899, "ONE SPIRAL. TWO PRIME NUMBERS. FOUR APART.",
             color=MUTED, fontsize=11, ha="center")
    phase_text = fig.text(0.5, 0.865, "", color=GOLD, fontsize=13,
                          fontweight="bold", ha="center")
    counter_text = fig.text(0.5, 0.841, "", color=WHITE,
                            fontsize=10.5, ha="center")
    # Studio-style lower third: large type, clear contrast, mobile-safe padding.
    fig.patches.extend([
        Rectangle((0.062, 0.034), 0.876, 0.128,
                  transform=fig.transFigure, facecolor="#0B172B",
                  edgecolor="none", zorder=1),
        Rectangle((0.062, 0.160), 0.876, 0.0022,
                  transform=fig.transFigure, facecolor=CYAN,
                  edgecolor="none", zorder=2),
    ])
    footer_stage = fig.text(0.5, 0.135, "", color=GOLD,
                            fontsize=11, fontweight="bold", ha="center", zorder=3)
    bottom_text = fig.text(0.5, 0.099, "", color=WHITE,
                           fontsize=17, fontweight="bold", ha="center", zorder=3)
    detail_text = fig.text(0.5, 0.069, "", color="#C9D7E9",
                           fontsize=11, ha="center", zorder=3)

    # Artists are reused across all frames: fast and memory-efficient.
    if total_primes >= 300_000:
        prime_size, cousin_size = 1.7, 4.2
    elif total_primes >= 30_000:
        prime_size, cousin_size = 3.5, 7.0
    else:
        prime_size, cousin_size = 6.5, 10.0

    prime_dots = ax.scatter([], [], s=prime_size, color=WHITE,
                            edgecolors="none", rasterized=True, zorder=2)
    cousin_dots = ax.scatter([], [], s=cousin_size, color=PINK,
                           edgecolors="none", rasterized=True, zorder=4)
    # Two additional dot layers reveal the prime residue classes.
    # Every cousin prime > 3 is either 1 or 5 modulo 6.
    # The exceptional prime 3 remains visible as a white point.
    cousin_mod1 = cousin_xy[cousin % 6 == 1]
    cousin_mod5 = cousin_xy[cousin % 6 == 5]
    class_1_dots = ax.scatter(cousin_mod1[:, 0], cousin_mod1[:, 1],
                              s=cousin_size, color=CYAN, alpha=0,
                              edgecolors="none", rasterized=True, zorder=5)
    class_5_dots = ax.scatter(cousin_mod5[:, 0], cousin_mod5[:, 1],
                              s=cousin_size, color=GOLD, alpha=0,
                              edgecolors="none", rasterized=True, zorder=5)
    # n=3 is the only prime outside the 1/5 classes that can be a cousin prime.
    special_xy = polar_points(np.array([3])) if 3 in cousin else np.empty((0, 2))
    special_dots = ax.scatter(special_xy[:, 0], special_xy[:, 1],
                              s=cousin_size * 1.25, color=WHITE, alpha=0,
                              edgecolors="none", rasterized=True, zorder=6)

    frame_total = DURATION * FPS
    frame_times = np.arange(frame_total, dtype=np.float64) / FPS
    views = np.empty((frame_total, 3), dtype=np.float64)
    prime_counts = np.empty(frame_total, dtype=np.int32)
    camera_x = camera_y = 0.0
    camera_half = MIN_HALF_HEIGHT

    # Precompute views once. A full camera on all late stages prevents clipping.
    for f, t in enumerate(frame_times):
        if t < RESET_END:
            count = total_primes
            zoom = 1.075 - 0.075 * smoothstep(t / HOOK_END)
            views[f] = (full_cx, full_cy, full_half * zoom)
        elif t < PRIME_BUILD_END:
            fraction = (t - RESET_END) / (PRIME_BUILD_END - RESET_END)
            count = min(total_primes, 1 + int((fraction ** 1.14) * (total_primes - 1)))
            j = count - 1
            cx = float((lo_x[j] + hi_x[j]) / 2)
            cy = float((lo_y[j] + hi_y[j]) / 2)
            target_half = required_half(lo_x[j], hi_x[j], lo_y[j], hi_y[j], cx, cy)
            if f == int(RESET_END * FPS):
                camera_x, camera_y, camera_half = cx, cy, target_half
            else:
                camera_x += CAMERA_SMOOTHING * (cx - camera_x)
                camera_y += CAMERA_SMOOTHING * (cy - camera_y)
                camera_half += CAMERA_SMOOTHING * (target_half - camera_half)
                camera_half = max(camera_half, required_half(
                    lo_x[j], hi_x[j], lo_y[j], hi_y[j], camera_x, camera_y,
                    padding=1.04))
            views[f] = (camera_x, camera_y, camera_half)
        else:
            count = total_primes
            views[f] = (full_cx, full_cy, full_half)
        prime_counts[f] = count

    previous_prime_count = -1
    cousin_shown = False

    def draw_frame(f):
        nonlocal previous_prime_count, cousin_shown
        t = float(frame_times[f])
        count = int(prime_counts[f])
        if count != previous_prime_count:
            prime_dots.set_offsets(prime_xy[:count])
            prime_dots.set_facecolors(prime_colors[:count])
            previous_prime_count = count

        cx, cy, half = views[f]
        ax.set_xlim(cx - half * axes_aspect, cx + half * axes_aspect)
        ax.set_ylim(cy - half, cy + half)

        # Intro fade, prime buildup, selection, two-color dot reveal, finale.
        if t < HOOK_END:
            prime_alpha = 0.95
        elif t < RESET_END:
            prime_alpha = 0.95 * (1 - smoothstep((t - HOOK_END) / (RESET_END - HOOK_END)))
        elif t < RESET_END + 0.4:
            prime_alpha = 0.95 * smoothstep((t - RESET_END) / 0.4)
        elif t < PRIME_BUILD_END:
            prime_alpha = 0.95
        elif t < FILTER_END:
            prime_alpha = 0.95 * (1.0 - smoothstep((t - PRIME_BUILD_END) / (FILTER_END - PRIME_BUILD_END)))
        else:
            prime_alpha = 0.0
        prime_dots.set_alpha(prime_alpha)

        if t >= PRIME_BUILD_END:
            if not cousin_shown:
                cousin_dots.set_offsets(cousin_xy)
                cousin_shown = True
            cousin_alpha = float(smoothstep((t - PRIME_BUILD_END) / 4.5))
        else:
            cousin_alpha = 0.0
        # The final stage uses only colored dots, never lines.
        color_mix = float(smoothstep((t - FILTER_END) / (COLOR_END - FILTER_END)))
        cousin_dots.set_alpha(cousin_alpha * (1.0 - color_mix))
        class_1_dots.set_alpha(color_mix)
        class_5_dots.set_alpha(color_mix)
        special_dots.set_alpha(color_mix)

        if t < HOOK_END:
            footer_stage.set_text("THE EXPERIMENT")
            phase_text.set_text("THE FULL PRIME SPIRAL")
            counter_text.set_text(f"{total_primes:,} PRIMES  |  INTEGERS UP TO {max_n:,}")
            bottom_text.set_text("WHAT IF WE KEEP ONLY PRIMES FOUR APART?")
            detail_text.set_text("Each dot follows x = n cos(n) and y = n sin(n)")
        elif t < RESET_END:
            footer_stage.set_text("")
            phase_text.set_text("LET'S START WITH THE NUMBERS")
            counter_text.set_text("")
            bottom_text.set_text("")
            detail_text.set_text("")
        elif t < PRIME_BUILD_END:
            footer_stage.set_text("01   /   ALL PRIME NUMBERS")
            phase_text.set_text("01  /  PLOT ALL PRIME NUMBERS")
            counter_text.set_text(f"PRIMES PLOTTED: {count:,} / {total_primes:,}")
            bottom_text.set_text("EVERY DOT IS A PRIME NUMBER")
            detail_text.set_text("Polar mapping  |  Radius = n  |  Angle = n radians")
        elif t < FILTER_END:
            footer_stage.set_text("02   /   COUSIN PRIME SELECTION")
            phase_text.set_text("02  /  KEEP ONLY COUSIN PRIMES")
            counter_text.set_text(f"{total_cousin:,} DISTINCT PRIMES  |  {total_pairs:,} PAIRS")
            bottom_text.set_text("TWO PRIMES. EXACTLY FOUR APART.")
            detail_text.set_text("A cousin-prime pair is (p, p + 4), with both numbers prime")
        elif t < COLOR_END:
            footer_stage.set_text("03   /   TWO RESIDUE CLASSES")
            phase_text.set_text("03  /  PRIME RESIDUE CLASSES")
            counter_text.set_text(f"{total_pairs:,} COUSIN-PRIME PAIRS  |  {total_cousin:,} UNIQUE PRIMES")
            bottom_text.set_text("EVERY DOT IS A COUSIN PRIME")
            detail_text.set_text("Cyan: remainder 1  |  Gold: remainder 5  |  Modulo 6")
        else:
            footer_stage.set_text("THE FINAL REVEAL")
            phase_text.set_text("THE GEOMETRY OF COUSIN PRIMES")
            counter_text.set_text(f"{total_pairs:,} PAIRS  |  {total_cousin:,} UNIQUE PRIME NUMBERS")
            bottom_text.set_text("ONE SPIRAL. THE SECRET OF FOUR.")
            detail_text.set_text(f"{total_pairs:,} cousin-prime pairs  |  {total_cousin:,} distinct prime numbers")

    if preview_only:
        f = min(frame_total - 1, int(preview_second * FPS))
        draw_frame(f)
        filename = f"cousin_prime_no_connections_{max_n}_preview_{preview_second:g}s.png"
        fig.savefig(filename, dpi=DPI, facecolor=BG)
        plt.close(fig)
        print(f"PNG preview: {os.path.abspath(filename)}")
        print(f"Elapsed: {time.perf_counter() - began:.2f}s")
        if SHOW_PREVIEW_IN_NOTEBOOK:
            try:
                from IPython.display import display, Image
                display(Image(filename=filename, width=340))
            except ImportError:
                pass
        return filename

    if not FFMpegWriter.isAvailable():
        plt.close(fig)
        raise RuntimeError("FFmpeg is not installed or not on PATH.")
    filename = f"cousin_prime_no_connections_{max_n}_60s.mp4"
    writer = FFMpegWriter(
        fps=FPS, codec="libx264",
        extra_args=["-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart"]
    )
    print(f"Rendering {frame_total} frames: {filename}", flush=True)
    with writer.saving(fig, filename, dpi=DPI):
        for f in range(frame_total):
            draw_frame(f)
            writer.grab_frame()
            if (f + 1) % (5 * FPS) == 0:
                print(f"  {(f + 1) // FPS:2d}/{DURATION} seconds rendered", flush=True)
    plt.close(fig)
    print(f"MP4 saved: {os.path.abspath(filename)}")
    print(f"Elapsed: {(time.perf_counter() - began) / 60:.1f} min")
    if DOWNLOAD_IN_COLAB:
        try:
            from google.colab import files
            files.download(filename)
        except ImportError:
            pass
    return filename


def main():
    parser = argparse.ArgumentParser(description="Prime and cousin-prime spiral, 60s video")
    parser.add_argument("--max-n", type=int, default=MAX_N)
    parser.add_argument("--preview-second", type=float, default=PREVIEW_SECOND)
    parser.add_argument("--render", action="store_true", help="Render MP4 instead of preview PNG")
    args, _ = parser.parse_known_args()
    run(max_n=args.max_n, preview_only=not args.render and PREVIEW_ONLY,
        preview_second=args.preview_second)


if __name__ == "__main__":
    main()
