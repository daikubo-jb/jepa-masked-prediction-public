#!/usr/bin/env python3
"""Create deterministic four-choice ball-gap videos as standard-library PPM frames."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

WIDTH = 64
HEIGHT = 64
FRAMES = 64
GAP = range(30, 34)
ALTERNATIVE_OFFSETS = (-7.0, -3.0, 3.0)
BASE_Y_MIN = 30.0
BASE_Y_MAX = 33.0
BALL_RADIUS = 4
SAFE_SOURCE_DIFF_MIN = 19
SAFE_SOURCE_DIFF_MAX = 44
PROCESSOR_RESIZE_EDGE = 292
PROCESSOR_CROP_EDGE = 256
PROCESSOR_CROP_OFFSET = (PROCESSOR_RESIZE_EDGE - PROCESSOR_CROP_EDGE) // 2


def _set_pixel(pixels: bytearray, x: int, y: int, rgb: tuple[int, int, int]) -> None:
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        offset = (y * WIDTH + x) * 3
        pixels[offset:offset + 3] = bytes(rgb)


def _frame(frame_index: int, offset: float, *, x_start: float, x_velocity: float,
           base_y: float) -> bytes:
    pixels = bytearray([236, 238, 240] * (WIDTH * HEIGHT))
    # Fixed landmarks outside the target box provide visible context without a candidate cue.
    for x in range(4, WIDTH - 4):
        _set_pixel(pixels, x, 8, (195, 200, 205))
        _set_pixel(pixels, x, HEIGHT - 9, (195, 200, 205))
    for y in range(12, HEIGHT - 12):
        _set_pixel(pixels, 8, y, (195, 200, 205))
        _set_pixel(pixels, WIDTH - 9, y, (195, 200, 205))

    x_center = x_start + x_velocity * frame_index
    y_center = base_y
    if frame_index in GAP:
        phase = (frame_index - 29) / 5.0
        y_center += offset * math.sin(math.pi * phase)

    radius = BALL_RADIUS
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                _set_pixel(pixels, x, y, (45, 105, 220))
    return b"P6\n64 64\n255\n" + bytes(pixels)


def _seeded_candidate_plan(seed: int) -> tuple[list[dict], dict[str, float]]:
    rng = random.Random(seed)
    candidate_specs = [
        {"trajectory": "constant-velocity", "offset": 0.0, "is_rule_correct": True},
        *({"trajectory": f"perturbed-{index}", "offset": value, "is_rule_correct": False}
          for index, value in enumerate(ALTERNATIVE_OFFSETS, start=1)),
    ]
    rng.shuffle(candidate_specs)
    # Keep every candidate's complete varied ball support inside the fixed mask's
    # inverse image after resize/crop, including one source-pixel bilinear support.
    trajectory = {
        "x_start": round(12.0 + rng.random() * 2.0, 6),
        "x_velocity": round(0.34 + rng.random() * 0.12, 6),
        "base_y": round(BASE_Y_MIN + rng.random() * (BASE_Y_MAX - BASE_Y_MIN), 6),
    }
    return candidate_specs, trajectory


def seeded_trajectory(seed: int) -> dict[str, float]:
    """Return only the visible trajectory template, never the rule-correct candidate."""
    _, trajectory = _seeded_candidate_plan(seed)
    return dict(trajectory)


def render_candidate_frames_in_memory(seed: int) -> tuple[tuple[bytes, ...], ...]:
    """Render fixed-seed PPM bytes without creating fixture files or exposing the answer."""
    candidate_specs, trajectory = _seeded_candidate_plan(seed)
    return tuple(
        tuple(_frame(frame_index, spec["offset"], **trajectory) for frame_index in range(FRAMES))
        for spec in candidate_specs
    )


def conservative_candidate_source_support() -> tuple[int, int]:
    """Bound all varied ball pixels using the full offset magnitude and radius."""
    max_offset = max(abs(offset) for offset in ALTERNATIVE_OFFSETS)
    return (math.floor(BASE_Y_MIN - max_offset - BALL_RADIUS),
            math.ceil(BASE_Y_MAX + max_offset + BALL_RADIUS))


def transformed_bilinear_support(source_index_min: int, source_index_max: int) -> tuple[float, float]:
    """Conservative output-coordinate support for changed source pixels on one axis."""
    if (type(source_index_min) is not int or type(source_index_max) is not int
            or not 0 <= source_index_min <= source_index_max < WIDTH):
        raise ValueError("source support indices must be an ordered in-frame integer range")
    scale = PROCESSOR_RESIZE_EDGE / WIDTH
    crop = PROCESSOR_CROP_OFFSET
    first_center = (source_index_min + 0.5) * scale - 0.5 - crop
    last_center = (source_index_max + 0.5) * scale - 0.5 - crop
    # A bilinear tap can reach one input pixel from a changed source pixel.
    support_radius = scale
    return first_center - support_radius, last_center + support_radius


def generate_fixture(output: Path, seed: int = 20261025) -> dict:
    output = output.expanduser()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing fixture directory: {output}")
    output.mkdir(parents=True)
    candidate_specs, trajectory = _seeded_candidate_plan(seed)
    candidates = []
    correct_candidate_id = None
    for candidate, spec in enumerate(candidate_specs):
        frame_paths = []
        candidate_dir = output / "frames" / f"candidate-{candidate}"
        candidate_dir.mkdir(parents=True)
        for frame_index in range(FRAMES):
            filename = f"{frame_index:03d}.ppm"
            (candidate_dir / filename).write_bytes(_frame(frame_index, spec["offset"], **trajectory))
            frame_paths.append(f"frames/candidate-{candidate}/{filename}")
        candidate_id = f"candidate-{candidate}"
        candidates.append({"id": candidate_id, "frames": frame_paths})
        if spec["is_rule_correct"]:
            correct_candidate_id = candidate_id

    manifest = {
        "schema_version": 3,
        "kind": "synthetic-ball-middle-gap",
        "width": WIDTH,
        "height": HEIGHT,
        "frames": FRAMES,
        "target_frame_range": [30, 33],
        "visible_context_frame_ranges": [[0, 29], [34, 63]],
        "candidate_count": 4,
        "candidate_order_seed": seed,
        "visible_trajectory_template": trajectory,
        "ground_truth_file": "answer_key.json",
        "model_input_reference": {
            "resolution": 256,
            "patch_size": 16,
            "tubelet_size": 2,
            "token_grid": [32, 16, 16],
            "target_tubelets": [15, 16],
            "target_rows": [4, 11],
            "target_columns": [4, 11],
            "expected_target_tokens": 128,
            "expected_context_tokens": 8064,
        },
        "candidates": candidates,
        "provenance": "generated deterministically by scripts/generate_ball_gap.py; no real media",
    }
    (output / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "answer_key.json").write_text(json.dumps({
        "schema_version": 1,
        "correct_candidate_id": correct_candidate_id,
        "rule": "constant horizontal velocity and constant vertical position through the gap",
        "claim_scope": "synthetic generator rule only; not a unique real-world inference",
        "ambiguous_by_visible_context": True,
        "ambiguity_note": "All candidates share identical visible frames; additional motion assumptions define the synthetic rule answer.",
    }, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    manifest = generate_fixture(args.output, seed=args.seed)
    print(json.dumps({
        "generated": True,
        "candidate_count": manifest["candidate_count"],
        "frames_per_candidate": manifest["frames"],
        "resolution": [manifest["width"], manifest["height"]],
        "expected_target_tokens": manifest["model_input_reference"]["expected_target_tokens"],
        "real_media": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
