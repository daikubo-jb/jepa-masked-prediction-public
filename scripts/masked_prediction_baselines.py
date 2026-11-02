"""Model-free, label-free baselines for the fixed synthetic middle-gap pilot."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable, Mapping, Sequence


def seeded_random_scores(candidate_ids: Sequence[str], seed: int) -> dict[str, float]:
    """Return a reproducible random ordering with no access to labels or trajectories."""
    ids = list(candidate_ids)
    if len(ids) != 4 or len(set(ids)) != 4 or type(seed) is not int:
        raise ValueError("random baseline needs four unique candidate IDs and an integer seed")
    random.Random(seed ^ 0x5EED5EED).shuffle(ids)
    return {candidate_id: float(rank) for rank, candidate_id in enumerate(ids)}


def visible_context_tie_scores(candidate_ids: Sequence[str]) -> dict[str, float]:
    """All candidates have identical visible inputs, so a context-only model must tie."""
    ids = list(candidate_ids)
    if len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError("context-only baseline needs four unique candidate IDs")
    return {candidate_id: 0.0 for candidate_id in ids}


def _parse_ppm(data: bytes) -> tuple[int, int, bytes]:
    parts = data.split(b"\n", 3)
    if len(parts) != 4 or parts[0] != b"P6" or parts[2] != b"255":
        raise ValueError("expected a binary P6 PPM with max value 255")
    try:
        width, height = (int(value) for value in parts[1].split())
    except (ValueError, TypeError) as error:
        raise ValueError("invalid PPM dimensions") from error
    header_end = len(parts[0]) + len(parts[1]) + len(parts[2]) + len(b"\n") * 3
    # The fixture contract uses exactly P6\\n64 64\\n255\\n, avoiding ambiguous headers.
    if (width, height) != (64, 64) or data[:header_end] != b"P6\n64 64\n255\n":
        raise ValueError("unsupported PPM dimensions or header")
    pixels = data[header_end:]
    if len(pixels) != 64 * 64 * 3:
        raise ValueError("PPM payload length does not match 64x64 RGB")
    return width, height, pixels


def _centroid_from_blue(data: bytes) -> tuple[float, float]:
    _, _, pixels = _parse_ppm(data)
    xs: list[int] = []
    ys: list[int] = []
    for offset in range(0, len(pixels), 3):
        red, green, blue = pixels[offset:offset + 3]
        if red < 100 and green > 60 and blue > 150:
            pixel = offset // 3
            ys.append(pixel // 64)
            xs.append(pixel % 64)
    if not xs:
        raise ValueError("visible frame does not contain the synthetic blue ball")
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _fit_line(points: Sequence[tuple[int, float]]) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("at least two visible observations are required")
    mean_t = sum(t for t, _ in points) / len(points)
    mean_v = sum(value for _, value in points) / len(points)
    denominator = sum((t - mean_t) ** 2 for t, _ in points)
    if denominator <= 0:
        raise ValueError("visible observations do not span time")
    slope = sum((t - mean_t) * (value - mean_v) for t, value in points) / denominator
    return mean_v - slope * mean_t, slope


def _appearance_from_visible_frame(data: bytes) -> tuple[bytes, bytes, set[tuple[int, int]]]:
    _, _, pixels = _parse_ppm(data)
    background = pixels[:3]
    xs: list[int] = []
    ys: list[int] = []
    foreground: list[bytes] = []
    for offset in range(0, len(pixels), 3):
        color = pixels[offset:offset + 3]
        if color[0] < 100 and color[1] > 60 and color[2] > 150:
            pixel = offset // 3
            ys.append(pixel // 64)
            xs.append(pixel % 64)
            foreground.append(color)
    if not xs:
        raise ValueError("visible appearance frame does not contain a foreground ball")
    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)
    anchor_x, anchor_y = round(center_x), round(center_y)
    template = {(x - anchor_x, y - anchor_y) for x, y in zip(xs, ys)}
    color_counts: dict[bytes, int] = {}
    for color in foreground:
        color_counts[color] = color_counts.get(color, 0) + 1
    foreground_color = max(color_counts, key=color_counts.get)
    return background, foreground_color, template


def _predicted_frame(x_center: float, y_center: float, background: bytes,
                     foreground: bytes, template: set[tuple[int, int]]) -> bytes:
    pixels = bytearray(background * (64 * 64))
    anchor_x, anchor_y = round(x_center), round(y_center)
    for delta_x, delta_y in template:
        x, y = anchor_x + delta_x, anchor_y + delta_y
        if 0 <= x < 64 and 0 <= y < 64:
            offset = (y * 64 + x) * 3
            pixels[offset:offset + 3] = foreground
    return bytes(pixels)


def image_space_interpolation_scores(
    fixture_root: Path,
    candidates: Sequence[Mapping[str, object]],
    *,
    read_frame: Callable[[Path, str], bytes],
    target_frames: Sequence[int] = (30, 31, 32, 33),
    visible_frames: Sequence[int] = tuple(range(0, 30)) + tuple(range(34, 64)),
) -> dict[str, float]:
    """Fit ball motion only from visible frames; compare predicted pixels in the gap.

    Candidate target frames are scoring targets, like the model target encoder. This
    baseline never receives the answer key, candidate role, or generator offset.
    """
    if len(candidates) != 4 or tuple(target_frames) != (30, 31, 32, 33):
        raise ValueError("interpolation baseline supports the fixed four-candidate gap")
    reference_frames = candidates[0].get("frames")
    if not isinstance(reference_frames, list) or len(reference_frames) != 64:
        raise ValueError("candidate frame list is invalid")
    observations = [
        (index, _centroid_from_blue(read_frame(fixture_root, reference_frames[index])))
        for index in visible_frames
    ]
    appearance_frame = read_frame(fixture_root, reference_frames[0])
    background, foreground, template = _appearance_from_visible_frame(appearance_frame)
    x_intercept, x_slope = _fit_line([(index, center[0]) for index, center in observations])
    y_intercept, y_slope = _fit_line([(index, center[1]) for index, center in observations])

    expected = {
        frame: _predicted_frame(x_intercept + x_slope * frame, y_intercept + y_slope * frame,
                                background, foreground, template)
        for frame in target_frames
    }
    scores: dict[str, float] = {}
    for candidate in candidates:
        candidate_id = candidate.get("id")
        frame_paths = candidate.get("frames")
        if not isinstance(candidate_id, str) or not isinstance(frame_paths, list) or len(frame_paths) != 64:
            raise ValueError("candidate metadata is invalid")
        total_error = 0
        value_count = 0
        for frame in target_frames:
            _, _, actual = _parse_ppm(read_frame(fixture_root, frame_paths[frame]))
            predicted = expected[frame]
            for y in range(16, 48):
                for x in range(16, 48):
                    offset = (y * 64 + x) * 3
                    total_error += sum(abs(actual[offset + channel] - predicted[offset + channel])
                                       for channel in range(3))
                    value_count += 3
        if value_count == 0:
            raise ValueError("interpolation baseline has no target pixels")
        score = total_error / value_count
        if not math.isfinite(score):
            raise ValueError("interpolation baseline produced a non-finite score")
        scores[candidate_id] = score
    return scores
