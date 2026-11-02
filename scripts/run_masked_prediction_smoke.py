#!/usr/bin/env python3
"""Explicit one-fixture V-JEPA 2 masked-prediction smoke; dry-run is the default."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import socket
import stat
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from masked_prediction import (evaluate_ranking, rank_candidates, require_supported_loss_contract,
                               token_partition)  # noqa: E402


def _progress(event: str, **fields) -> None:
    print(json.dumps({"runner_event": event, "elapsed_seconds": time.monotonic() - _RUN_STARTED,
                      **fields}, sort_keys=True), file=sys.stderr, flush=True)


_RUN_STARTED = time.monotonic()
MAX_MANIFEST_BYTES = 128 * 1024
MAX_ANSWER_KEY_BYTES = 16 * 1024
MAX_FRAME_BYTES = 16 * 1024
EXPECTED_CANDIDATES = 4
EXPECTED_FRAMES = 64
HIDDEN_FRAMES = range(30, 34)


def _install_best_effort_network_guard() -> None:
    """Block ordinary Python socket connects before importing model libraries."""
    def denied(*_args, **_kwargs):
        raise RuntimeError("network access is disabled for this local smoke")

    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    # Hub clients honor these; they are defense in depth, not OS isolation.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _fixture_file(fixture: Path, relative: str, *, max_bytes: int) -> Path:
    """Resolve one bounded regular fixture file without following any symlink."""
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ValueError("fixture path must be a non-empty POSIX relative path")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or rel.as_posix() != relative or any(part in (".", "..") for part in rel.parts):
        raise ValueError("fixture path must be canonical and remain inside the fixture directory")
    root = Path(fixture)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("fixture root must be a regular directory, not a symlink")
    root = root.resolve(strict=True)
    candidate = root
    for part in rel.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("fixture paths may not contain symlinks")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        info = resolved.lstat()
    except (OSError, ValueError) as error:
        raise ValueError("fixture path is missing or escapes its fixture directory") from error
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or info.st_size > max_bytes:
        raise ValueError("fixture file must be a non-empty regular file within its size limit")
    return resolved


def _read_fixture_json(fixture: Path, relative: str, *, max_bytes: int) -> dict:
    path = _fixture_file(fixture, relative, max_bytes=max_bytes)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid fixture JSON: {relative}") from error
    if not isinstance(value, dict):
        raise ValueError(f"fixture JSON must contain an object: {relative}")
    return value


def _validate_fixture_manifest(fixture: Path, manifest: dict) -> list[dict]:
    base_fields = {
        "schema_version", "kind", "width", "height", "frames", "target_frame_range",
        "visible_context_frame_ranges", "candidate_count", "candidate_order_seed",
        "ground_truth_file", "model_input_reference", "candidates", "provenance",
    }
    schema_version = manifest.get("schema_version")
    if (schema_version == 3 and set(manifest) != base_fields | {"visible_trajectory_template"}
            or schema_version == 2 and set(manifest) != base_fields
            or schema_version not in (2, 3)):
        raise ValueError("fixture manifest fields do not match the fixed synthetic schema")
    if manifest.get("kind") != "synthetic-ball-middle-gap":
        raise ValueError("unsupported synthetic fixture schema")
    if manifest.get("width") != 64 or manifest.get("height") != 64 or manifest.get("frames") != EXPECTED_FRAMES:
        raise ValueError("fixture dimensions or frame count are unsupported")
    if manifest.get("target_frame_range") != [30, 33] or manifest.get("visible_context_frame_ranges") != [[0, 29], [34, 63]]:
        raise ValueError("fixture hidden and visible frame ranges do not match the fixed evaluation")
    if manifest.get("ground_truth_file") != "answer_key.json":
        raise ValueError("fixture answer key path must be the fixed local filename")
    if (manifest.get("candidate_count") != EXPECTED_CANDIDATES
            or type(manifest.get("candidate_order_seed")) is not int
            or manifest.get("provenance") != "generated deterministically by scripts/generate_ball_gap.py; no real media"):
        raise ValueError("fixture provenance or candidate metadata is invalid")
    if schema_version == 3:
        trajectory = manifest.get("visible_trajectory_template")
        if (not isinstance(trajectory, dict)
                or set(trajectory) != {"x_start", "x_velocity", "base_y"}
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in trajectory.values())
                or not 12.0 <= trajectory["x_start"] <= 14.0
                or not 0.34 <= trajectory["x_velocity"] <= 0.46
                or not 26.0 <= trajectory["base_y"] <= 41.0):
            raise ValueError("fixture trajectory metadata is outside the generator bounds")
    reference = manifest.get("model_input_reference")
    expected_reference = {
        "resolution": 256, "patch_size": 16, "tubelet_size": 2,
        "token_grid": [32, 16, 16], "target_tubelets": [15, 16],
        "target_rows": [4, 11], "target_columns": [4, 11],
        "expected_target_tokens": 128, "expected_context_tokens": 8064,
    }
    if reference != expected_reference:
        raise ValueError("fixture token mask does not match the fixed tubelet and patch mask")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != EXPECTED_CANDIDATES:
        raise ValueError("fixture must contain exactly four candidates")
    ids = [item.get("id") for item in candidates if isinstance(item, dict)]
    if len(ids) != EXPECTED_CANDIDATES or ids != [f"candidate-{index}" for index in range(EXPECTED_CANDIDATES)]:
        raise ValueError("candidate IDs must be the four fixed candidate names")
    seen_paths: set[str] = set()
    for candidate in candidates:
        if set(candidate) != {"id", "frames"}:
            raise ValueError("candidate fields do not match the fixed synthetic schema")
        frame_paths = candidate.get("frames")
        if not isinstance(frame_paths, list) or len(frame_paths) != EXPECTED_FRAMES:
            raise ValueError("every candidate must contain exactly 64 frames")
        expected_paths = [f"frames/{candidate['id']}/{frame:03d}.ppm"
                          for frame in range(EXPECTED_FRAMES)]
        if frame_paths != expected_paths or len(set(frame_paths)) != EXPECTED_FRAMES:
            raise ValueError("candidate frame paths must be unique")
        for relative in frame_paths:
            if relative in seen_paths:
                raise ValueError("candidate frame paths must not overlap")
            seen_paths.add(relative)
            _fixture_file(fixture, relative, max_bytes=MAX_FRAME_BYTES)
    return candidates


def _validate_candidate_frames(fixture: Path, candidates: list[dict]) -> None:
    reference = candidates[0]["frames"]
    for candidate in candidates[1:]:
        for frame_index, (reference_path, candidate_path) in enumerate(zip(reference, candidate["frames"])):
            first = _fixture_file(fixture, reference_path, max_bytes=MAX_FRAME_BYTES).read_bytes()
            other = _fixture_file(fixture, candidate_path, max_bytes=MAX_FRAME_BYTES).read_bytes()
            if frame_index not in HIDDEN_FRAMES:
                _validate_frame_pair(first, other, hidden=False)
                continue
            _validate_frame_pair(first, other, hidden=True)


def _validate_frame_pair(first: bytes, other: bytes, *, hidden: bool) -> None:
    """Check one pair of fixed-size PPM frames without importing image packages."""
    header = b"P6\n64 64\n255\n"
    if not first.startswith(header) or not other.startswith(header):
        raise ValueError("fixture frame must use the fixed 64x64 binary PPM header")
    first_pixels, other_pixels = first[len(header):], other[len(header):]
    if len(first_pixels) != 64 * 64 * 3 or len(other_pixels) != 64 * 64 * 3:
        raise ValueError("fixture frame payload must contain exactly 64x64 RGB bytes")
    if not hidden:
        if first != other:
            raise ValueError("candidate visible frames must be byte-identical")
        return
    changed = False
    for pixel in range(64 * 64):
        start = pixel * 3
        if first_pixels[start:start + 3] == other_pixels[start:start + 3]:
            continue
        changed = True
        y, x = divmod(pixel, 64)
        if not (16 <= x < 48 and 16 <= y < 48):
            raise ValueError("candidate differences must remain within the hidden target patch")
    if not changed:
        raise ValueError("each candidate must differ from the reference inside the hidden interval")


def _load_ppm_bytes(data: bytes):
    import numpy as np

    header, payload = data.split(b"\n255\n", 1)
    fields = header.split()
    if fields[0] != b"P6" or tuple(map(int, fields[1:])) != (64, 64) or len(payload) != 64 * 64 * 3:
        raise ValueError("fixture frame must be a 64x64 binary PPM")
    return np.frombuffer(payload, dtype=np.uint8).reshape(64, 64, 3).copy()


def _load_ppm(path: Path):
    return _load_ppm_bytes(path.read_bytes())


def _predictor_forward_for_ids(predictor, context, context_ids, target_ids, *, return_trace=False):
    """Apply the loaded V-JEPA 2 predictor blocks to explicit arbitrary target IDs."""
    import torch

    batch = context.shape[0]
    context_embed = predictor.predictor_embed(context)
    target_embed = predictor.mask_tokens[1].expand(batch, target_ids.shape[1], -1)
    joined = torch.cat([context_embed, target_embed], dim=1)
    ids = torch.cat([context_ids, target_ids], dim=1).to(device=joined.device, dtype=torch.long)
    order = torch.argsort(ids, dim=1)
    joined = torch.gather(joined, 1, order.unsqueeze(-1).expand(-1, -1, joined.shape[-1]))
    sorted_ids = torch.gather(ids, 1, order)
    block_input_shapes = []
    block_output_shapes = []
    for block in predictor.predictor_blocks:
        block_input_shapes.append(list(joined.shape))
        parameters = inspect.signature(block.forward).parameters
        if {"T", "H_patches", "W_patches"}.issubset(parameters):
            joined = block(joined, mask=sorted_ids, T=32, H_patches=16, W_patches=16)
        elif {"height", "width"}.issubset(parameters):
            joined = block(joined, mask=sorted_ids, height=16, width=16)
        else:
            raise TypeError("predictor block does not expose a known V-JEPA 2 RoPE grid signature")
        block_output_shapes.append(list(joined.shape))
    joined = predictor.predictor_norm(joined)
    positions = torch.searchsorted(sorted_ids.contiguous(), target_ids.contiguous())
    if not torch.equal(torch.gather(sorted_ids, 1, positions), target_ids):
        raise ValueError("predictor token order did not preserve the explicit target IDs")
    target_hidden = torch.gather(joined, 1, positions.unsqueeze(-1).expand(-1, -1, joined.shape[-1]))
    output = predictor.predictor_proj(target_hidden)
    if return_trace:
        return output, {
            "sorted_token_ids": sorted_ids.detach().cpu().tolist(),
            "block_input_shapes": block_input_shapes,
            "block_output_shapes": block_output_shapes,
            "target_positions": positions.detach().cpu().tolist(),
            "target_hidden_shape": list(target_hidden.shape),
            "output_shape": list(output.shape),
        }
    return output


def _load_identity(identity_path: Path, checkpoint: Path) -> dict:
    from run_masked_prediction_pilot import _read_identity_safe
    return _read_identity_safe(identity_path, checkpoint)


def run(checkpoint: Path, identity_record: Path, fixture: Path, device_name: str,
        verified_loss_exp: float, processor_config: Path,
        compatibility_loader: Path) -> dict:
    # Official pinned ViT-L/16 leaf configs use loss_exp=1.0; the caller must still
    # pass the verified value explicitly because the checkpoint identity record does
    # not authenticate the provider artifact. Only exponent 1 maps to this MAE scorer.
    require_supported_loss_contract(verified_loss_exp, "layer_norm_last_dim")
    _install_best_effort_network_guard()
    from run_masked_prediction_pilot import (
        _load_verified_loader_callable, _processor_from_pinned_config,
        _read_bounded_file,
    )
    identity = _load_identity(identity_record, checkpoint)
    config_source = _read_bounded_file(processor_config, max_bytes=128 * 1024)
    torch, processor, config_hash, config_bytes = _processor_from_pinned_config(config_source)
    load_official_bundle, loader_hash = _load_verified_loader_callable(compatibility_loader)
    import numpy as np
    import torch.nn.functional as F

    manifest = _read_fixture_json(fixture, "fixture.json", max_bytes=MAX_MANIFEST_BYTES)
    candidates = _validate_fixture_manifest(fixture, manifest)
    _validate_candidate_frames(fixture, candidates)
    ids = manifest["model_input_reference"]
    context_ids, target_ids = token_partition(
        temporal_tokens=32,
        height_tokens=16,
        width_tokens=16,
        target_temporal=ids["target_tubelets"],
        target_rows=range(ids["target_rows"][0], ids["target_rows"][1] + 1),
        target_cols=range(ids["target_columns"][0], ids["target_columns"][1] + 1),
    )
    device = torch.device(device_name)
    _progress("checkpoint_hash_started", checkpoint=str(checkpoint))
    _progress("checkpoint_sha256_verified", checkpoint_bytes=identity["checkpoint_bytes"])
    _progress("strict_bundle_load_started")
    bundle = load_official_bundle(checkpoint, device=device, dtype=torch.float32)
    bundle.encoder.eval()
    bundle.target_encoder.eval()
    bundle.predictor.eval()
    _progress("weights_only_strict_bundle_loaded", encoder=True, target_encoder=True, predictor=True)

    context_frames = [_load_ppm(_fixture_file(fixture, relative, max_bytes=MAX_FRAME_BYTES))
                      for relative in candidates[0]["frames"]]
    context_encoded = processor([context_frames], do_resize=True, do_center_crop=True,
                                 input_data_format="channels_last", return_tensors="pt")
    context_videos = context_encoded["pixel_values_videos"].to(device=device, dtype=torch.float32)
    if tuple(context_videos.shape) != (1, 64, 3, 256, 256):
        raise ValueError(f"unexpected preprocessor shape {tuple(context_videos.shape)}")
    context_mask = torch.tensor(context_ids, device=device, dtype=torch.long).unsqueeze(0)
    target_mask = torch.tensor(target_ids, device=device, dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        _progress("context_encoder_call_started", context_encoder_calls=1)
        context_features = bundle.encoder(context_videos.permute(0, 2, 1, 3, 4), mask=context_mask)
        _progress("context_encoder_call_returned", output_shape=list(context_features.shape))
        if context_features.ndim != 3 or tuple(context_features.shape) != (1, len(context_ids), 1024):
            raise ValueError(f"unexpected context encoder shape {tuple(context_features.shape)}; expected [1,{len(context_ids)},1024]")
        _progress("predictor_call_started", predictor_calls=1)
        predicted = _predictor_forward_for_ids(bundle.predictor, context_features, context_mask, target_mask)
        _progress("predictor_call_returned", output_shape=list(predicted.shape))
        if predicted.ndim != 3 or tuple(predicted.shape) != (1, len(target_ids), 1024):
            raise ValueError(f"unexpected predictor output shape {tuple(predicted.shape)}; expected [1,{len(target_ids)},1024]")
        predicted_rows = predicted.float()[0].cpu().tolist()
        _progress("context_predictor_completed", context_encoder_calls=1, predictor_calls=1,
                  context_shape=list(context_features.shape), prediction_shape=list(predicted.shape))
        del context_features, context_videos, context_frames
        scored_targets = {}
        index = target_mask.unsqueeze(-1).expand(-1, -1, predicted.shape[-1])
        for target_index, candidate in enumerate(candidates, start=1):
            frames = [_load_ppm(_fixture_file(fixture, relative, max_bytes=MAX_FRAME_BYTES))
                      for relative in candidate["frames"]]
            encoded = processor([frames], do_resize=True, do_center_crop=True,
                                input_data_format="channels_last", return_tensors="pt")
            videos = encoded["pixel_values_videos"].to(device=device, dtype=torch.float32)
            _progress("target_encoder_call_started", target_encoder_calls=target_index,
                      candidate_id=candidate["id"])
            full = bundle.target_encoder(videos.permute(0, 2, 1, 3, 4))
            _progress("target_encoder_call_returned", target_encoder_calls=target_index,
                      candidate_id=candidate["id"], output_shape=list(full.shape))
            if full.ndim != 3 or tuple(full.shape) != (1, 8192, 1024):
                raise ValueError(f"unexpected target encoder shape {tuple(full.shape)}; expected [1,8192,1024]")
            target = full.gather(1, index)
            if tuple(target.shape) != (1, len(target_ids), 1024):
                raise ValueError(f"unexpected gathered target shape {tuple(target.shape)}")
            target = F.layer_norm(target, (target.shape[-1],)).float()[0]
            scored_targets[candidate["id"]] = target.cpu().tolist()
            _progress("target_encoder_completed", target_encoder_calls=target_index,
                      candidate_id=candidate["id"], output_shape=[1, len(target_ids), 1024])
            del full, target, videos, frames, encoded
    ranking = rank_candidates(predicted_rows, scored_targets)
    # Read evaluator-only truth after all model scoring has completed.
    answer_key = _read_fixture_json(fixture, "answer_key.json", max_bytes=MAX_ANSWER_KEY_BYTES)
    answer_key_fields = {"schema_version", "correct_candidate_id", "rule", "claim_scope",
                         "ambiguous_by_visible_context", "ambiguity_note"}
    if (set(answer_key) != answer_key_fields
            or answer_key.get("schema_version") != 1
            or answer_key.get("correct_candidate_id") not in {item["id"] for item in candidates}
            or not isinstance(answer_key.get("ambiguous_by_visible_context"), bool)
            or answer_key.get("rule") != "constant horizontal velocity and constant vertical position through the gap"
            or answer_key.get("claim_scope") != "synthetic generator rule only; not a unique real-world inference"):
        raise ValueError("answer key does not match the fixed synthetic scoring contract")
    evaluation = evaluate_ranking(
        ranking,
        answer_key["correct_candidate_id"],
        visible_context_ambiguous=answer_key["ambiguous_by_visible_context"],
    )
    _progress("ranking_completed", candidate_count=len(scored_targets), ranking=ranking,
              evaluation=evaluation)
    del bundle
    if hasattr(torch, "mps"):
        torch.mps.empty_cache() if device.type == "mps" else None
    return {
        "checkpoint_bytes": identity["checkpoint_bytes"],
        "checkpoint_sha256_match": True,
        "model": "V-JEPA 2 ViT-L/16 256px",
        "device": str(device),
        "fixture": manifest["kind"],
        "candidate_count": len(scored_targets),
        "context_tokens": len(context_ids),
        "target_tokens": len(target_ids),
        "ranking_lower_score_is_better": ranking,
        "evaluation": evaluation,
        "real_media_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true",
                      help="actually read model weights and run one synthetic 4-candidate inference")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--identity-record", type=Path)
    parser.add_argument("--verified-loss-exp", type=float,
                        help="loss_exp from the official config for this exact checkpoint; only 1.0 is supported")
    parser.add_argument("--processor-config", type=Path)
    parser.add_argument("--compatibility-loader", type=Path)
    parser.add_argument("--fixture", type=Path, default=PROJECT_DIR / "fixtures" / "ball-gap-v2")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()

    context_ids, target_ids = token_partition(
        temporal_tokens=32, height_tokens=16, width_tokens=16,
        target_temporal=(15, 16), target_rows=range(4, 12), target_cols=range(4, 12),
    )
    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "model_weights_read": False,
            "candidate_count": 4,
            "frames_per_candidate": 64,
            "model_input": [1, 64, 3, 256, 256],
            "context_tokens": len(context_ids),
            "target_tokens": len(target_ids),
            "checkpoint_recorded_bytes": 5127726842,
            "network_guard": "child-only sandbox-exec deny network* + offline env + Python socket guard",
            "guardian_outside_additional_network_sandbox": True,
            "passes_after_approval": {
                "masked_context_encoder_and_predictor": 1,
                "full_target_encoder_sequential": 4,
            },
        }, indent=2))
        return 0

    if args.checkpoint is None or args.identity_record is None:
        parser.error("--execute requires --checkpoint and --identity-record")
    if args.verified_loss_exp is None:
        parser.error("--execute requires --verified-loss-exp copied from the official config for this checkpoint")
    if args.processor_config is None or args.compatibility_loader is None:
        parser.error("--execute requires explicit --processor-config and --compatibility-loader paths")
    if os.environ.get("JEPA_GUARDED_RUN") != "1":
        parser.error("--execute must be launched via scripts/run_guarded_smoke.py")
    if not args.fixture.is_dir():
        parser.error("fixture directory not found")
    print("RUNNER_STARTED", flush=True)
    result = run(args.checkpoint, args.identity_record, args.fixture, args.device,
                 args.verified_loss_exp, args.processor_config,
                 args.compatibility_loader)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
