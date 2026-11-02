#!/usr/bin/env python3
"""Run a fixed, offline multi-seed synthetic masked-prediction pilot."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import time
from contextlib import contextmanager
import types

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import masked_prediction_baselines as baselines  # noqa: E402
from masked_prediction import (evaluate_ranking_metrics, rank_candidates,
                               require_supported_loss_contract, token_partition)  # noqa: E402
from run_masked_prediction_smoke import (  # noqa: E402
    EXPECTED_CANDIDATES, EXPECTED_FRAMES, MAX_ANSWER_KEY_BYTES, MAX_FRAME_BYTES,
    MAX_MANIFEST_BYTES, _fixture_file, _install_best_effort_network_guard,
    _load_ppm_bytes, _predictor_forward_for_ids, _validate_frame_pair,
    _validate_fixture_manifest,
)

FIXED_SEEDS = tuple(range(20261101, 20261109))
DEFAULT_COUNT = 4
MAX_COUNT = 8
MAX_FILES = 2064
MAX_TOTAL_BYTES = 40 * 1024 * 1024
MAX_FIRST_FIXTURE_SECONDS = 180.0
MAX_PROGRESS_EVENTS = 192
MAX_PROGRESS_EVENT_BYTES = 1024
TIE_TOLERANCE = 1e-6
LOSS_EXP = 1.0
DEVICE = "cpu"
PROCESSOR_CONFIG_SHA256 = "738a3476c6191bd054541f641cd39103b854490c723dd6a8d66a0e80a4e6deae"
COMPATIBILITY_LOADER_SHA256 = "f0b195d1e56a9d4391d538be1ba24f62508074cbbfa7dd5182f29e429cff7155"
FIXTURE_ROOT = PROJECT_DIR / "fixtures" / "pilot-20261102-v1"
INDEX_PATH = FIXTURE_ROOT / "INDEX.json"
REPORT_JSON = PROJECT_DIR / "reports" / "masked-prediction-pilot-20261102-v1.json"
REPORT_MARKDOWN = PROJECT_DIR / "reports" / "masked-prediction-pilot-20261102-v1.md"
METHODS = ("seeded_random", "visible_context_encoder_only_tie",
           "image_space_constant_velocity_interpolation", "masked_context_predictor")


class PilotProgress:
    """Emit bounded, flushed phase events without tensor, answer, or path content."""

    _stages = {
        "run_started", "fixtures_generated", "bundle_loaded", "index_written",
        "fixture_started", "candidate_preprocessed", "preprocess_gate_passed",
        "preprocess_failed", "baseline_scores_completed", "forward_started",
        "forward_returned", "forward_failed", "forward_interrupted", "fixture_scores_completed",
        "answer_evaluation_started", "answer_evaluation_completed", "run_completed",
        "fixture_failed", "run_failed",
    }
    _forward_kinds = {"encoder", "predictor", "target_encoder"}
    _safe_errors = {
        "visible_context_mismatch", "processor_shape_mismatch", "processor_error",
        "fixture_input_error", "baseline_error", "forward_error", "ranking_error",
        "unclassified_failure", "interrupted",
    }

    def __init__(self, stream=None):
        self.stream = sys.stdout if stream is None else stream
        self.event_count = 0
        self.completed_fixture_scores = 0
        self.attempted = {kind: 0 for kind in self._forward_kinds}
        self.returned = {kind: 0 for kind in self._forward_kinds}
        self.fixture_attempted = {kind: 0 for kind in self._forward_kinds}
        self.fixture_returned = {kind: 0 for kind in self._forward_kinds}
        self.fixture_index = None
        self.seed = None
        self.fixture_failure_logged = False

    def _emit(self, stage: str, *, candidate_id: str | None = None,
              forward_kind: str | None = None, safe_error_code: str | None = None,
              next_completed_count: int | None = None, global_event: bool = False) -> None:
        if stage not in self._stages:
            raise ValueError("progress stage is outside the fixed event set")
        if forward_kind is not None and forward_kind not in self._forward_kinds:
            raise ValueError("progress forward kind is outside the fixed event set")
        if candidate_id is not None and candidate_id not in {
                f"candidate-{index}" for index in range(EXPECTED_CANDIDATES)}:
            raise ValueError("progress candidate ID is outside the fixed event set")
        if safe_error_code is not None and safe_error_code not in self._safe_errors:
            raise ValueError("progress error code is outside the fixed event set")
        if self.event_count >= MAX_PROGRESS_EVENTS:
            raise RuntimeError("pilot progress event limit exceeded")
        attempted = (sum(self.attempted.values()) if global_event else
                     self.fixture_attempted[forward_kind] if forward_kind is not None else
                     sum(self.fixture_attempted.values()))
        returned = (sum(self.returned.values()) if global_event else
                    self.fixture_returned[forward_kind] if forward_kind is not None else
                    sum(self.fixture_returned.values()))
        event = {
            "fixture_index": None if global_event else self.fixture_index,
            "seed": None if global_event else self.seed,
            "stage": stage,
            "candidate_id_if_known": candidate_id,
            "forward_kind": forward_kind,
            "attempted_count": attempted,
            "returned_count": returned,
            "completed_fixture_scores": (self.completed_fixture_scores
                                          if next_completed_count is None
                                          else next_completed_count),
            "safe_error_code": safe_error_code,
        }
        encoded = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        if len(encoded.encode("utf-8")) > MAX_PROGRESS_EVENT_BYTES:
            raise RuntimeError("pilot progress event exceeded its fixed size limit")
        self.stream.write(encoded)
        self.stream.flush()
        self.event_count += 1
        if stage in {"preprocess_failed", "forward_failed", "forward_interrupted", "fixture_failed"}:
            self.fixture_failure_logged = True

    def emit_run(self, stage: str, safe_error_code: str | None = None) -> None:
        self._emit(stage, safe_error_code=safe_error_code, global_event=True)

    def begin_fixture(self, fixture_index: int, seed: int) -> None:
        if (type(fixture_index) is not int or not 0 <= fixture_index < len(FIXED_SEEDS)
                or seed != FIXED_SEEDS[fixture_index]):
            raise ValueError("progress fixture identity is outside the fixed seed set")
        self.fixture_index = fixture_index
        self.seed = seed
        self.fixture_attempted = {kind: 0 for kind in self._forward_kinds}
        self.fixture_returned = {kind: 0 for kind in self._forward_kinds}
        self.fixture_failure_logged = False
        self._emit("fixture_started")

    def fail_current_fixture_if_needed(self) -> None:
        if (self.event_count < MAX_PROGRESS_EVENTS and self.fixture_index is not None
                and not self.fixture_failure_logged):
            self._emit("fixture_failed", safe_error_code="unclassified_failure")

    def forward_started(self, kind: str, candidate_id: str | None = None) -> None:
        if kind not in self._forward_kinds:
            raise ValueError("unknown model forward kind")
        self.attempted[kind] += 1
        self.fixture_attempted[kind] += 1
        self._emit("forward_started", candidate_id=candidate_id, forward_kind=kind)

    def forward_returned(self, kind: str, candidate_id: str | None = None) -> None:
        if kind not in self._forward_kinds:
            raise ValueError("unknown model forward kind")
        self.returned[kind] += 1
        self.fixture_returned[kind] += 1
        self._emit("forward_returned", candidate_id=candidate_id, forward_kind=kind)

    def forward_failed(self, kind: str, candidate_id: str | None = None) -> None:
        self._emit("forward_failed", candidate_id=candidate_id, forward_kind=kind,
                   safe_error_code="forward_error")

    def forward_interrupted(self, kind: str, candidate_id: str | None = None) -> None:
        self._emit("forward_interrupted", candidate_id=candidate_id, forward_kind=kind,
                   safe_error_code="interrupted")

    def mark_fixture_scores_completed(self) -> None:
        next_count = self.completed_fixture_scores + 1
        self._emit("fixture_scores_completed", next_completed_count=next_count)
        self.completed_fixture_scores = next_count

    def summary(self) -> dict:
        return {
            "attempted": dict(self.attempted),
            "returned": dict(self.returned),
            "completed_fixture_scores": self.completed_fixture_scores,
            "events_emitted": self.event_count,
        }


def _tracked_forward(progress: PilotProgress, kind: str, function,
                     candidate_id: str | None = None):
    progress.forward_started(kind, candidate_id)
    try:
        result = function()
    except KeyboardInterrupt:
        progress.forward_interrupted(kind, candidate_id)
        raise
    except BaseException:
        progress.forward_failed(kind, candidate_id)
        raise
    progress.forward_returned(kind, candidate_id)
    return result


def _record_safe_run_failure(progress: PilotProgress, *, interrupted: bool) -> None:
    try:
        progress.fail_current_fixture_if_needed()
        if progress.event_count < MAX_PROGRESS_EVENTS:
            code = "interrupted" if interrupted else "unclassified_failure"
            progress.emit_run("run_failed", safe_error_code=code)
        else:
            print("PILOT_PROGRESS_LIMIT", flush=True)
    except Exception:
        print("PILOT_PROGRESS_UNAVAILABLE", flush=True)


def _sha256_file(path: Path, *, max_bytes: int | None = None) -> tuple[str, int]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("the platform cannot provide required no-follow file reads")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError("expected a readable non-symlink file") from error
    info = os.fstat(descriptor)
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or (max_bytes is not None and info.st_size > max_bytes)):
        os.close(descriptor)
        raise ValueError("file type, link count, or size violates the fixed pilot bound")
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), info.st_size


def _read_bounded_file(path: Path, *, max_bytes: int) -> bytes:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("the platform cannot provide required no-follow file reads")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError("expected a readable non-symlink file") from error
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > max_bytes:
        os.close(descriptor)
        raise ValueError("file type, link count, or size violates the fixed pilot bound")
    with os.fdopen(descriptor, "rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) != info.st_size or len(data) > max_bytes:
        raise ValueError("file changed while being read")
    return data


def _read_pinned_bytes(path: Path, *, expected_sha256: str, max_bytes: int) -> bytes:
    data = _read_bounded_file(path, max_bytes=max_bytes)
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError("pinned resource SHA-256 mismatch")
    return data


def validate_loader_import_closure(source: bytes) -> None:
    """Reject package-relative or undeclared imports before executing pinned loader bytes."""
    try:
        tree = ast.parse(source, filename="verified-loader-bytes")
    except SyntaxError as error:
        raise ValueError("pinned loader source is not valid Python") from error
    allowed = {"__future__", "dataclasses", "math", "pathlib", "typing",
               "torch", "torch.nn", "torch.nn.functional"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                raise ValueError("relative loader imports would require unregistered package initialization")
            modules = [node.module or ""]
        else:
            continue
        if any(module not in allowed for module in modules):
            raise ValueError("pinned loader imports a module outside the reviewed closure")
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "__import__"):
            raise ValueError("dynamic loader imports are not in the reviewed closure")


def _visible_pixel_regions(frame_index: int) -> tuple[tuple[slice, slice], ...]:
    if frame_index not in range(30, 34):
        return ((slice(0, 256), slice(0, 256)),)
    # Mask tokens cover temporal tubelets 15-16 and spatial rows/columns 4-11.
    # At 16px patches this is [64,192); row/column 192 is visible and checked.
    return (
        (slice(0, 64), slice(0, 256)),
        (slice(192, 256), slice(0, 256)),
        (slice(64, 192), slice(0, 64)),
        (slice(64, 192), slice(192, 256)),
    )


def assert_preprocessed_visible_context_equal(reference, candidate, *, tensor_equal) -> None:
    """Fail before any encoder call if transformed visible pixels differ outside mask tokens."""
    expected_shape = (1, 64, 3, 256, 256)
    if tuple(reference.shape) != expected_shape or tuple(candidate.shape) != expected_shape:
        raise ValueError("preprocessed candidate tensors must have shape [1,64,3,256,256]")
    for frame_index in range(64):
        for rows, columns in _visible_pixel_regions(frame_index):
            left = reference[:, frame_index, :, rows, columns]
            right = candidate[:, frame_index, :, rows, columns]
            if not tensor_equal(left, right):
                raise ValueError(
                    "preprocessed candidate context differs outside the fixed masked token region")


def _read_fixture_json_safe(root: Path, relative: str, *, max_bytes: int) -> dict:
    path = _fixture_file(root, relative, max_bytes=max_bytes)
    try:
        value = json.loads(_read_bounded_file(path, max_bytes=max_bytes).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid fixture JSON") from error
    if not isinstance(value, dict):
        raise ValueError("fixture JSON must contain an object")
    return value


def _read_identity_safe(identity_path: Path, checkpoint: Path) -> dict:
    try:
        identity = json.loads(_read_bounded_file(identity_path, max_bytes=64 * 1024).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("identity record is not valid bounded JSON") from error
    required = {"checkpoint_path", "checkpoint_bytes", "sha256"}
    if (not isinstance(identity, dict) or not required.issubset(identity)
            or type(identity.get("checkpoint_bytes")) is not int
            or identity["checkpoint_bytes"] <= 0
            or not isinstance(identity.get("sha256"), str)
            or len(identity["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in identity["sha256"])):
        raise ValueError("identity record is missing checkpoint path, size, or SHA-256")
    if not checkpoint.is_absolute() or not identity_path.is_absolute():
        raise ValueError("checkpoint and identity must be explicit absolute paths")
    if checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.stat().st_nlink != 1:
        raise ValueError("checkpoint must be a regular unlinked file")
    recorded = Path(identity["checkpoint_path"]).expanduser().resolve(strict=True)
    actual = checkpoint.expanduser().resolve(strict=True)
    checkpoint_hash, checkpoint_bytes = _sha256_file(checkpoint)
    if (recorded != actual or checkpoint_bytes != identity["checkpoint_bytes"]
            or checkpoint_hash != identity["sha256"]):
        raise ValueError("checkpoint identity path, byte count, or SHA-256 mismatch")
    return identity


def _write_new(path: Path, data: bytes) -> None:
    """Create a new fixed output without following links or replacing prior evidence."""
    path.parent.mkdir(parents=False, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _validate_count(count: int) -> tuple[int, ...]:
    if type(count) is not int or not DEFAULT_COUNT <= count <= MAX_COUNT:
        raise ValueError(f"fixture count must be between {DEFAULT_COUNT} and {MAX_COUNT}")
    return FIXED_SEEDS[:count]


@contextmanager
def _first_fixture_deadline(enabled: bool):
    if not enabled:
        yield
        return
    if not all(hasattr(signal, name) for name in ("SIGALRM", "ITIMER_REAL", "setitimer", "getitimer")):
        raise RuntimeError("platform cannot enforce the first-fixture wall-clock stop")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise RuntimeError("an existing wall-clock timer prevents bounded first-fixture execution")
    # The OS default terminates even if a native/PyTorch call does not return to
    # Python to dispatch a custom handler. The outer guardian then waits/reports it.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, MAX_FIRST_FIXTURE_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def validate_fixed_resources(checkpoint: Path, identity_record: Path, loss_exp: float) -> None:
    if not Path(checkpoint).is_absolute() or not Path(identity_record).is_absolute():
        raise ValueError("checkpoint and identity must be explicit absolute paths")
    require_supported_loss_contract(loss_exp, "layer_norm_last_dim")
    if float(loss_exp) != LOSS_EXP:
        raise ValueError("pilot accepts only the fixed official loss exponent 1.0")


def require_absent_outputs(paths: tuple[Path, ...]) -> None:
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing pilot output: {path.name}")


def _require_fixed_output_directories() -> None:
    for directory in (PROJECT_DIR, PROJECT_DIR / "fixtures", PROJECT_DIR / "reports"):
        try:
            mode = directory.lstat().st_mode
        except OSError as error:
            raise ValueError("required fixed output directory is unavailable") from error
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError("fixed output path contains a symlink or non-directory component")


def _read_frame(root: Path, relative: str) -> bytes:
    return _read_bounded_file(
        _fixture_file(root, relative, max_bytes=MAX_FRAME_BYTES), max_bytes=MAX_FRAME_BYTES)


def _validate_candidate_frame_set(root: Path, candidates: list[dict]) -> None:
    reference = candidates[0]["frames"]
    for candidate in candidates[1:]:
        for index, (left, right) in enumerate(zip(reference, candidate["frames"])):
            _validate_frame_pair(
                _read_frame(root, left), _read_frame(root, right),
                hidden=index in range(30, 34))


def _validate_exact_fixture_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("fixture root must be a regular directory")
    if {entry.name for entry in root.iterdir()} != {"fixture.json", "answer_key.json", "frames"}:
        raise ValueError("fixture root contains missing or unregistered entries")
    frames_root = root / "frames"
    if frames_root.is_symlink() or not frames_root.is_dir():
        raise ValueError("frames path must be a regular directory")
    expected_candidates = {f"candidate-{index}" for index in range(EXPECTED_CANDIDATES)}
    if {entry.name for entry in frames_root.iterdir()} != expected_candidates:
        raise ValueError("frames directory contains missing or unregistered candidates")
    expected_frames = {f"{index:03d}.ppm" for index in range(EXPECTED_FRAMES)}
    for candidate_id in expected_candidates:
        candidate_root = frames_root / candidate_id
        if candidate_root.is_symlink() or not candidate_root.is_dir():
            raise ValueError("candidate frame path must be a regular directory")
        if {entry.name for entry in candidate_root.iterdir()} != expected_frames:
            raise ValueError("candidate frame directory contains missing or unregistered names")


def _fixture_record(root: Path, seed: int) -> dict:
    _validate_exact_fixture_tree(root)
    manifest = _read_fixture_json_safe(root, "fixture.json", max_bytes=MAX_MANIFEST_BYTES)
    candidates = _validate_fixture_manifest(root, manifest)
    if manifest.get("schema_version") != 3 or manifest.get("candidate_order_seed") != seed:
        raise ValueError("fixture schema or seed does not match the frozen pilot plan")
    _validate_candidate_frame_set(root, candidates)
    trajectory = manifest.get("visible_trajectory_template")
    if not isinstance(trajectory, dict):
        raise ValueError("fixture must include the bounded synthetic trajectory metadata")
    candidate_records = []
    total_bytes = 0
    total_files = 0
    for candidate in candidates:
        digest = hashlib.sha256()
        candidate_bytes = 0
        for index, relative in enumerate(candidate["frames"]):
            frame_path = _fixture_file(root, relative, max_bytes=MAX_FRAME_BYTES)
            frame_hash, size = _sha256_file(frame_path, max_bytes=MAX_FRAME_BYTES)
            total_bytes += size
            candidate_bytes += size
            total_files += 1
            digest.update(index.to_bytes(2, "big"))
            digest.update(size.to_bytes(8, "big"))
            digest.update(bytes.fromhex(frame_hash))
        candidate_records.append({
            "candidate_id": candidate["id"],
            "frame_count": EXPECTED_FRAMES,
            "bytes": candidate_bytes,
            "ordered_frame_sha256": digest.hexdigest(),
        })
    manifest_path = _fixture_file(root, "fixture.json", max_bytes=MAX_MANIFEST_BYTES)
    manifest_hash, manifest_bytes = _sha256_file(manifest_path, max_bytes=MAX_MANIFEST_BYTES)
    total_bytes += manifest_bytes
    total_files += 1
    if total_bytes > MAX_TOTAL_BYTES or total_files > 257:
        raise ValueError("fixture exceeds the registered per-seed count or byte bound")
    return {
        "seed": seed,
        "manifest_sha256": manifest_hash,
        "manifest_bytes": manifest_bytes,
        "file_count_excluding_answer": total_files,
        "bytes_excluding_answer": total_bytes,
        "visible_trajectory_template": trajectory,
        "candidates": candidate_records,
    }


def build_index(records: list[dict], *, source_sha256: dict[str, str],
                processor_config_sha256: str, compatibility_loader_sha256: str) -> dict:
    """Build an answer-free index from already validated manifests and frames."""
    if not records or len(records) > MAX_COUNT:
        raise ValueError("index must contain one to eight validated fixtures")
    seeds = [record.get("seed") for record in records]
    if len(set(seeds)) != len(seeds) or any(seed not in FIXED_SEEDS for seed in seeds):
        raise ValueError("index seeds must be unique members of the fixed allowlist")
    record_fields = {"seed", "manifest_sha256", "manifest_bytes", "file_count_excluding_answer",
                     "bytes_excluding_answer", "visible_trajectory_template", "candidates"}
    candidate_fields = {"candidate_id", "frame_count", "bytes", "ordered_frame_sha256"}
    if any(set(record) != record_fields
           or any(set(candidate) != candidate_fields for candidate in record["candidates"])
           for record in records):
        raise ValueError("index inputs contain unregistered fields or answer-bearing metadata")
    if (sum(record["file_count_excluding_answer"] for record in records) + len(records) > MAX_FILES
            or sum(record["bytes_excluding_answer"] for record in records)
            + len(records) * MAX_ANSWER_KEY_BYTES > MAX_TOTAL_BYTES):
        raise ValueError("index inputs exceed aggregate file or byte limits")
    for record in records:
        if (record["file_count_excluding_answer"] != 257
                or len(record["candidates"]) != EXPECTED_CANDIDATES
                or any(candidate["frame_count"] != EXPECTED_FRAMES
                       for candidate in record["candidates"])):
            raise ValueError("index input counts do not match the fixed fixture shape")
    trajectories = [
        (record["visible_trajectory_template"]["x_start"],
         record["visible_trajectory_template"]["x_velocity"],
         record["visible_trajectory_template"]["base_y"])
        for record in records
    ]
    if len(set(trajectories)) != len(trajectories):
        raise ValueError("every selected seed must produce a distinct visible trajectory")
    if set(source_sha256) != {"generator", "pilot", "smoke", "baselines"}:
        raise ValueError("index source digests do not match the fixed source set")
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
           for value in [*source_sha256.values(), processor_config_sha256,
                         compatibility_loader_sha256]):
        raise ValueError("source and config digests must be lowercase SHA-256 values")
    return {
        "schema_version": 1,
        "kind": "synthetic-ball-middle-gap-pilot",
        "seed_count": len(records),
        "seeds": records,
        "source_sha256": dict(source_sha256),
        "processor_config_sha256": processor_config_sha256,
        "compatibility_loader_sha256": compatibility_loader_sha256,
        "contains_answer_or_correct_id": False,
    }


def evaluate_after_scoring(
    completed_scores: list[dict],
    answer_reader,
) -> tuple[list[dict], dict]:
    """Only join labels after every fixture and method has completed scoring."""
    if not completed_scores:
        raise ValueError("no complete fixture scores are available")
    required_item_fields = {"seed", "root", "candidate_ids", "scores"}
    for item in completed_scores:
        candidate_ids = item.get("candidate_ids")
        if (set(item) != required_item_fields or not isinstance(candidate_ids, list)
                or len(candidate_ids) != EXPECTED_CANDIDATES
                or len(set(candidate_ids)) != EXPECTED_CANDIDATES
                or set(item.get("scores", {})) != set(METHODS)):
            raise ValueError("fixture scores are incomplete or contain evaluator-only fields")
        for values in item["scores"].values():
            if (set(values) != set(candidate_ids)
                    or any(type(score) not in (int, float) or not math.isfinite(score)
                           for score in values.values())):
                raise ValueError("every method must finish four finite candidate scores before evaluation")
    output = []
    correct_positions: dict[str, int] = {}
    for item in completed_scores:
        answer = answer_reader(item["root"])
        candidate_ids = item["candidate_ids"]
        if (set(answer) != {"schema_version", "correct_candidate_id", "rule",
                            "claim_scope", "ambiguous_by_visible_context", "ambiguity_note"}
                or answer["schema_version"] != 1
                or answer["correct_candidate_id"] not in candidate_ids
                or answer["rule"] != "constant horizontal velocity and constant vertical position through the gap"
                or answer["claim_scope"] != "synthetic generator rule only; not a unique real-world inference"
                or not isinstance(answer["ambiguous_by_visible_context"], bool)):
            raise ValueError("answer key does not match the fixed evaluator contract")
        correct_id = answer["correct_candidate_id"]
        correct_positions[correct_id] = correct_positions.get(correct_id, 0) + 1
        per_method = {}
        for method in METHODS:
            metrics = evaluate_ranking_metrics(
                item["scores"][method], correct_id, tie_tolerance=TIE_TOLERANCE)
            margin = metrics.pop("margin_min_wrong_minus_correct")
            if method == "masked_context_predictor":
                metrics["loss_margin_min_wrong_minus_correct"] = margin
            else:
                metrics["score_margin_min_wrong_minus_correct"] = (
                    None if method == "seeded_random" else margin)
            per_method[method] = {
                **metrics,
                "visible_context_ambiguous": answer["ambiguous_by_visible_context"],
                "rule_answer_is_unique_in_fixture": not answer["ambiguous_by_visible_context"],
                "scores": item["scores"][method],
            }
        output.append({
            "seed": item["seed"],
            "candidate_count": len(candidate_ids),
            "correct_candidate_position": candidate_ids.index(correct_id),
            "visible_context_ambiguous": answer["ambiguous_by_visible_context"],
            "methods": per_method,
        })
    summary = {}
    for method in METHODS:
        values = [row["methods"][method] for row in output]
        summary[method] = {
            "count": len(values),
            "top1_strict": sum(bool(row["top1_strict"]) for row in values) / len(values),
            "top1_expected": sum(row["top1_expected"] for row in values) / len(values),
            "mrr_expected": sum(row["mrr_expected"] for row in values) / len(values),
            "mean_loss_margin_min_wrong_minus_correct": (
                sum(row["loss_margin_min_wrong_minus_correct"] for row in values) / len(values)
                if method == "masked_context_predictor" else None),
            "mean_score_margin_min_wrong_minus_correct": (
                None if method in ("seeded_random", "masked_context_predictor") else
                sum(row["score_margin_min_wrong_minus_correct"] for row in values) / len(values)),
            "ties": sum(row["tie_count"] > 1 for row in values),
        }
    harmonic_mrr = sum(1.0 / rank for rank in range(1, EXPECTED_CANDIDATES + 1)) / EXPECTED_CANDIDATES
    return output, {
        "methods": summary,
        "random_chance": {"top1_expected": 1.0 / EXPECTED_CANDIDATES,
                          "mrr_expected": harmonic_mrr},
        "correct_position_counts": correct_positions,
    }


def _answer_reader(root: Path) -> dict:
    return _read_fixture_json_safe(root, "answer_key.json", max_bytes=MAX_ANSWER_KEY_BYTES)


def _read_only_baseline_scores(root: Path, candidates: list[dict], seed: int) -> dict:
    candidate_ids = [candidate["id"] for candidate in candidates]
    return {
        "seeded_random": baselines.seeded_random_scores(candidate_ids, seed),
        "visible_context_encoder_only_tie": baselines.visible_context_tie_scores(candidate_ids),
        "image_space_constant_velocity_interpolation": baselines.image_space_interpolation_scores(
            root, candidates, read_frame=_read_frame),
    }


def _validate_answer_free_outputs() -> None:
    _require_fixed_output_directories()
    require_absent_outputs((FIXTURE_ROOT, INDEX_PATH, REPORT_JSON, REPORT_MARKDOWN))


def _generate_fixtures(count: int) -> list[tuple[int, Path]]:
    seeds = _validate_count(count)
    _validate_answer_free_outputs()
    FIXTURE_ROOT.mkdir(mode=0o700, parents=False, exist_ok=False)
    sys.path.insert(0, str(SCRIPT_DIR))
    from generate_ball_gap import generate_fixture

    result = []
    for seed in seeds:
        root = FIXTURE_ROOT / f"seed-{seed}"
        generate_fixture(root, seed=seed)
        result.append((seed, root))
    return result


def _processor_from_pinned_config(config_source: bytes):
    if hashlib.sha256(config_source).hexdigest() != PROCESSOR_CONFIG_SHA256:
        raise ValueError("processor config SHA-256 mismatch")
    try:
        config = json.loads(config_source.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("processor config is not valid bounded JSON") from error
    if not isinstance(config, dict):
        raise ValueError("processor config must contain a JSON object")
    import torch
    from transformers import VJEPA2VideoProcessor

    args = {key: value for key, value in config.items()
            if key not in {"_valid_kwargs_names", "model_valid_processing_keys", "video_processor_type"}}
    processor = VJEPA2VideoProcessor(**args)
    return torch, processor, hashlib.sha256(config_source).hexdigest(), len(config_source)


def _load_verified_processor(processor_config: Path):
    """Load only the pinned processor for bounded synthetic checks; never loads a model."""
    config_source = _read_pinned_bytes(
        processor_config, expected_sha256=PROCESSOR_CONFIG_SHA256, max_bytes=128 * 1024)
    return _processor_from_pinned_config(config_source)


def _load_verified_loader_callable(compatibility_loader: Path):
    """Compile only the pinned, bounded loader bytes after checking its imports."""
    loader_source = _read_pinned_bytes(
        compatibility_loader, expected_sha256=COMPATIBILITY_LOADER_SHA256,
        max_bytes=128 * 1024)
    validate_loader_import_closure(loader_source)
    loader_hash = hashlib.sha256(loader_source).hexdigest()
    module_name = f"_jepa_verified_predictor_{loader_hash[:16]}"
    loader_module = types.ModuleType(module_name)
    loader_module.__file__ = str(compatibility_loader)
    loader_module.__package__ = ""
    sys.modules[module_name] = loader_module
    try:
        exec(compile(loader_source, str(compatibility_loader), "exec"), loader_module.__dict__)
        load_official_bundle = getattr(loader_module, "load_official_bundle", None)
    finally:
        sys.modules.pop(module_name, None)
    if not callable(load_official_bundle):
        raise ValueError("verified loader bytes do not define load_official_bundle")
    return load_official_bundle, loader_hash


def _load_processor_and_bundle(checkpoint: Path, identity_record: Path,
                               processor_config: Path, compatibility_loader: Path):
    _install_best_effort_network_guard()
    config_source = _read_pinned_bytes(
        processor_config, expected_sha256=PROCESSOR_CONFIG_SHA256, max_bytes=128 * 1024)
    load_official_bundle, loader_hash = _load_verified_loader_callable(compatibility_loader)
    torch, processor, config_hash, config_bytes = _processor_from_pinned_config(config_source)

    identity = _read_identity_safe(identity_record, checkpoint)
    bundle = load_official_bundle(checkpoint, device=torch.device("cpu"), dtype=torch.float32)
    bundle.encoder.eval()
    bundle.target_encoder.eval()
    bundle.predictor.eval()
    return torch, processor, bundle, identity, config_hash, config_bytes, loader_hash


def _model_scores_for_fixture(root: Path, candidates: list[dict], processor, bundle, torch,
                              progress: PilotProgress) -> dict[str, float]:
    import torch.nn.functional as F

    context_ids, target_ids = token_partition(
        temporal_tokens=32, height_tokens=16, width_tokens=16,
        target_temporal=(15, 16), target_rows=range(4, 12), target_cols=range(4, 12))
    processed_videos = {}
    for candidate in candidates:
        try:
            frames = [_load_ppm_bytes(_read_frame(root, relative))
                      for relative in candidate["frames"]]
            encoded = processor([frames], do_resize=True, do_center_crop=True,
                                input_data_format="channels_last", return_tensors="pt")
        except Exception:
            progress._emit("preprocess_failed", candidate_id=candidate["id"],
                           safe_error_code="processor_error")
            raise
        video = encoded["pixel_values_videos"].to(device="cpu", dtype=torch.float32)
        if tuple(video.shape) != (1, 64, 3, 256, 256):
            progress._emit("preprocess_failed", candidate_id=candidate["id"],
                           safe_error_code="processor_shape_mismatch")
            raise ValueError("processor output shape differs from fixed [1,64,3,256,256]")
        processed_videos[candidate["id"]] = video
        progress._emit("candidate_preprocessed", candidate_id=candidate["id"])
        del frames, encoded

    # Check the actual transformed tensors before the first encoder/predictor call.
    reference_video = processed_videos[candidates[0]["id"]]
    for candidate in candidates[1:]:
        try:
            assert_preprocessed_visible_context_equal(
                reference_video, processed_videos[candidate["id"]], tensor_equal=torch.equal)
        except ValueError:
            progress._emit("preprocess_failed", candidate_id=candidate["id"],
                           safe_error_code="visible_context_mismatch")
            raise
    progress._emit("preprocess_gate_passed")

    context_mask = torch.tensor(context_ids, dtype=torch.long).unsqueeze(0)
    target_mask = torch.tensor(target_ids, dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        context = _tracked_forward(
            progress, "encoder",
            lambda: bundle.encoder(reference_video.permute(0, 2, 1, 3, 4), mask=context_mask))
        if tuple(context.shape) != (1, 8064, 1024):
            raise ValueError("context encoder output must be [1,8064,1024]")
        predicted = _tracked_forward(
            progress, "predictor",
            lambda: _predictor_forward_for_ids(bundle.predictor, context, context_mask, target_mask))
        if tuple(predicted.shape) != (1, 128, 1024):
            raise ValueError("predictor output must be [1,128,1024]")
        predicted = predicted.float()
        index = target_mask.unsqueeze(-1).expand(-1, -1, 1024)
        scores = {}
        for candidate in candidates:
            video = processed_videos[candidate["id"]]
            full = _tracked_forward(
                progress, "target_encoder",
                lambda video=video: bundle.target_encoder(video.permute(0, 2, 1, 3, 4)),
                candidate_id=candidate["id"])
            if tuple(full.shape) != (1, 8192, 1024):
                raise ValueError("target encoder output must be [1,8192,1024]")
            target = full.gather(1, index)
            target = F.layer_norm(target, (target.shape[-1],)).float()
            delta = (predicted - target).abs()
            score = float(delta.mean().item())
            if not math.isfinite(score):
                raise ValueError("model loss is non-finite")
            scores[candidate["id"]] = score
            del full, target, delta
    del processed_videos, reference_video, context, predicted
    return scores


def _summary_markdown(report: dict) -> str:
    lines = [
        "# 合成中央gap pilot",
        "",
        f"- 状態: {report['status']}",
        f"- fixture数: {report['fixture_count']}（4択）",
        f"- device: {report['device']}",
        f"- 経過秒: {report['elapsed_seconds']:.3f}",
        "- 結果の範囲: 合成generator rule下の探索的候補識別のみ。",
        "",
        "| 方法 | 件数 | strict top-1 | tie期待top-1 | tie期待MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, values in report["summary"]["methods"].items():
        lines.append(
            f"| {method} | {values['count']} | {values['top1_strict']:.3f} | "
            f"{values['top1_expected']:.3f} | {values['mrr_expected']:.3f} |")
    lines.extend([
        "",
        f"- ランダム順位の理論基準: top-1 {report['summary']['random_chance']['top1_expected']:.3f}, "
        f"MRR {report['summary']['random_chance']['mrr_expected']:.3f}。",
        "",
        "小標本で統計的な性能・実動画汎化・物理理解・危険予測を示しません。",
        "全候補/全baselineのscore確定後にだけanswer keyを評価しています。",
        "",
    ])
    return "\n".join(lines)


def run_pilot(count: int, checkpoint: Path, identity_record: Path, verified_loss_exp: float,
              processor_config: Path, compatibility_loader: Path,
              progress: PilotProgress | None = None) -> dict:
    validate_fixed_resources(checkpoint, identity_record, verified_loss_exp)
    starts = time.monotonic()
    progress = PilotProgress() if progress is None else progress
    progress.emit_run("run_started")
    fixtures = _generate_fixtures(count)
    progress.emit_run("fixtures_generated")
    records = [_fixture_record(root, seed) for seed, root in fixtures]
    if (sum(record["file_count_excluding_answer"] for record in records) + len(records) > MAX_FILES
            or sum(record["bytes_excluding_answer"] for record in records)
            + len(records) * MAX_ANSWER_KEY_BYTES > MAX_TOTAL_BYTES):
        raise ValueError("selected fixture set exceeds the registered total file or byte bound")
    trajectories = [record["visible_trajectory_template"] for record in records]
    if len({tuple(item[key] for key in ("x_start", "x_velocity", "base_y"))
            for item in trajectories}) != len(trajectories):
        raise ValueError("fixture trajectories are not distinct")

    source_paths = {
        "generator": SCRIPT_DIR / "generate_ball_gap.py",
        "pilot": Path(__file__),
        "smoke": SCRIPT_DIR / "run_masked_prediction_smoke.py",
        "baselines": SCRIPT_DIR / "masked_prediction_baselines.py",
    }
    source_digests = {name: _sha256_file(path)[0] for name, path in source_paths.items()}
    torch, processor, bundle, identity, config_hash, config_bytes, loader_hash = _load_processor_and_bundle(
        checkpoint, identity_record, processor_config, compatibility_loader)
    progress.emit_run("bundle_loaded")
    index = build_index(records, source_sha256=source_digests,
                        processor_config_sha256=config_hash,
                        compatibility_loader_sha256=loader_hash)
    if config_bytes <= 0:
        raise ValueError("processor config must be non-empty")
    index_bytes = (json.dumps(index, sort_keys=True, indent=2) + "\n").encode()
    if len(index_bytes) > 128 * 1024:
        raise ValueError("pilot index exceeds its registered size limit")
    _write_new(INDEX_PATH, index_bytes)
    progress.emit_run("index_written")

    scored = []
    fixture_durations = []
    for fixture_index, (seed, root) in enumerate(fixtures):
        progress.begin_fixture(fixture_index, seed)
        fixture_start = time.monotonic()
        with _first_fixture_deadline(fixture_index == 0):
            try:
                manifest = _read_fixture_json_safe(root, "fixture.json", max_bytes=MAX_MANIFEST_BYTES)
                candidates = _validate_fixture_manifest(root, manifest)
            except Exception:
                progress._emit("fixture_failed", safe_error_code="fixture_input_error")
                raise
            try:
                baseline_scores = _read_only_baseline_scores(root, candidates, seed)
            except Exception:
                progress._emit("fixture_failed", safe_error_code="baseline_error")
                raise
            progress._emit("baseline_scores_completed")
            try:
                baseline_scores["masked_context_predictor"] = _model_scores_for_fixture(
                    root, candidates, processor, bundle, torch, progress)
            except Exception:
                progress._emit("fixture_failed", safe_error_code="unclassified_failure")
                raise
            if set(baseline_scores) != set(METHODS):
                raise ValueError("not every baseline and model candidate was scored")
            candidate_ids = [candidate["id"] for candidate in candidates]
            if any(set(values) != set(candidate_ids) for values in baseline_scores.values()):
                raise ValueError("a scorer omitted or added a candidate")
            ranked_scores = {
                method: dict(rank_candidates(
                    [[0.0]], {candidate_id: [[score]] for candidate_id, score in values.items()}))
                if method != "masked_context_predictor" else values
                for method, values in baseline_scores.items()
            }
            # The model scores are direct masked-target MAE; the helper call above is
            # an identity ranking transform for deterministic candidate validation.
            scored.append({"seed": seed, "root": root, "candidate_ids": candidate_ids,
                           "scores": ranked_scores})
            progress.mark_fixture_scores_completed()
        duration = time.monotonic() - fixture_start
        fixture_durations.append(duration)

    # Do not open any answer key until every candidate from every method is scored.
    progress.emit_run("answer_evaluation_started")
    per_fixture, summary = evaluate_after_scoring(scored, _answer_reader)
    progress.emit_run("answer_evaluation_completed")
    report = {
        "schema_version": 1,
        "status": "completed",
        "fixture_count": len(fixtures),
        "candidate_count_per_fixture": EXPECTED_CANDIDATES,
        "device": DEVICE,
        "dtype": "float32",
        "loss_exp": LOSS_EXP,
        "tie_tolerance": TIE_TOLERANCE,
        "max_first_fixture_seconds": MAX_FIRST_FIXTURE_SECONDS,
        "elapsed_seconds": time.monotonic() - starts,
        "fixture_seconds": fixture_durations,
        "checkpoint_bytes": identity["checkpoint_bytes"],
        "checkpoint_sha256_match": True,
        "processor_config_sha256": config_hash,
        "compatibility_loader_sha256": loader_hash,
        "index_sha256": _sha256_file(INDEX_PATH)[0],
        "progress_counts": progress.summary(),
        "summary": summary,
        "fixtures": per_fixture,
        "answer_read_order": "only after every seed, candidate, and method score completed",
        "claims": "exploratory synthetic generator-rule candidate ranking only",
        "real_media_accessed": False,
    }
    if report["elapsed_seconds"] > 600:
        raise TimeoutError("pilot exceeded the fixed 600-second workflow budget")
    _write_new(REPORT_JSON, (json.dumps(report, sort_keys=True, indent=2) + "\n").encode())
    _write_new(REPORT_MARKDOWN, _summary_markdown(report).encode())
    progress.emit_run("run_completed")
    del bundle
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--identity-record", type=Path)
    parser.add_argument("--processor-config", type=Path)
    parser.add_argument("--compatibility-loader", type=Path)
    parser.add_argument("--verified-loss-exp", type=float, default=LOSS_EXP)
    args = parser.parse_args()
    _validate_count(args.count)
    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "model_weights_read": False,
            "fixture_generation": False,
            "fixture_count": args.count,
            "candidate_count": EXPECTED_CANDIDATES,
            "seed_allowlist": list(FIXED_SEEDS[:args.count]),
            "answer_key_read": False,
            "device": DEVICE,
            "max_wall_seconds": 600,
            "network": "offline plus child-only network deny",
        }, sort_keys=True, indent=2))
        return 0
    if os.environ.get("JEPA_GUARDED_RUN") != "1":
        parser.error("--execute is allowed only from the fixed resource guardian")
    if (args.checkpoint is None or args.identity_record is None
            or args.processor_config is None or args.compatibility_loader is None):
        parser.error("--execute requires explicit checkpoint, identity, processor config, and loader paths")
    print("PILOT_RUNNER_STARTED", flush=True)
    progress = PilotProgress()
    try:
        result = run_pilot(args.count, args.checkpoint, args.identity_record,
                           args.verified_loss_exp, args.processor_config,
                           args.compatibility_loader, progress=progress)
    except KeyboardInterrupt:
        _record_safe_run_failure(progress, interrupted=True)
        return 130
    except Exception:
        _record_safe_run_failure(progress, interrupted=False)
        return 1
    print(json.dumps({"pilot_completed": True, "fixture_count": result["fixture_count"],
                      "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
