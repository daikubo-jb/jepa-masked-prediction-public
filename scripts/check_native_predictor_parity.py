#!/usr/bin/env python3
"""Compare the pinned native V-JEPA 2 predictor with the local masked-ID adapter."""

from __future__ import annotations

import argparse
import ast
import hashlib
from importlib.machinery import ModuleSpec
import json
import os
from pathlib import Path
import socket
import stat
import sys
import types

PROJECT = Path(__file__).resolve().parents[1]
VENDOR = PROJECT / "vendor" / "vjepa2-204698b45b3712590f06245fbfba32d3be539812"
COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
REPORT = PROJECT / "reports" / "native-predictor-parity-20261101.json"
PREPARATION_REPORT_SHA256 = "e320638f069b952129f05c7fe1124a0c87f63238ee44004add7c97f71e36e8bd"
EXPECTED_MANIFEST_SHA256 = "fc2821d30ddc45c3e7f6a9c7bb621419324a44a2baf7b64496828fff5522028f"
SEED = 20261101
ATOL = 1e-5
RTOL = 1e-5
CONTEXT_IDS = (0, 3975, 4248, 8191)
TARGET_IDS = (3976, 4249)
GRID_SIZE = 16
GRID_TOKENS = 32 * GRID_SIZE * GRID_SIZE
MODEL_CONFIG_VERIFIED = True
EXPECTED_CONFIGS = {
    "configs/train/vitl16/pretrain-256px-16f.yaml": {
        "sha256": "45d3528ae0e0b08a2da40a54958eff4ecd86321f58cde26d93d84faf9799b48b",
        "git_blob_sha1": "535736f64af6492584a189d015f133e0d6ed8ed8",
        "size": 1628,
    },
    "configs/train/vitl16/cooldown-256px-64f.yaml": {
        "sha256": "2fd99a75504990c3bd48b510991e5e5d4772210e5fd917ac8f8708a0b71c86f1",
        "git_blob_sha1": "fd7e9fd5cd09ffe97a1cc2b0c664a717da26706d",
        "size": 1731,
    },
}
# The YAML mask list has two mask groups; the checkpoint uses ten learned mask-token slots.
# This separate count is grounded by the existing compatibility loader and its strict-load smoke record.
EXPECTED_MASK_GROUP_COUNT = 2
CHECKPOINT_MASK_TOKEN_COUNT = 10
CHECKPOINT_PREDICTOR_KEY_PREFIX = "module.backbone."
EXPECTED_MODEL_SETTINGS = {
    "model_name": "vit_large", "pred_depth": 12, "pred_embed_dim": 384,
    "pred_num_heads": 12, "use_rope": True, "use_mask_tokens": True,
    "zero_init_mask_tokens": True,
}
EXPECTED_FILES = {
    "src/models/predictor.py": "3d8481bf2544b85ed92b3feb418e3b6d85ca1f81034c931b8c860f2d3d9aac2b",
    "src/masks/utils.py": "833f111a0fa5ffdbd3a6412e2dace2517c3c178f49c14f8bb631d9f6a070dfd0",
    "src/models/utils/modules.py": "b93f6c7e0747deb216419c000c2878f11a9189024a9adeacfd437e172396dff0",
    "src/models/utils/pos_embs.py": "e6d3b3863ade47b3734a3b443f1bcc93eab02e39bf596ea199517f68ccab443e",
    "src/utils/tensors.py": "782b58bd2af456e184750e5318ab773105108383f61b280fe4c7a90f46add2c8",
    "LICENSE": "cf9b17822d1fcd4ff32ccbe14183386fb3adf6f2ff92dc184130823f7fc28173",
}
ALLOWED_LOCAL_MODULES = {
    "src.utils.tensors", "src.masks.utils", "src.models.utils.pos_embs",
    "src.models.utils.modules", "src.models.predictor",
}


def _read_pinned_file(path: Path) -> bytes:
    if path.resolve(strict=True) != path or path.is_symlink():
        raise ValueError("pinned vendor path must not resolve through a symlink")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("safe vendor source open is unavailable")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("pinned vendor source must be a regular file")
        return stream.read()


def verify_vendor_manifest(*, hash_files: bool = True, include_sources: bool = False):
    manifest_path = VENDOR / "SOURCE-MANIFEST.json"
    manifest_bytes = _read_pinned_file(manifest_path)
    manifest = json.loads(manifest_bytes)
    if manifest.get("repository") != "https://github.com/facebookresearch/vjepa2":
        raise ValueError("unexpected upstream repository")
    if manifest.get("commit") != COMMIT or set(manifest.get("files", {})) != set(EXPECTED_FILES):
        raise ValueError("vendor manifest does not match the pinned source set")
    verified_sources = {}
    for relative, expected in EXPECTED_FILES.items():
        record = manifest["files"][relative]
        if record.get("sha256") != expected:
            raise ValueError("vendor manifest hash differs from the reviewed pin")
        if hash_files:
            source = _read_pinned_file(VENDOR / relative)
            digest = hashlib.sha256(source).hexdigest()
            if digest != expected:
                raise ValueError("vendor source bytes differ from the reviewed pin")
            verified_sources[relative] = source
    if hash_files:
        _validate_import_closure(verified_sources)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    return (manifest_digest, verified_sources) if include_sources else manifest_digest


def _validate_import_closure(source_bytes: dict[str, bytes]) -> None:
    for relative, raw in source_bytes.items():
        if not relative.endswith(".py"):
            continue
        tree = ast.parse(raw, filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError("relative vendor imports are outside the pinned import closure")
                if node.module == "src" or (node.module or "").startswith("src."):
                    if node.module not in ALLOWED_LOCAL_MODULES:
                        raise ValueError("vendor source imports an unregistered local src module")
            elif isinstance(node, ast.Import):
                if any(alias.name == "src" or alias.name.startswith("src.") for alias in node.names):
                    raise ValueError("vendor source imports an unregistered local src module")
            elif isinstance(node, ast.Call):
                function = node.func
                dynamic_import = (isinstance(function, ast.Name) and function.id == "__import__")
                dynamic_import = dynamic_import or (
                    isinstance(function, ast.Attribute) and function.attr == "import_module")
                if dynamic_import and node.args and isinstance(node.args[0], ast.Constant):
                    name = node.args[0].value
                    if isinstance(name, str) and (name == "src" or name.startswith("src.")):
                        if name not in ALLOWED_LOCAL_MODULES:
                            raise ValueError("vendor source dynamically imports an unregistered src module")


def validate_mask_ids(context_ids, target_ids, *, total=GRID_TOKENS):
    context = tuple(context_ids)
    target = tuple(target_ids)
    if not context or not target or len(set(context)) != len(context) or len(set(target)) != len(target):
        raise ValueError("context and target IDs must be non-empty and unique")
    if set(context).intersection(target):
        raise ValueError("context and target IDs overlap")
    if any(type(index) is not int or not 0 <= index < total for index in (*context, *target)):
        raise ValueError("token ID is outside the fixed grid")
    if tuple(sorted(context)) != context or tuple(sorted(target)) != target:
        raise ValueError("token IDs must be in canonical ascending order")
    return context, target


def _read_identity(identity_path: Path) -> dict:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("safe identity open is unavailable")
    descriptor = os.open(identity_path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not __import__("stat").S_ISREG(info.st_mode) or info.st_size > 16 * 1024:
            raise ValueError("identity record must be a bounded regular non-symlink file")
        identity = json.loads(stream.read().decode("utf-8"))
    if not isinstance(identity, dict):
        raise ValueError("identity record must contain a JSON object")
    return identity


def open_verified_checkpoint(identity_path: Path, checkpoint: Path):
    identity = _read_identity(identity_path)
    required = {"checkpoint_path", "checkpoint_bytes", "sha256"}
    if not isinstance(identity, dict) or not required.issubset(identity):
        raise ValueError("identity record lacks required fields")
    recorded = Path(identity["checkpoint_path"]).expanduser().resolve(strict=False)
    supplied = Path(checkpoint).expanduser().resolve(strict=False)
    if recorded != supplied:
        raise ValueError("checkpoint path differs from the supplied local identity")
    if type(identity["checkpoint_bytes"]) is not int:
        raise ValueError("identity checkpoint size is malformed")
    if not isinstance(identity["sha256"], str) or len(identity["sha256"]) != 64:
        raise ValueError("identity SHA-256 field is malformed")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("safe checkpoint open is unavailable")
    descriptor = os.open(checkpoint, os.O_RDONLY | os.O_NOFOLLOW)
    stream = os.fdopen(descriptor, "rb")
    info = os.fstat(stream.fileno())
    if not __import__("stat").S_ISREG(info.st_mode) or info.st_size != identity["checkpoint_bytes"]:
        stream.close()
        raise ValueError("checkpoint is not a regular file of the recorded size")
    digest = hashlib.sha256()
    try:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != identity["sha256"]:
            raise ValueError("checkpoint SHA-256 differs from the local identity")
        stream.seek(0)
        return identity, stream
    except BaseException:
        stream.close()
        raise


def validate_identity_paths(identity_path: Path, checkpoint: Path) -> dict:
    """Model-free path-only gate. Does not open or inspect the checkpoint."""
    identity = _read_identity(identity_path)
    required = {"checkpoint_path", "checkpoint_bytes", "sha256"}
    if not isinstance(identity, dict) or not required.issubset(identity):
        raise ValueError("identity record lacks required fields")
    recorded = Path(identity["checkpoint_path"]).expanduser().resolve(strict=False)
    supplied = Path(checkpoint).expanduser().resolve(strict=False)
    if recorded != supplied:
        raise ValueError("checkpoint path differs from the supplied local identity")
    return identity


def _offline_guard() -> None:
    def denied(*_args, **_kwargs):
        raise RuntimeError("network access is disabled for native predictor parity")

    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })


def _load_native_factory(verified_sources):
    if any(name == "src" or name.startswith("src.") for name in sys.modules):
        raise RuntimeError("a conflicting 'src' module is already loaded")
    package_paths = {
        "src": VENDOR / "src",
        "src.models": VENDOR / "src" / "models",
        "src.models.utils": VENDOR / "src" / "models" / "utils",
        "src.masks": VENDOR / "src" / "masks",
        "src.utils": VENDOR / "src" / "utils",
    }
    for name, path in package_paths.items():
        resolved = path.resolve(strict=True)
        if resolved != path:
            raise RuntimeError("native namespace path resolves outside the fixed vendor tree")
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = package
    try:
        modules = (
            ("src.utils.tensors", "src/utils/tensors.py"),
            ("src.masks.utils", "src/masks/utils.py"),
            ("src.models.utils.pos_embs", "src/models/utils/pos_embs.py"),
            ("src.models.utils.modules", "src/models/utils/modules.py"),
            ("src.models.predictor", "src/models/predictor.py"),
        )
        for name, relative in modules:
            path = VENDOR / relative
            resolved = path.resolve(strict=True)
            if resolved != path or relative not in verified_sources:
                raise RuntimeError("native source was not pinned before execution")
            module = types.ModuleType(name)
            module.__file__ = str(path)
            module.__package__ = name.rpartition(".")[0]
            module.__spec__ = None
            sys.modules[name] = module
            code = compile(verified_sources[relative], str(path), "exec")
            exec(code, module.__dict__)
        return sys.modules["src.models.predictor"].vit_predictor
    except ImportError as error:
        raise RuntimeError("native predictor dependencies are unavailable; no fallback is permitted") from error
    except BaseException:
        for name in tuple(sys.modules):
            if name == "src" or name.startswith("src."):
                del sys.modules[name]
        raise


def require_verified_model_config() -> dict:
    """Fail closed unless both pinned official leaf configs and predictor settings match."""
    if not MODEL_CONFIG_VERIFIED:
        raise RuntimeError("predictor configuration is not approved")
    manifest_path = VENDOR / "SOURCE-MANIFEST.json"
    manifest_bytes = _read_pinned_file(manifest_path)
    if hashlib.sha256(manifest_bytes).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError("vendor manifest bytes differ from the reviewed config pin")
    manifest = json.loads(manifest_bytes)
    if manifest.get("repository") != "https://github.com/facebookresearch/vjepa2":
        raise ValueError("unexpected upstream repository for predictor config")
    if manifest.get("commit") != COMMIT:
        raise ValueError("predictor config manifest commit differs from the pinned source")
    records = manifest.get("configuration_files")
    if not isinstance(records, dict) or set(records) != set(EXPECTED_CONFIGS):
        raise ValueError("official predictor config set differs from the reviewed pin")
    import yaml

    verified = {}
    for relative, expected in EXPECTED_CONFIGS.items():
        record = records[relative]
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("config manifest metadata differs from the reviewed pin")
        raw = _read_pinned_file(VENDOR / relative)
        if len(raw) != expected["size"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError("official predictor config bytes differ from the reviewed pin")
        document = yaml.safe_load(raw)
        model = document.get("model") if isinstance(document, dict) else None
        if not isinstance(model, dict):
            raise ValueError("official predictor config has no model mapping")
        settings = {key: model.get(key) for key in EXPECTED_MODEL_SETTINGS}
        if settings != EXPECTED_MODEL_SETTINGS:
            raise ValueError("official predictor model settings differ from the reviewed constructor")
        masks = document.get("mask")
        if not isinstance(masks, list) or len(masks) != EXPECTED_MASK_GROUP_COUNT:
            raise ValueError("official config mask-group count differs from the reviewed pin")
        verified[relative] = {**settings, "num_mask_tokens": CHECKPOINT_MASK_TOKEN_COUNT}
    if len(set(tuple(sorted(item.items())) for item in verified.values())) != 1:
        raise ValueError("official predictor leaf configs disagree on model settings")
    return next(iter(verified.values()))


def normalize_predictor_state_dict(items, *, expected_keys):
    """Strip only the documented wrapper prefix, proving a one-to-one full key map."""
    pairs = list(items)
    if not pairs:
        raise ValueError("checkpoint predictor state dictionary is empty")
    source_keys = [key for key, _value in pairs]
    if any(type(key) is not str or not key for key in source_keys):
        raise ValueError("checkpoint predictor state keys must be non-empty strings")
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("checkpoint predictor state contains duplicate/colliding source keys")
    prefix = CHECKPOINT_PREDICTOR_KEY_PREFIX
    if not all(key.startswith(prefix) for key in source_keys):
        raise ValueError("checkpoint predictor keys must all use the documented module.backbone prefix")
    normalized = {}
    for key, value in pairs:
        target_key = key[len(prefix):]
        if not target_key:
            raise ValueError("checkpoint predictor prefix has an empty destination key")
        if target_key in normalized:
            raise ValueError("checkpoint predictor prefix stripping caused a key collision")
        normalized[target_key] = value
    expected = set(expected_keys)
    actual = set(normalized)
    if actual != expected:
        raise ValueError(
            "checkpoint predictor key mapping is not complete "
            f"({len(expected - actual)} missing, {len(actual - expected)} unexpected)"
        )
    return normalized


def _load_predictor(checkpoint_stream, factory, model_settings):
    import torch

    # The existing compatibility loader strips exactly module.backbone. from every
    # checkpoint branch, then strict-loads it. Its prior successful strict smoke
    # also establishes ten mask-token slots; YAML mask-group count is separate.
    # This native path performs the same prefix-only mapping, verifies every key,
    # and still requires strict=True so shapes and values are never silently dropped.
    predictor = factory(
        img_size=(256, 256), patch_size=16, num_frames=64, tubelet_size=2,
        embed_dim=1024, predictor_embed_dim=model_settings["pred_embed_dim"], out_embed_dim=1024,
        depth=model_settings["pred_depth"], num_heads=model_settings["pred_num_heads"],
        use_rope=model_settings["use_rope"],
        use_mask_tokens=model_settings["use_mask_tokens"],
        num_mask_tokens=model_settings["num_mask_tokens"],
        zero_init_mask_tokens=model_settings["zero_init_mask_tokens"],
    ).to(device="cpu", dtype=torch.float32)
    payload = torch.load(checkpoint_stream, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("predictor"), dict):
        raise ValueError("checkpoint has no plain predictor state dictionary")
    mapped_state = normalize_predictor_state_dict(
        payload["predictor"].items(), expected_keys=predictor.state_dict().keys())
    predictor.load_state_dict(mapped_state, strict=True)
    predictor.eval()
    return predictor


def _capture_native_blocks(predictor, context, context_ids, target_ids):
    trace = []
    handles = []

    def capture(_module, args, kwargs):
        mask = kwargs.get("mask")
        if mask is None:
            raise ValueError("native block did not receive explicit token IDs")
        trace.append({"input_shape": list(args[0].shape),
                      "mask_ids": mask.detach().cpu().tolist()})

    try:
        for block in predictor.predictor_blocks:
            handles.append(block.register_forward_pre_hook(capture, with_kwargs=True))
        output = predictor(
            context,
            [context_ids],
            [target_ids],
            mask_index=1,
        )
    finally:
        for handle in handles:
            handle.remove()
    return output, trace


def _safe_report(manifest_sha256: str, identity_sha256: str, native, adapter,
                 native_trace, adapter_trace) -> dict:
    import torch

    if tuple(native.shape) != (1, len(TARGET_IDS), 1024) or tuple(adapter.shape) != tuple(native.shape):
        raise ValueError("predictor outputs do not match the fixed target shape")
    if native_trace != adapter_trace:
        raise ValueError("native and adapter block token ordering or shapes differ")
    if not bool(torch.isfinite(native).all()) or not bool(torch.isfinite(adapter).all()):
        raise ValueError("predictor output contains non-finite values")
    delta = (native - adapter).abs()
    if not torch.allclose(native, adapter, atol=ATOL, rtol=RTOL):
        raise ValueError("native and adapter outputs exceed the predeclared tolerance")
    return {
        "schema_version": 1,
        "status": "parity_pass",
        "upstream_commit": COMMIT,
        "vendor_manifest_sha256": manifest_sha256,
        "checkpoint_identity_sha256": identity_sha256,
        "checkpoint_provider_authenticity_verified": False,
        "seed": SEED,
        "device": "cpu",
        "dtype": "float32",
        "context_token_ids": list(CONTEXT_IDS),
        "target_token_ids": list(TARGET_IDS),
        "native_block_trace": native_trace,
        "adapter_block_trace": adapter_trace,
        "output_shape": list(native.shape),
        "atol": ATOL,
        "rtol": RTOL,
        "max_absolute_error": float(delta.max().item()),
        "mean_absolute_error": float(delta.mean().item()),
        "real_media_accessed": False,
    }


def _write_report_safely(report_path: Path, result: dict) -> None:
    if report_path.parent != PROJECT / "reports" or report_path.parent.is_symlink():
        raise RuntimeError("report directory is outside the fixed output or is a symlink")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("safe report output is unavailable")
    directory_fd = os.open(report_path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            descriptor = os.open(report_path.name, os.O_RDWR | os.O_NOFOLLOW, dir_fd=directory_fd)
        except FileNotFoundError:
            descriptor = os.open(
                report_path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            mode = "wb"
        else:
            mode = "r+b"
        with os.fdopen(descriptor, mode) as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 256 * 1024:
                raise RuntimeError("existing report is not a private regular file")
            if mode == "r+b":
                previous = stream.read()
                if hashlib.sha256(previous).hexdigest() != PREPARATION_REPORT_SHA256:
                    raise RuntimeError("existing report differs from the reviewed preparation report")
                stream.seek(0)
                stream.truncate(0)
            stream.write(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(directory_fd)


def validate_report_output(report_path: Path) -> None:
    """Preflight the sole report target before any model or checkpoint access."""
    if report_path != REPORT or report_path.parent != PROJECT / "reports" or report_path.parent.is_symlink():
        raise RuntimeError("report directory is outside the fixed output or is a symlink")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("safe report output is unavailable")
    directory_fd = os.open(report_path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            descriptor = os.open(report_path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        except FileNotFoundError:
            return
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 256 * 1024:
                raise RuntimeError("existing report is not a private regular file")
            digest = hashlib.sha256(stream.read()).hexdigest()
            if digest != PREPARATION_REPORT_SHA256:
                raise RuntimeError("existing report differs from the reviewed preparation report")
    finally:
        os.close(directory_fd)


def run(checkpoint: Path, identity_record: Path, report: Path) -> dict:
    model_settings = require_verified_model_config()
    if os.environ.get("JEPA_GUARDED_RUN") != "1":
        raise RuntimeError("parity must be launched by the fixed resource guardian")
    if not Path(checkpoint).is_absolute() or not Path(identity_record).is_absolute():
        raise ValueError("checkpoint and identity paths must be explicit absolute paths")
    report_path = Path(report)
    if not report_path.is_absolute():
        report_path = PROJECT / report_path
    if report_path.resolve(strict=False) != REPORT.resolve(strict=False):
        raise ValueError("report path is outside the fixed project output")
    if report_path.is_symlink():
        raise ValueError("report output may not be a symlink")
    validate_report_output(report_path)
    _offline_guard()
    manifest_sha256, verified_sources = verify_vendor_manifest(include_sources=True)
    import torch
    factory = _load_native_factory(verified_sources)
    identity, checkpoint_stream = open_verified_checkpoint(identity_record, checkpoint)
    with checkpoint_stream:
        predictor = _load_predictor(checkpoint_stream, factory, model_settings)
    validate_mask_ids(CONTEXT_IDS, TARGET_IDS)
    torch.manual_seed(SEED)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    context = torch.randn((1, len(CONTEXT_IDS), 1024), generator=generator,
                          dtype=torch.float32, device="cpu")
    context_ids = torch.tensor([CONTEXT_IDS], dtype=torch.long)
    target_ids = torch.tensor([TARGET_IDS], dtype=torch.long)
    from run_masked_prediction_smoke import _predictor_forward_for_ids

    with torch.inference_mode():
        native, native_trace = _capture_native_blocks(predictor, context, context_ids, target_ids)
        adapter, adapter_trace = _predictor_forward_for_ids(
            predictor, context, context_ids, target_ids, return_trace=True)
        adapter_trace = [
            {"input_shape": shape, "mask_ids": adapter_trace["sorted_token_ids"]}
            for shape in adapter_trace["block_input_shapes"]
        ]
        result = _safe_report(
            manifest_sha256, identity["sha256"], native, adapter,
            native_trace, adapter_trace,
        )
    _write_report_safely(report_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--identity-record", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.checkpoint, args.identity_record, args.report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
