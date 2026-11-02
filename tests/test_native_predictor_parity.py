from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_native_predictor_parity as parity


class NativeParityContractTests(unittest.TestCase):
    def test_vendor_manifest_matches_pin(self):
        try:
            import yaml  # noqa: F401
        except ImportError as error:
            self.skipTest(f"PyYAML test dependency unavailable; manifest parse not exercised: {error}")
        digest = parity.verify_vendor_manifest(hash_files=True)
        self.assertEqual(digest, parity.EXPECTED_MANIFEST_SHA256)
        self.assertEqual(parity.COMMIT, "204698b45b3712590f06245fbfba32d3be539812")
        self.assertEqual(parity.require_verified_model_config(),
                         {**parity.EXPECTED_MODEL_SETTINGS, "num_mask_tokens": 10})

    def test_checkpoint_path_identity_mismatch_refused_without_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity_path = root / "identity.json"
            checkpoint = root / "checkpoint.pt"
            identity_path.write_text(json.dumps({
                "checkpoint_path": str(root / "different.pt"),
                "checkpoint_bytes": 1,
                "sha256": "0" * 64,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "path differs"):
                parity.validate_identity_paths(identity_path, checkpoint)
            self.assertFalse(checkpoint.exists())

    def test_explicit_checkpoint_path_is_portable_but_still_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "chosen-model.pt"
            identity_path = root / "identity.json"
            payload = b"small test payload"
            checkpoint.write_bytes(payload)
            identity_path.write_text(json.dumps({
                "checkpoint_path": str(checkpoint),
                "checkpoint_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }), encoding="utf-8")
            self.assertEqual(parity.validate_identity_paths(identity_path, checkpoint)["checkpoint_bytes"],
                             len(payload))
            identity, stream = parity.open_verified_checkpoint(identity_path, checkpoint)
            with stream:
                self.assertEqual(stream.read(), payload)
            checkpoint.write_bytes(b"other test payload")
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                parity.open_verified_checkpoint(identity_path, checkpoint)

    def test_input_mask_partition_contract(self):
        context, target = parity.validate_mask_ids(parity.CONTEXT_IDS, parity.TARGET_IDS)
        self.assertEqual(set(context).intersection(target), set())
        self.assertEqual(len(context) + len(target), 6)
        self.assertEqual([(index // 256, (index % 256) // 16, index % 16)
                          for index in target], [(15, 8, 8), (16, 9, 9)])
        for bad_context, bad_target in (
            ((0, 1), (1,)),
            ((0, 0), (1,)),
            ((-1,), (1,)),
        ):
            with self.subTest(context=bad_context, target=bad_target), self.assertRaises(ValueError):
                parity.validate_mask_ids(bad_context, bad_target)

        try:
            import torch
        except ImportError as error:
            self.skipTest(f"existing torch unavailable: {error}")
        from run_masked_prediction_smoke import _predictor_forward_for_ids

        class GridBlock:
            def forward(self, x, mask=None, T=None, H_patches=None, W_patches=None):
                self.mask = mask.clone()
                self.grid = (T, H_patches, W_patches)
                return x

            __call__ = forward

        block = GridBlock()
        class ToyPredictor:
            predictor_embed = torch.nn.Identity()
            mask_tokens = [torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)]
            predictor_blocks = [block]
            predictor_norm = torch.nn.Identity()
            predictor_proj = torch.nn.Identity()

        output, trace = _predictor_forward_for_ids(
            ToyPredictor(), torch.ones(1, 2, 2), torch.tensor([[0, 2]]),
            torch.tensor([[1, 4]]), return_trace=True,
        )
        self.assertEqual(tuple(output.shape), (1, 2, 2))
        self.assertEqual(block.grid, (32, 16, 16))
        self.assertEqual(block.mask.tolist(), [[0, 1, 2, 4]])
        self.assertEqual(trace["sorted_token_ids"], [[0, 1, 2, 4]])

    def test_predictor_only_weights_only_strict_load_contract(self):
        source = (ROOT / "scripts" / "check_native_predictor_parity.py").read_text(encoding="utf-8")
        self.assertIn("weights_only=True", source)
        self.assertIn("weights_only=True", source)
        self.assertIn("normalize_predictor_state_dict", source)
        self.assertIn("predictor.load_state_dict(mapped_state, strict=True)", source)
        self.assertNotIn("load_official_bundle", source)
        self.assertNotIn("target_encoder", source)
        with self.assertRaisesRegex(ValueError, "manifest bytes differ"):
            with mock.patch.object(parity, "EXPECTED_MANIFEST_SHA256", "0" * 64):
                parity.require_verified_model_config()
        weight = object()
        mapped = parity.normalize_predictor_state_dict(
            [("module.backbone.weight", weight), ("module.backbone.bias", object())],
            expected_keys={"weight", "bias"},
        )
        self.assertIs(mapped["weight"], weight)
        with self.assertRaisesRegex(ValueError, "must all use"):
            parity.normalize_predictor_state_dict(
                [("module.backbone.weight", weight), ("bias", object())],
                expected_keys={"weight", "bias"},
            )
        with self.assertRaisesRegex(ValueError, "duplicate/colliding source"):
            parity.normalize_predictor_state_dict(
                [("module.backbone.weight", weight), ("module.backbone.weight", object())],
                expected_keys={"weight"},
            )
        with self.assertRaisesRegex(ValueError, "not complete"):
            parity.normalize_predictor_state_dict(
                [("module.backbone.weight", weight)],
                expected_keys={"weight", "bias"},
            )

    def test_native_imports_resolve_only_vendor_tree(self):
        parity_source = (ROOT / "scripts" / "check_native_predictor_parity.py").read_text(encoding="utf-8")
        self.assertIn("compile(verified_sources[relative]", parity_source)
        self.assertNotIn("spec_from_file_location", parity_source)
        expected = set(parity.EXPECTED_FILES)
        for relative in expected:
            if not relative.endswith(".py"):
                continue
            source = (parity.VENDOR / relative).read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
                    module_path = node.module.removeprefix("src.").replace(".", "/")
                    possible = {f"src/{module_path}.py", f"src/{module_path}/__init__.py"}
                    self.assertTrue(possible.intersection(expected), node.module)
        smoke_source = (ROOT / "scripts" / "run_masked_prediction_smoke.py").read_text(encoding="utf-8")
        self.assertNotIn("RISK_ROOT", smoke_source)
        self.assertIn("_load_verified_loader_callable(compatibility_loader)", smoke_source)
        self.assertIn("--compatibility-loader", smoke_source)

    def test_report_excludes_tensors_secrets_and_raw_paths(self):
        if parity.REPORT.exists():
            # A completed run is evidence, not the preparation placeholder. The
            # writer must refuse to replace it, and the refusal must preserve bytes.
            before = parity.REPORT.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "differs from the reviewed preparation report"):
                parity.validate_report_output(parity.REPORT)
            self.assertEqual(hashlib.sha256(parity.REPORT.read_bytes()).digest(),
                             hashlib.sha256(before).digest())
        else:
            parity.validate_report_output(parity.REPORT)
        source = (ROOT / "scripts" / "check_native_predictor_parity.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        safe_report = next(node for node in tree.body
                           if isinstance(node, ast.FunctionDef) and node.name == "_safe_report")
        returned = next(node for node in ast.walk(safe_report) if isinstance(node, ast.Return))
        self.assertIsInstance(returned.value, ast.Dict)
        keys = {key.value for key in returned.value.keys if isinstance(key, ast.Constant)}
        self.assertNotIn("tensor", keys)
        self.assertNotIn("raw_path", keys)
        self.assertNotIn("secret", keys)
        self.assertNotIn("checkpoint_path", keys)
        self.assertIn("checkpoint_identity_sha256", keys)
        self.assertTrue(all(isinstance(value, str) for value in keys))
        self.assertIn("PREPARATION_REPORT_SHA256", source)


if __name__ == "__main__":
    unittest.main()
