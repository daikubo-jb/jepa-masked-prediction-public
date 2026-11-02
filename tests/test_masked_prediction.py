from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_ball_gap import generate_fixture
from masked_prediction import (evaluate_ranking, rank_candidates,
                               require_supported_loss_contract,
                               token_ids_for_frame_range, token_partition)
from run_masked_prediction_smoke import (_fixture_file, _predictor_forward_for_ids,
                                        _validate_candidate_frames, _validate_fixture_manifest)


class TokenMaskTests(unittest.TestCase):
    def test_target_ids_are_removed_from_encoder_context(self):
        context, target = token_partition(
            temporal_tokens=32,
            height_tokens=16,
            width_tokens=16,
            target_temporal=(15, 16),
            target_rows=range(4, 12),
            target_cols=range(4, 12),
        )
        total = 32 * 16 * 16
        self.assertEqual(len(target), 128)
        self.assertEqual(len(context), total - 128)
        self.assertFalse(set(context).intersection(target))
        self.assertEqual(set(context).union(target), set(range(total)))
        self.assertNotIn(15 * 256 + 4 * 16 + 4, context)

    def test_rejects_out_of_bounds_or_empty_mask(self):
        for values in (
            dict(temporal_tokens=2, height_tokens=2, width_tokens=2,
                 target_temporal=(2,), target_rows=(0,), target_cols=(0,)),
            dict(temporal_tokens=2, height_tokens=2, width_tokens=2,
                 target_temporal=(), target_rows=(0,), target_cols=(0,)),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                token_partition(**values)

    def test_predictor_adapter_extracts_explicit_interleaved_target_ids(self):
        try:
            import torch
        except ImportError as error:
            self.skipTest(f"local torch unavailable: {error}")

        class IdentityBlock:
            def __call__(self, x, *, mask, height, width):
                self.seen_ids = mask.clone()
                return x

        class ToyPredictor:
            predictor_embed = torch.nn.Identity()
            mask_tokens = [torch.nn.Parameter(torch.ones(1, 1, 2) * 99),
                           torch.nn.Parameter(torch.zeros(1, 1, 2))]
            predictor_blocks = [IdentityBlock()]
            predictor_norm = torch.nn.Identity()
            predictor_proj = torch.nn.Identity()

        predictor = ToyPredictor()
        context = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        context_ids = torch.tensor([[0, 2]])
        target_ids = torch.tensor([[1, 4]])
        output = _predictor_forward_for_ids(predictor, context, context_ids, target_ids)
        self.assertEqual(tuple(output.shape), (1, 2, 2))
        torch.testing.assert_close(output, torch.zeros_like(output))
        self.assertEqual(predictor.predictor_blocks[0].seen_ids.tolist(), [[0, 1, 2, 4]])

    def test_frame_mask_requires_tubelet_alignment(self):
        ids = token_ids_for_frame_range(30, 33, tubelet_size=2,
                                        height_tokens=16, width_tokens=16)
        self.assertEqual(len(ids), 2 * 16 * 16)
        self.assertEqual(sorted({token // 256 for token in ids}), [15, 16])
        with self.assertRaises(ValueError):
            token_ids_for_frame_range(29, 33, tubelet_size=2,
                                      height_tokens=16, width_tokens=16)


class CandidateRankingTests(unittest.TestCase):
    def test_loss_contract_fails_closed_when_unverified_or_incompatible(self):
        require_supported_loss_contract(1.0, "layer_norm_last_dim")
        for exponent, norm in ((2.0, "layer_norm_last_dim"), (1.0, "none")):
            with self.subTest(exponent=exponent, norm=norm), self.assertRaises(ValueError):
                require_supported_loss_contract(exponent, norm)

    def test_lowest_masked_target_error_ranks_first(self):
        prediction = [[0.0, 1.0], [2.0, 3.0]]
        candidates = {
            "wrong-a": [[5.0, 5.0], [5.0, 5.0]],
            "answer": [[0.1, 1.1], [1.9, 2.9]],
            "wrong-b": [[-4.0, -2.0], [1.0, 2.0]],
        }
        ranking = rank_candidates(prediction, candidates)
        self.assertEqual(ranking[0][0], "answer")
        self.assertLess(ranking[0][1], ranking[1][1])

    def test_ties_are_reported_not_broken_by_candidate_name(self):
        ranking = rank_candidates([[1.0]], {"z": [[0.0]], "a": [[2.0]]})
        self.assertEqual(ranking, [("z", 1.0), ("a", 1.0)])
        result = evaluate_ranking(ranking, "a", visible_context_ambiguous=True)
        self.assertEqual(result["tied_best_candidate_ids"], ["z", "a"])
        self.assertTrue(result["best_is_tied"])
        self.assertFalse(result["top1_success"])
        self.assertTrue(result["visible_context_ambiguous"])

    def test_rejects_mismatched_or_nonfinite_features(self):
        with self.assertRaises(ValueError):
            rank_candidates([[0.0]], {"bad": [[0.0], [1.0]]})
        with self.assertRaises(ValueError):
            rank_candidates([[float("nan")]], {"bad": [[0.0]]})


class FixtureTests(unittest.TestCase):
    def _sha_tree(self, root: Path) -> list[tuple[str, str]]:
        return sorted(
            (path.relative_to(root).as_posix(),
             hashlib.sha256(path.read_bytes()).hexdigest())
            for path in root.rglob("*") if path.is_file()
        )

    def test_generator_is_deterministic_and_preserves_visible_context(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            generate_fixture(first, seed=123)
            generate_fixture(second, seed=123)
            self.assertEqual(self._sha_tree(first), self._sha_tree(second))

            manifest = json.loads((first / "fixture.json").read_text())
            answer = json.loads((first / "answer_key.json").read_text())
            self.assertNotIn("correct_candidate_id", manifest)
            self.assertEqual(answer["correct_candidate_id"], "candidate-3")
            self.assertEqual(len(manifest["candidates"]), 4)
            self.assertEqual(manifest["candidate_order_seed"], 123)
            other = Path(temp) / "other-seed"
            second_manifest = generate_fixture(other, seed=77)
            second_answer = json.loads((other / "answer_key.json").read_text())
            self.assertEqual(second_answer["correct_candidate_id"], "candidate-1")
            candidates = manifest["candidates"]
            for relative in candidates[0]["frames"]:
                frame_index = int(Path(relative).stem)
                if frame_index not in range(30, 34):
                    reference = (first / relative).read_bytes()
                    for candidate in candidates[1:]:
                        self.assertEqual(reference, (first / candidate["frames"][frame_index]).read_bytes())

    def test_candidate_differences_stay_inside_hidden_time_and_center_box(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fixture"
            generate_fixture(root)
            manifest = json.loads((root / "fixture.json").read_text())
            first = manifest["candidates"][0]
            other = manifest["candidates"][1]
            for frame_index in range(64):
                left = (root / first["frames"][frame_index]).read_bytes()
                right = (root / other["frames"][frame_index]).read_bytes()
                if frame_index not in range(30, 34):
                    self.assertEqual(left, right)
                    continue
                header = len(b"P6\n64 64\n255\n")
                a, b = left[header:], right[header:]
                for y in range(64):
                    for x in range(64):
                        if a[(y * 64 + x) * 3:(y * 64 + x + 1) * 3] != b[(y * 64 + x) * 3:(y * 64 + x + 1) * 3]:
                            self.assertTrue(16 <= x < 48 and 16 <= y < 48)

    def test_fixture_paths_reject_absolute_parent_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "fixture"
            manifest = generate_fixture(root, seed=987)
            candidates = _validate_fixture_manifest(root, manifest)
            _validate_candidate_frames(root, candidates)
            legacy_manifest = dict(manifest)
            legacy_manifest["schema_version"] = 2
            legacy_manifest.pop("visible_trajectory_template")
            self.assertEqual(len(_validate_fixture_manifest(root, legacy_manifest)), 4)
            leaked_manifest = dict(manifest)
            leaked_manifest["correct_candidate_id"] = "candidate-0"
            with self.assertRaises(ValueError):
                _validate_fixture_manifest(root, leaked_manifest)
            outside = base / "outside.ppm"
            outside.write_bytes(b"synthetic")
            (root / "inside.ppm").write_bytes(b"synthetic")
            try:
                (root / "link.ppm").symlink_to(outside)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"temporary symlinks unavailable: {error}")
            self.assertEqual(_fixture_file(root, "inside.ppm", max_bytes=64), (root / "inside.ppm").resolve())
            for relative in (str(outside), "../outside.ppm", "link.ppm"):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    _fixture_file(root, relative, max_bytes=64)
            with self.assertRaises(ValueError):
                _fixture_file(root, "inside.ppm", max_bytes=4)

    def test_seed_changes_trajectory_not_only_candidate_order(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "seed-a"
            second = Path(temp) / "seed-b"
            first_manifest = generate_fixture(first, seed=1001)
            second_manifest = generate_fixture(second, seed=1002)
            self.assertNotEqual(first_manifest["visible_trajectory_template"],
                                second_manifest["visible_trajectory_template"])
            first_frame = first / first_manifest["candidates"][0]["frames"][0]
            second_frame = second / second_manifest["candidates"][0]["frames"][0]
            self.assertNotEqual(first_frame.read_bytes(), second_frame.read_bytes())

    def test_actual_video_processor_preserves_visible_frames_and_patch_mask(self):
        try:
            import numpy as np
            from transformers import VJEPA2VideoProcessor
        except ImportError as error:
            self.skipTest(f"local V-JEPA processor dependencies unavailable: {error}")
        configured = os.environ.get("JEPA_PROCESSOR_CONFIG")
        if not configured:
            self.skipTest("JEPA_PROCESSOR_CONFIG is not configured; no local config discovery or network retrieval")
        config = Path(configured)
        if not config.is_file():
            self.skipTest("local processor config unavailable; no network/config retrieval allowed")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fixture"
            manifest = generate_fixture(root, seed=77)
            processor_config = json.loads(config.read_text())
            processor_args = {key: value for key, value in processor_config.items()
                              if key not in {"_valid_kwargs_names", "model_valid_processing_keys", "video_processor_type"}}
            processor = VJEPA2VideoProcessor(**processor_args)
            outputs = []
            for candidate in manifest["candidates"]:
                frames = []
                for relative in candidate["frames"]:
                    data = (root / relative).read_bytes()
                    header, payload = data.split(b"\n255\n", 1)
                    self.assertEqual(header, b"P6\n64 64")
                    frames.append(np.frombuffer(payload, dtype=np.uint8).reshape(64, 64, 3).copy())
                encoded = processor([frames], do_resize=True, do_center_crop=True,
                                    input_data_format="channels_last", return_tensors="pt")
                video = encoded["pixel_values_videos"]
                self.assertEqual(tuple(video.shape), (1, 64, 3, 256, 256))
                outputs.append(video[0].cpu().numpy())
            reference = outputs[0]
            for video in outputs[1:]:
                for frame_index in list(range(0, 30)) + list(range(34, 64)):
                    np.testing.assert_array_equal(reference[frame_index], video[frame_index])
                for frame_index in range(30, 34):
                    different = np.any(reference[frame_index] != video[frame_index], axis=0)
                    ys, xs = np.where(different)
                    self.assertGreater(len(xs), 0)
                    self.assertTrue(np.all((ys // 16 >= 4) & (ys // 16 <= 11)))
                    self.assertTrue(np.all((xs // 16 >= 4) & (xs // 16 <= 11)))


if __name__ == "__main__":
    unittest.main()
