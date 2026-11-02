from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import masked_prediction_baselines as baselines
from masked_prediction import evaluate_ranking_metrics
from run_masked_prediction_pilot import (
    COMPATIBILITY_LOADER_SHA256, FIXED_SEEDS, FIXTURE_ROOT, METHODS,
    PROCESSOR_CONFIG_SHA256, REPORT_JSON, REPORT_MARKDOWN, PilotProgress,
    assert_preprocessed_visible_context_equal, build_index, evaluate_after_scoring,
    main, require_absent_outputs, validate_fixed_resources, validate_loader_import_closure,
    _first_fixture_deadline, _load_verified_processor, _read_pinned_bytes,
    _load_verified_loader_callable, _tracked_forward, _validate_count,
)
from run_masked_prediction_smoke import _fixture_file, _load_ppm_bytes, _validate_frame_pair


PPM_HEADER = b"P6\n64 64\n255\n"


def _fixture_frame(cx: float, cy: float) -> bytes:
    template = {(x, y) for y in range(-4, 5) for x in range(-4, 5)
                if x * x + y * y <= 16}
    return PPM_HEADER + baselines._predicted_frame(
        cx, cy, bytes((236, 238, 240)), bytes((45, 105, 220)), template)


class PilotContractTests(unittest.TestCase):
    def test_pilot_import_is_model_free(self):
        code = (
            "import sys; sys.path.insert(0, " + repr(str(SCRIPTS)) + "); "
            "import run_masked_prediction_pilot; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fixed_seeds_and_count_bounds(self):
        self.assertEqual(_validate_count(4), FIXED_SEEDS[:4])
        self.assertEqual(len(_validate_count(8)), 8)
        for count in (0, 3, 9, True, 4.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                _validate_count(count)
        with patch("run_masked_prediction_pilot.signal.getitimer",
                   return_value=(0.0, 0.0)), patch(
                "run_masked_prediction_pilot.signal.setitimer") as set_timer:
            with _first_fixture_deadline(True):
                pass
        self.assertEqual(set_timer.call_args_list[0].args[1], 180.0)
        self.assertEqual(set_timer.call_args_list[1].args[1], 0.0)

    def test_fixture_paths_refuse_escape_links_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            (root / "frame.ppm").write_bytes(PPM_HEADER + bytes(64 * 64 * 3))
            self.assertEqual(
                _fixture_file(root, "frame.ppm", max_bytes=16 * 1024),
                (root / "frame.ppm").resolve(),
            )
            with self.assertRaises(ValueError):
                _fixture_file(root, "../outside", max_bytes=16 * 1024)
            outside = Path(directory) / "outside.ppm"
            outside.write_bytes(b"x")
            (root / "link.ppm").symlink_to(outside)
            with self.assertRaises(ValueError):
                _fixture_file(root, "link.ppm", max_bytes=16 * 1024)
            with self.assertRaises(FileExistsError):
                require_absent_outputs((root,))

    def test_visible_context_and_hidden_patch_contract(self):
        first = _fixture_frame(28, 30)
        same = _fixture_frame(28, 30)
        _validate_frame_pair(first, same, hidden=False)
        changed_inside = bytearray(first)
        offset = len(PPM_HEADER) + (20 * 64 + 20) * 3
        changed_inside[offset:offset + 3] = bytes((1, 2, 3))
        _validate_frame_pair(first, bytes(changed_inside), hidden=True)
        changed_outside = bytearray(first)
        offset = len(PPM_HEADER) + (8 * 64 + 20) * 3
        changed_outside[offset:offset + 3] = bytes((1, 2, 3))
        with self.assertRaises(ValueError):
            _validate_frame_pair(first, bytes(changed_outside), hidden=True)
        with self.assertRaises(ValueError):
            _validate_frame_pair(first, bytes(changed_inside), hidden=False)

        class FakeTensor:
            shape = (1, 64, 3, 256, 256)

            def __init__(self, differences=()):
                self.differences = set(differences)

            def __getitem__(self, key):
                _, frame, _, rows, columns = key
                return (self.differences, frame, rows.start, rows.stop,
                        columns.start, columns.stop)

        def fake_equal(_left, right):
            differences, frame, row_start, row_end, col_start, col_end = right
            return not any(
                changed_frame == frame and row_start <= row < row_end
                and col_start <= col < col_end
                for changed_frame, row, col in differences)

        reference = FakeTensor()
        # Variations inside a fully masked output patch are excluded from comparison.
        assert_preprocessed_visible_context_equal(
            reference, FakeTensor({(31, 100, 100)}), tensor_equal=fake_equal)
        # Row 192 is the first visible row after the mask; bilinear bleed here must stop.
        with self.assertRaisesRegex(ValueError, "preprocessed candidate context"):
            assert_preprocessed_visible_context_equal(
                reference, FakeTensor({(31, 192, 100)}), tensor_equal=fake_equal)
        # Other temporal tubelets are visible even inside the spatial box.
        with self.assertRaisesRegex(ValueError, "preprocessed candidate context"):
            assert_preprocessed_visible_context_equal(
                reference, FakeTensor({(34, 100, 100)}), tensor_equal=fake_equal)

    def _complete_score_record(self):
        scores = {
            "seeded_random": {"candidate-0": 0., "candidate-1": 1.,
                              "candidate-2": 2., "candidate-3": 3.},
            "visible_context_encoder_only_tie": {f"candidate-{i}": 0. for i in range(4)},
            "image_space_constant_velocity_interpolation": {
                "candidate-0": 0., "candidate-1": 1., "candidate-2": 2., "candidate-3": 3.},
            "masked_context_predictor": {
                "candidate-0": 1., "candidate-1": 0., "candidate-2": 2., "candidate-3": 3.},
        }
        return {"seed": FIXED_SEEDS[0], "root": Path("fake-seed"),
                "candidate_ids": [f"candidate-{i}" for i in range(4)], "scores": scores}

    def _answer(self):
        return {
            "schema_version": 1,
            "correct_candidate_id": "candidate-1",
            "rule": "constant horizontal velocity and constant vertical position through the gap",
            "claim_scope": "synthetic generator rule only; not a unique real-world inference",
            "ambiguous_by_visible_context": True,
            "ambiguity_note": "identical visible context",
        }

    def test_scores_all_baselines_before_answer_read(self):
        completed = [self._complete_score_record(), self._complete_score_record()]
        scoring_finished = {"value": True}
        reads = []

        def answer_reader(root):
            self.assertTrue(scoring_finished["value"])
            reads.append(root)
            return self._answer()

        rows, summary = evaluate_after_scoring(completed, answer_reader)
        self.assertEqual(len(reads), 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["methods"]["masked_context_predictor"]["count"], 2)

    def test_incomplete_scoring_never_reads_answers(self):
        item = self._complete_score_record()
        del item["scores"]["image_space_constant_velocity_interpolation"]
        reads = []
        with self.assertRaises(ValueError):
            evaluate_after_scoring([item], lambda root: reads.append(root) or self._answer())
        self.assertEqual(reads, [])

    def test_baselines_random_ties_interpolation_are_label_free(self):
        ids = [f"candidate-{index}" for index in range(4)]
        self.assertEqual(baselines.seeded_random_scores(ids, 17),
                         baselines.seeded_random_scores(ids, 17))
        self.assertEqual(set(baselines.visible_context_tie_scores(ids).values()), {0.0})
        visible = {f"visible/{frame}": _fixture_frame(12. + .4 * frame, 30.)
                   for frame in tuple(range(30)) + tuple(range(34, 64))}
        candidates = []
        for candidate_index, y_offset in enumerate((0., -5., 5., 8.)):
            paths = []
            for frame in range(64):
                path = f"candidate-{candidate_index}/{frame}"
                paths.append(path)
                if frame not in range(30, 34):
                    visible[path] = visible[f"visible/{frame}"]
                else:
                    visible[path] = _fixture_frame(12. + .4 * frame, 30. + y_offset)
            candidates.append({"id": ids[candidate_index], "frames": paths})

        scores = baselines.image_space_interpolation_scores(
            Path("."), candidates, read_frame=lambda _root, name: visible[name])
        self.assertEqual(min(scores, key=scores.get), "candidate-0")
        self.assertNotIn("correct_candidate_id", baselines.image_space_interpolation_scores.__code__.co_varnames)

    def test_ranking_metrics_finite_ties_and_denominators(self):
        tied = evaluate_ranking_metrics({f"candidate-{i}": 0. for i in range(4)}, "candidate-2")
        self.assertEqual(tied["correct_rank_expected"], 2.5)
        self.assertAlmostEqual(tied["top1_expected"], .25)
        self.assertAlmostEqual(tied["mrr_expected"], sum(1 / i for i in range(1, 5)) / 4)
        self.assertFalse(tied["top1_strict"])
        unique = evaluate_ranking_metrics(
            {"a": 1., "b": 0., "c": 2., "d": 3.}, "b")
        self.assertEqual(unique["correct_rank_min"], 1)
        self.assertTrue(unique["top1_strict"])
        self.assertAlmostEqual(unique["margin_min_wrong_minus_correct"], 1.0)
        with self.assertRaises(ValueError):
            evaluate_ranking_metrics({"a": float("nan"), "b": 0.}, "a")
        near_chain = evaluate_ranking_metrics(
            {"a": 0.0, "b": .9e-6, "c": 1.8e-6, "d": 3.0}, "c")
        self.assertEqual((near_chain["correct_rank_min"], near_chain["correct_rank_max"]), (3, 3))
        self.assertAlmostEqual(near_chain["mrr_expected"], 1 / 3)

    def test_index_binds_frames_without_answer_content(self):
        trajectories = [
            {"x_start": 12.1 + seed, "x_velocity": .4, "base_y": 30. + seed}
            for seed in range(2)
        ]
        records = [{
            "seed": FIXED_SEEDS[index],
            "manifest_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            "manifest_bytes": 10,
            "file_count_excluding_answer": 257,
            "bytes_excluding_answer": 1000,
            "visible_trajectory_template": trajectories[index],
            "candidates": [{"candidate_id": f"candidate-{i}",
                            "frame_count": 64,
                            "bytes": 10,
                            "ordered_frame_sha256": hashlib.sha256(
                                f"{index}-{i}".encode()).hexdigest()}
                           for i in range(4)],
        } for index in range(2)]
        source = {name: hashlib.sha256(name.encode()).hexdigest()
                  for name in ("generator", "pilot", "smoke", "baselines")}
        loader_hash = hashlib.sha256(b"loader").hexdigest()
        index = build_index(records, source_sha256=source,
                            processor_config_sha256=hashlib.sha256(b"config").hexdigest(),
                            compatibility_loader_sha256=loader_hash)
        encoded = json.dumps(index)
        self.assertNotIn("correct_candidate_id", encoded)
        self.assertNotIn("answer_key", encoded)
        self.assertEqual(index["seed_count"], 2)
        self.assertEqual(index["compatibility_loader_sha256"], loader_hash)

    def test_fixed_resources_and_strict_model_provenance(self):
        checkpoint = Path("/tmp/candidate.pt")
        identity = Path("/tmp/checkpoint-identity.json")
        validate_fixed_resources(checkpoint, identity, 1.0)
        with self.assertRaises(ValueError):
            validate_fixed_resources(Path("relative.pt"), identity, 1.0)
        with self.assertRaises(ValueError):
            validate_fixed_resources(checkpoint, identity, 2.0)
        with tempfile.TemporaryDirectory() as directory:
            resource = Path(directory) / "pinned.py"
            payload = b"from dataclasses import dataclass\nimport torch\n"
            resource.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            self.assertEqual(_read_pinned_bytes(
                resource, expected_sha256=digest, max_bytes=1024), payload)
            validate_loader_import_closure(payload)
            with self.assertRaises(ValueError):
                _read_pinned_bytes(resource, expected_sha256="0" * 64, max_bytes=1024)
            with self.assertRaisesRegex(ValueError, "pinned resource SHA-256 mismatch"):
                _load_verified_processor(resource)
            with self.assertRaisesRegex(ValueError, "pinned resource SHA-256 mismatch"):
                _load_verified_loader_callable(resource)
            with self.assertRaisesRegex(ValueError, "relative loader imports"):
                validate_loader_import_closure(b"from .predictor import load\n")
            with self.assertRaisesRegex(ValueError, "outside the reviewed closure"):
                validate_loader_import_closure(b"from jepa_risk.predictor import load\n")
        self.assertEqual(len(PROCESSOR_CONFIG_SHA256), 64)
        self.assertEqual(len(COMPATIBILITY_LOADER_SHA256), 64)
        source = (SCRIPTS / "run_masked_prediction_pilot.py").read_text(encoding="utf-8")
        self.assertIn("load_official_bundle", source)
        self.assertNotIn("strict=False", source)
        self.assertIn("_read_identity_safe(identity_record, checkpoint)", source)
        self.assertIn("PROCESSOR_CONFIG_SHA256", source)
        self.assertIn("config_source.decode(\"utf-8\")", source)
        self.assertIn("compile(loader_source", source)
        self.assertNotIn("from jepa_risk.predictor import", source)

    def test_generator_support_bounds_include_processor_margin_all_fixed_seeds(self):
        from generate_ball_gap import (
            GAP, SAFE_SOURCE_DIFF_MAX, SAFE_SOURCE_DIFF_MIN,
            conservative_candidate_source_support, render_candidate_frames_in_memory,
            seeded_trajectory,
            transformed_bilinear_support,
        )

        source_min, source_max = conservative_candidate_source_support()
        self.assertEqual((source_min, source_max), (SAFE_SOURCE_DIFF_MIN, SAFE_SOURCE_DIFF_MAX))
        support_min, support_max = transformed_bilinear_support(source_min, source_max)
        self.assertGreaterEqual(support_min, 64)
        self.assertLess(support_max, 192)
        trajectories = [seeded_trajectory(seed) for seed in FIXED_SEEDS]
        self.assertEqual(len({tuple(item[key] for key in ("x_start", "x_velocity", "base_y"))
                              for item in trajectories}), len(FIXED_SEEDS))
        for seed in FIXED_SEEDS:
            candidates = render_candidate_frames_in_memory(seed)
            self.assertEqual(len(candidates), 4)
            self.assertTrue(all(len(candidate) == 64 for candidate in candidates))
            header = b"P6\n64 64\n255\n"
            for candidate in candidates[1:]:
                for frame_index in range(64):
                    first = candidates[0][frame_index]
                    other = candidate[frame_index]
                    if frame_index not in GAP:
                        self.assertEqual(first, other)
                        continue
                    self.assertTrue(first.startswith(header) and other.startswith(header))
                    first_pixels, other_pixels = first[len(header):], other[len(header):]
                    changed = [index for index in range(64 * 64)
                               if first_pixels[index * 3:index * 3 + 3]
                               != other_pixels[index * 3:index * 3 + 3]]
                    self.assertGreater(len(changed), 0)
                    ys, xs = zip(*(divmod(index, 64) for index in changed))
                    self.assertGreater(len(xs), 0)
                    self.assertGreaterEqual(min(xs), source_min)
                    self.assertLessEqual(max(xs), source_max)
                    self.assertGreaterEqual(min(ys), source_min)
                    self.assertLessEqual(max(ys), source_max)
            del candidates

    def test_real_processor_visible_context_equal_all_fixed_seeds(self):
        import gc
        try:
            import torch
            import transformers  # noqa: F401
        except ImportError as error:
            self.skipTest(f"local torch/Transformers unavailable; processor integration test not run: {error}")
        processor_config = os.environ.get("JEPA_PROCESSOR_CONFIG")
        if not processor_config:
            self.skipTest("JEPA_PROCESSOR_CONFIG is unset; no local path discovery or download")
        from generate_ball_gap import render_candidate_frames_in_memory

        loaded_torch, processor, config_hash, config_bytes = _load_verified_processor(Path(processor_config))
        self.assertIs(loaded_torch, torch)
        self.assertEqual(config_hash, PROCESSOR_CONFIG_SHA256)
        self.assertGreater(config_bytes, 0)
        for seed in FIXED_SEEDS:
            generated = render_candidate_frames_in_memory(seed)
            videos = []
            for frame_bytes in generated:
                frames = [_load_ppm_bytes(frame) for frame in frame_bytes]
                encoded = processor([frames], do_resize=True, do_center_crop=True,
                                    input_data_format="channels_last", return_tensors="pt")
                video = encoded["pixel_values_videos"].to(device="cpu", dtype=torch.float32)
                self.assertEqual(tuple(video.shape), (1, 64, 3, 256, 256))
                videos.append(video)
                del frames, encoded, video
            for video in videos[1:]:
                assert_preprocessed_visible_context_equal(
                    videos[0], video, tensor_equal=torch.equal)
            del generated, videos
            gc.collect()

    def test_fixture_stage_and_forward_started_completed_events(self):
        import argparse
        import io
        from contextlib import redirect_stdout

        class FlushAwareStream(io.StringIO):
            def __init__(self):
                super().__init__()
                self.flushes = 0

            def flush(self):
                self.flushes += 1
                super().flush()

        stream = FlushAwareStream()
        progress = PilotProgress(stream)
        progress.begin_fixture(0, FIXED_SEEDS[0])

        def fake_forward():
            self.assertGreaterEqual(stream.flushes, 2)
            events = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(events[-1]["stage"], "forward_started")
            return "ok"

        result = _tracked_forward(progress, "encoder", fake_forward)
        self.assertEqual(result, "ok")
        progress.mark_fixture_scores_completed()
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([event["stage"] for event in events], [
            "fixture_started", "forward_started", "forward_returned", "fixture_scores_completed"])
        self.assertEqual(events[1]["attempted_count"], 1)
        self.assertEqual(events[1]["returned_count"], 0)
        self.assertEqual(events[2]["returned_count"], 1)
        self.assertEqual(progress.summary()["completed_fixture_scores"], 1)
        progress.begin_fixture(1, FIXED_SEEDS[1])
        self.assertEqual(_tracked_forward(progress, "encoder", lambda: "ok"), "ok")
        progress.mark_fixture_scores_completed()
        progress.emit_run("run_completed")
        global_event = json.loads(stream.getvalue().splitlines()[-1])
        self.assertIsNone(global_event["fixture_index"])
        self.assertEqual(global_event["attempted_count"], 2)
        self.assertEqual(global_event["returned_count"], 2)
        self.assertEqual(global_event["completed_fixture_scores"], 2)

        failed_stream = FlushAwareStream()
        failed = PilotProgress(failed_stream)
        failed.begin_fixture(1, FIXED_SEEDS[1])
        with self.assertRaisesRegex(RuntimeError, "secret-marker"):
            _tracked_forward(failed, "predictor", lambda: (_ for _ in ()).throw(
                RuntimeError("secret-marker")))
        failed_events = [json.loads(line) for line in failed_stream.getvalue().splitlines()]
        self.assertEqual(failed_events[-1]["stage"], "forward_failed")
        self.assertEqual(failed_events[-1]["attempted_count"], 1)
        self.assertEqual(failed_events[-1]["returned_count"], 0)
        self.assertNotIn("secret-marker", failed_stream.getvalue())

        interrupted_stream = FlushAwareStream()
        interrupted = PilotProgress(interrupted_stream)
        interrupted.begin_fixture(2, FIXED_SEEDS[2])
        with self.assertRaises(KeyboardInterrupt):
            _tracked_forward(interrupted, "target_encoder",
                             lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
        interrupted_events = [json.loads(line) for line in interrupted_stream.getvalue().splitlines()]
        self.assertEqual(interrupted_events[-1]["stage"], "forward_interrupted")
        self.assertEqual(interrupted_events[-1]["attempted_count"], 1)
        self.assertEqual(interrupted_events[-1]["returned_count"], 0)

        checkpoint = Path("/tmp/candidate.pt")
        identity = Path("/tmp/checkpoint-identity.json")
        processor_config = Path("/tmp/processor-config.json")
        compatibility_loader = Path("/tmp/compatibility-loader.py")
        args = argparse.Namespace(
            dry_run=False, execute=True, count=4, checkpoint=checkpoint,
            identity_record=identity, processor_config=processor_config,
            compatibility_loader=compatibility_loader, verified_loss_exp=1.0)
        main_output = io.StringIO()
        with patch("run_masked_prediction_pilot.argparse.ArgumentParser.parse_args",
                   return_value=args), patch("run_masked_prediction_pilot.os.environ.get",
                                             return_value="1"), patch(
                "run_masked_prediction_pilot.run_pilot", side_effect=KeyboardInterrupt()), \
                redirect_stdout(main_output):
            exit_code = main()
        self.assertEqual(exit_code, 130)
        self.assertNotIn("Traceback", main_output.getvalue())
        self.assertNotIn("/Users/", main_output.getvalue())
        self.assertIn('"safe_error_code":"interrupted"', main_output.getvalue())

    def test_repair_keeps_fixed_mask_and_uses_separate_outputs(self):
        from masked_prediction import token_partition
        from run_masked_prediction_pilot import PROJECT_DIR

        _, target_ids = token_partition(
            temporal_tokens=32, height_tokens=16, width_tokens=16,
            target_temporal=(15, 16), target_rows=range(4, 12), target_cols=range(4, 12))
        self.assertEqual(len(target_ids), 128)
        # The historical failure artifact is intentionally excluded from Git. Keep this
        # clean-clone test focused on the tracked repair contract: fixed mask and a new versioned output path.
        self.assertEqual(FIXTURE_ROOT, PROJECT_DIR / "fixtures" / "pilot-20261102-v1")
        self.assertEqual(REPORT_JSON, PROJECT_DIR / "reports" / "masked-prediction-pilot-20261102-v1.json")
        self.assertEqual(REPORT_MARKDOWN, PROJECT_DIR / "reports" / "masked-prediction-pilot-20261102-v1.md")
        self.assertFalse(FIXTURE_ROOT.exists())
        self.assertNotEqual(FIXTURE_ROOT, PROJECT_DIR / "fixtures" / "pilot-20261101-v2")


if __name__ == "__main__":
    unittest.main()
