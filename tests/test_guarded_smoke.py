from __future__ import annotations

import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_guarded_smoke import (HELPER_TIMEOUT_SECONDS, NETWORK_DENY_PROFILE,
                               MAX_RSS, MAX_WALL_SECONDS, PARITY_PYTHON, PARITY_REPORT,
                               PILOT_FIXTURE_ROOT, PILOT_REPORT_JSON, PILOT_REPORT_MARKDOWN,
                               _child_command, _dummy_probe_source, _process_group_exists,
                               build_native_parity_command, build_pilot_command,
                               _stop_process, parse_vm_stat,
                               main as guarded_main, read_available_memory, read_rss_bytes,
                               run_limited_process)
from run_masked_prediction_smoke import PROJECT_DIR, _fixture_file

SLEEPER = "import signal,time; signal.signal(signal.SIGINT, lambda *args: exit(0)); time.sleep(5)"
CHECKPOINT = ROOT / "fixtures" / "dummy.pt"
IDENTITY = ROOT / "fixtures" / "dummy-identity.json"
PROCESSOR_CONFIG = ROOT / "fixtures" / "dummy-processor.json"
COMPATIBILITY_LOADER = ROOT / "fixtures" / "dummy-loader.py"


class GuardedRunnerTests(unittest.TestCase):
    def test_pilot_mode_selects_fixed_script_and_paths(self):
        command = build_pilot_command(
            python=PARITY_PYTHON, checkpoint=CHECKPOINT, identity_record=IDENTITY,
            processor_config=PROCESSOR_CONFIG, compatibility_loader=COMPATIBILITY_LOADER,
            count=4, device="cpu", timeout=600,
            verified_loss_exp=1.0,
        )
        self.assertEqual(command[0], PARITY_PYTHON)
        self.assertEqual(command[1], str(ROOT / "scripts" / "run_masked_prediction_pilot.py"))
        self.assertEqual(PILOT_FIXTURE_ROOT, "fixtures/pilot-20261101")
        self.assertEqual(PILOT_REPORT_JSON, "reports/masked-prediction-pilot-20261101.json")
        self.assertEqual(PILOT_REPORT_MARKDOWN, "reports/masked-prediction-pilot-20261101.md")
        self.assertIn(str(CHECKPOINT), command)
        self.assertIn(str(PROCESSOR_CONFIG), command)
        self.assertIn(str(COMPATIBILITY_LOADER), command)
        self.assertNotIn("shell", command)
        argv = ["run_guarded_smoke.py", "--pilot", "--pilot-count", "4",
                "--python", PARITY_PYTHON, "--device", "cpu",
                "--verified-loss-exp", "1.0", "--timeout", "600",
                "--checkpoint", str(CHECKPOINT), "--identity-record", str(IDENTITY),
                "--processor-config", str(PROCESSOR_CONFIG),
                "--compatibility-loader", str(COMPATIBILITY_LOADER)]
        with patch("run_guarded_smoke.run_limited_process",
                   return_value=(0, "child exited")) as launch:
            with patch.object(sys, "argv", argv):
                self.assertEqual(guarded_main(), 0)
        launch.assert_called_once()
        self.assertEqual(launch.call_args.args[0], command)

    def test_pilot_mode_forces_cpu_budget_and_network_deny(self):
        for count, device, timeout, loss_exp in (
                (3, "cpu", 600, 1.0), (9, "cpu", 600, 1.0),
                (4, "mps", 600, 1.0), (4, "cpu", 601, 1.0),
                (4, "cpu", 600, 2.0)):
            with self.subTest(count=count, device=device, timeout=timeout, loss_exp=loss_exp):
                with self.assertRaises(ValueError):
                    build_pilot_command(
                        python=PARITY_PYTHON, checkpoint=CHECKPOINT,
                        identity_record=IDENTITY, processor_config=PROCESSOR_CONFIG,
                        compatibility_loader=COMPATIBILITY_LOADER,
                        count=count, device=device,
                        timeout=timeout, verified_loss_exp=loss_exp)
        self.assertEqual(MAX_WALL_SECONDS, 600)
        self.assertEqual(MAX_RSS, 16 * 1024**3)
        argv = ["run_guarded_smoke.py", "--pilot", "--pilot-count", "4",
                "--python", PARITY_PYTHON, "--device", "cpu",
                "--verified-loss-exp", "1.0", "--timeout", "600",
                "--checkpoint", str(CHECKPOINT), "--identity-record", str(IDENTITY),
                "--processor-config", str(PROCESSOR_CONFIG),
                "--compatibility-loader", str(COMPATIBILITY_LOADER)]
        with patch.object(sys, "argv", argv), patch(
                "run_guarded_smoke.run_limited_process", return_value=(0, "child exited")) as launch:
            self.assertEqual(guarded_main(), 0)
        kwargs = launch.call_args.kwargs
        self.assertEqual(kwargs["timeout"], 600)
        self.assertEqual(kwargs["start_min_available"], 32 * 1024**3)
        self.assertEqual(kwargs["stop_min_available"], 16 * 1024**3)
        self.assertEqual(kwargs["max_rss"], 16 * 1024**3)
        self.assertTrue(kwargs["network_deny"])

    def test_parity_mode_selects_fixed_script_without_shell(self):
        command = build_native_parity_command(
            python=PARITY_PYTHON, checkpoint=CHECKPOINT,
            identity_record=IDENTITY, report=__import__("pathlib").Path(PARITY_REPORT),
            device="cpu", timeout=600,
        )
        self.assertEqual(command[0], PARITY_PYTHON)
        self.assertEqual(command[1], str(ROOT / "scripts" / "check_native_predictor_parity.py"))
        self.assertNotIn("shell", command)
        self.assertIn("--report", command)
        with patch("run_guarded_smoke.run_limited_process") as launched:
            argv = [
                "run_guarded_smoke.py", "--dummy-network-probe", "--native-parity",
                "--python", PARITY_PYTHON, "--checkpoint", str(CHECKPOINT),
                "--identity-record", str(IDENTITY), "--report", PARITY_REPORT,
                "--device", "cpu", "--timeout", "600",
            ]
            with redirect_stderr(io.StringIO()), patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                guarded_main()
            launched.assert_not_called()

    def test_parity_mode_refuses_unregistered_output_path(self):
        with self.assertRaises(ValueError):
            build_native_parity_command(
                python=PARITY_PYTHON, checkpoint=CHECKPOINT,
                identity_record=IDENTITY,
                report=__import__("pathlib").Path("reports/elsewhere.json"),
                device="cpu", timeout=600,
            )

    def test_model_runner_rejects_a_different_python_or_relative_resources(self):
        with self.assertRaisesRegex(ValueError, "guardian's current Python"):
            build_native_parity_command(
                python="/tmp/other-python", checkpoint=CHECKPOINT,
                identity_record=IDENTITY,
                report=__import__("pathlib").Path(PARITY_REPORT),
                device="cpu", timeout=600,
            )
        with self.assertRaisesRegex(ValueError, "explicit absolute paths"):
            build_pilot_command(
                python=PARITY_PYTHON, checkpoint=Path("relative.pt"),
                identity_record=IDENTITY, processor_config=PROCESSOR_CONFIG,
                compatibility_loader=COMPATIBILITY_LOADER,
                count=4, device="cpu", timeout=600, verified_loss_exp=1.0,
            )

    def test_parity_mode_forces_cpu_and_hard_bounds(self):
        for device, timeout in (("mps", 600), ("cpu", 601)):
            with self.subTest(device=device, timeout=timeout), self.assertRaises(ValueError):
                build_native_parity_command(
                    python=PARITY_PYTHON, checkpoint=CHECKPOINT,
                    identity_record=IDENTITY,
                    report=__import__("pathlib").Path(PARITY_REPORT),
                    device=device, timeout=timeout,
                )
        self.assertEqual(MAX_WALL_SECONDS, 600)
        self.assertEqual(MAX_RSS, 16 * 1024**3)
        argv = [
            "run_guarded_smoke.py", "--native-parity", "--python", PARITY_PYTHON,
            "--checkpoint", str(CHECKPOINT), "--identity-record", str(IDENTITY),
            "--report", PARITY_REPORT, "--device", "cpu", "--timeout", "600",
        ]
        with patch.object(sys, "argv", argv), patch(
                "run_guarded_smoke.run_limited_process", return_value=(0, "child exited")) as launch:
            self.assertEqual(guarded_main(), 0)
        launch.assert_called_once()
        kwargs = launch.call_args.kwargs
        self.assertEqual(kwargs["timeout"], 600)
        self.assertEqual(kwargs["start_min_available"], 32 * 1024**3)
        self.assertEqual(kwargs["stop_min_available"], 16 * 1024**3)
        self.assertEqual(kwargs["max_rss"], 16 * 1024**3)
        self.assertTrue(kwargs["network_deny"])

    def test_vm_stat_parser_counts_only_free_inactive_and_speculative(self):
        result = parse_vm_stat("""Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               10.
Pages inactive:                           20.
Pages speculative:                        3.
Pages wired down:                         999.
""")
        self.assertEqual(result, 33 * 16384)

    def test_preflight_measurement_failure_does_not_launch_child(self):
        launched = []
        code, message = run_limited_process(
            [sys.executable, "-c", "pass"],
            available_memory_reader=lambda: (_ for _ in ()).throw(OSError("missing")),
            popen_factory=lambda *a, **k: launched.append((a, k)),
        )
        self.assertEqual(code, 2)
        self.assertIn("not started", message)
        self.assertEqual(launched, [])

    def test_low_start_memory_does_not_launch_child(self):
        launched = []
        code, message = run_limited_process(
            [sys.executable, "-c", "pass"],
            available_memory_reader=lambda: 31 * 1024**3,
            popen_factory=lambda *a, **k: launched.append((a, k)),
        )
        self.assertEqual(code, 2)
        self.assertIn("below start threshold", message)
        self.assertEqual(launched, [])

    def test_child_receives_offline_environment_without_parent_environment_copy(self):
        observed = {}

        class FinishedProcess:
            pid = 123
            returncode = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

        def popen(_command, *, env, start_new_session):
            observed.update(env)
            observed["start_new_session"] = start_new_session
            return FinishedProcess()

        code, _ = run_limited_process(
            [sys.executable, "-c", "pass"],
            available_memory_reader=lambda: 40 * 1024**3,
            rss_reader=lambda _pid: 4096,
            popen_factory=popen,
        )
        self.assertEqual(code, 0)
        self.assertEqual(observed["HF_HUB_OFFLINE"], "1")
        self.assertEqual(observed["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(observed["HF_DATASETS_OFFLINE"], "1")
        self.assertEqual(observed["JEPA_GUARDED_RUN"], "1")
        self.assertTrue(observed["start_new_session"])
        self.assertNotIn("HOME", observed)

    def test_network_deny_wraps_only_the_child_command(self):
        command = [sys.executable, "-c", "pass"]
        self.assertEqual(_child_command(command, network_deny=False), command)
        self.assertEqual(
            _child_command(command, network_deny=True),
            ["/usr/bin/sandbox-exec", "-p", NETWORK_DENY_PROFILE, *command],
        )

    def test_dummy_child_source_is_valid_and_marks_network_and_timeout(self):
        source = _dummy_probe_source(30)
        compile(source, "<dummy-network-probe>", "exec")
        self.assertIn("DUMMY_CHILD_STARTED", source)
        self.assertIn("DUMMY_NETWORK_ERRNO", source)
        self.assertIn("time.sleep(30)", source)

    def test_dummy_probe_rejects_a_different_python_interpreter(self):
        argv = ["run_guarded_smoke.py", "--dummy-network-probe", "--python", "/tmp/other-python"]
        with patch.object(sys, "argv", argv), patch(
                "run_guarded_smoke.run_limited_process") as launch, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            guarded_main()
        launch.assert_not_called()

    def test_rss_preflight_failure_does_not_start_child(self):
        launched = []
        code, message = run_limited_process(
            [sys.executable, "-c", "pass"],
            available_memory_reader=lambda: 40 * 1024**3,
            rss_reader=lambda _pid: (_ for _ in ()).throw(PermissionError(1, "denied")),
            popen_factory=lambda *a, **k: launched.append((a, k)),
        )
        self.assertEqual(code, 2)
        self.assertIn("RSS preflight unavailable", message)
        self.assertEqual(launched, [])

    def test_python_socket_guard_blocks_before_any_network_call(self):
        code = (
            "import sys; sys.path.insert(0, " + repr(str(PROJECT_DIR / "scripts")) + "); "
            "import run_masked_prediction_smoke as r; r._install_best_effort_network_guard(); "
            "import socket; socket.create_connection(('127.0.0.1', 1))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("network access is disabled", result.stderr)

    def test_wall_clock_limit_interrupts_dummy_child(self):
        code, message = run_limited_process(
            [sys.executable, "-c", SLEEPER],
            available_memory_reader=lambda: 40 * 1024**3,
            rss_reader=lambda _pid: 1,
            timeout=0.1,
            poll_interval=0.02,
            interrupt_grace=1,
        )
        self.assertEqual(code, 124)
        self.assertIn("wall-clock limit", message)

    def test_rss_limit_stops_dummy_child(self):
        code, message = run_limited_process(
            [sys.executable, "-c", SLEEPER],
            available_memory_reader=lambda: 40 * 1024**3,
            rss_reader=lambda _pid: 2,
            max_rss=1,
            timeout=5,
            poll_interval=0.02,
            interrupt_grace=1,
        )
        self.assertEqual(code, 2)
        self.assertIn("RSS limit", message)

    def test_runtime_measurement_failure_stops_dummy_child(self):
        reads = 0

        def memory_reader():
            nonlocal reads
            reads += 1
            if reads == 1:
                return 40 * 1024**3
            raise OSError("measurement lost")

        log = io.StringIO()
        with redirect_stderr(log):
            code, message = run_limited_process(
                [sys.executable, "-c", SLEEPER],
                available_memory_reader=memory_reader,
                rss_reader=lambda _pid: 1,
                timeout=5,
                poll_interval=0.02,
                interrupt_grace=1,
            )
        self.assertEqual(code, 2)
        self.assertIn("measurement unavailable", message)
        self.assertIn('"reader": "available_memory_reader"', log.getvalue())
        self.assertIn('"stop_reason": "measurement_failed"', log.getvalue())
        self.assertIn('"waited": true', log.getvalue())

    def test_child_exit_is_logged_as_waited(self):
        log = io.StringIO()
        with redirect_stderr(log):
            code, message = run_limited_process(
                [sys.executable, "-c", "pass"],
                available_memory_reader=lambda: 40 * 1024**3,
                rss_reader=lambda _pid: 4096,
            )
        self.assertEqual(code, 0)
        self.assertEqual(message, "child exited")
        self.assertIn('"guard_event": "preflight_ok"', log.getvalue())
        self.assertIn('"guard_event": "child_waited"', log.getvalue())
        self.assertIn('"waited": true', log.getvalue())

    def test_low_runtime_memory_stops_dummy_child(self):
        reads = 0

        def memory_reader():
            nonlocal reads
            reads += 1
            return 40 * 1024**3 if reads == 1 else 8 * 1024**3

        code, message = run_limited_process(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            available_memory_reader=memory_reader,
            rss_reader=lambda _pid: 1,
            timeout=5,
            poll_interval=0.02,
            interrupt_grace=1,
        )
        self.assertEqual(code, 2)
        self.assertIn("below stop threshold", message)

    def test_memory_reader_timeout_stops_and_waits_child(self):
        for reader, args in ((read_available_memory, ()), (read_rss_bytes, (123,))):
            with patch("run_guarded_smoke.subprocess.run",
                       side_effect=subprocess.TimeoutExpired("synthetic-helper", HELPER_TIMEOUT_SECONDS)) as mocked:
                with self.assertRaises(subprocess.TimeoutExpired):
                    reader(*args)
                self.assertEqual(mocked.call_args.kwargs["timeout"], HELPER_TIMEOUT_SECONDS)

        reads = 0

        def memory_reader():
            nonlocal reads
            reads += 1
            if reads == 1:
                return 40 * 1024**3
            raise subprocess.TimeoutExpired("vm_stat", 2)

        log = io.StringIO()
        with redirect_stderr(log):
            code, message = run_limited_process(
                [sys.executable, "-c", SLEEPER],
                available_memory_reader=memory_reader,
                rss_reader=lambda _pid: 1,
                timeout=5,
                poll_interval=0.02,
                interrupt_grace=1,
            )
        self.assertEqual(code, 2)
        self.assertIn("measurement unavailable", message)
        self.assertIn('"exception": "TimeoutExpired"', log.getvalue())
        self.assertIn('"waited": true', log.getvalue())

    def test_monitor_exception_always_stops_and_waits_child(self):
        rss_reads = 0

        def rss_reader(_pid):
            nonlocal rss_reads
            rss_reads += 1
            if rss_reads == 1:
                return 1
            raise OSError("synthetic monitor failure")

        log = io.StringIO()
        with redirect_stderr(log):
            code, message = run_limited_process(
                [sys.executable, "-c", SLEEPER],
                available_memory_reader=lambda: 40 * 1024**3,
                rss_reader=rss_reader,
                timeout=5,
                poll_interval=0.02,
                interrupt_grace=1,
            )
        self.assertEqual(code, 2)
        self.assertIn("resource measurement unavailable", message)
        self.assertIn('"reader": "rss_reader"', log.getvalue())
        self.assertIn('"waited": true', log.getvalue())

        orphan_source = (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])"
        )
        orphan_parent = subprocess.Popen(
            [sys.executable, "-c", orphan_source], start_new_session=True)
        try:
            orphan_parent.wait(timeout=5)
            self.assertTrue(_process_group_exists(orphan_parent.pid))
            cleanup = _stop_process(orphan_parent, 0.2)
            self.assertTrue(cleanup["waited"])
            self.assertTrue(cleanup["process_group_gone"])
        finally:
            if _process_group_exists(orphan_parent.pid):
                _stop_process(orphan_parent, 0.2)


if __name__ == "__main__":
    unittest.main()
