#!/usr/bin/env python3
"""Run one offline JEPA smoke while keeping the resource guardian outside the child sandbox."""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
START_MIN_AVAILABLE = 32 * 1024**3
STOP_MIN_AVAILABLE = 16 * 1024**3
MAX_RSS = 16 * 1024**3
MAX_WALL_SECONDS = 10 * 60
PARITY_PYTHON = sys.executable
PARITY_REPORT = "reports/native-predictor-parity-20261101.json"
PILOT_FIXTURE_ROOT = "fixtures/pilot-20261101"
PILOT_REPORT_JSON = "reports/masked-prediction-pilot-20261101.json"
PILOT_REPORT_MARKDOWN = "reports/masked-prediction-pilot-20261101.md"
INTERRUPT_GRACE_SECONDS = 30
HELPER_TIMEOUT_SECONDS = 2
NETWORK_DENY_PROFILE = "(version 1) (allow default) (deny network*)"
_EVENT_LOG_FAILED = False


def _event(event: str, **fields) -> None:
    global _EVENT_LOG_FAILED
    try:
        print(json.dumps({"guard_event": event, **fields}, sort_keys=True), file=sys.stderr, flush=True)
    except BaseException:
        _EVENT_LOG_FAILED = True


def parse_vm_stat(output: str) -> int:
    """Estimate available bytes from macOS free+inactive+speculative pages."""
    import re

    page = re.search(r"page size of ([0-9,]+) bytes", output)
    if not page:
        raise ValueError("vm_stat did not report page size")
    page_size = int(page.group(1).replace(",", ""))
    counts = {}
    for line in output.splitlines():
        match = re.match(r"Pages (free|inactive|speculative):\s+([0-9,]+)", line)
        if match:
            counts[match.group(1)] = int(match.group(2).replace(",", ""))
    if set(counts) != {"free", "inactive", "speculative"}:
        raise ValueError("vm_stat omitted required page counters")
    return page_size * sum(counts.values())


def read_available_memory() -> int:
    result = subprocess.run(["/usr/bin/vm_stat"], check=True, capture_output=True,
                            text=True, timeout=HELPER_TIMEOUT_SECONDS)
    return parse_vm_stat(result.stdout)


def read_rss_bytes(pid: int) -> int:
    result = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(pid)],
                            check=True, capture_output=True, text=True,
                            timeout=HELPER_TIMEOUT_SECONDS)
    value = result.stdout.strip()
    if not value.isdigit():
        raise ValueError("ps did not return a numeric RSS")
    return int(value) * 1024


def _reader_call(name, stage, reader, *args):
    try:
        return reader(*args)
    except Exception as error:
        _event("measurement_error", stage=stage, reader=name,
               exception=type(error).__name__, errno=getattr(error, "errno", None),
               message=str(error))
        raise


def _child_command(command: list[str], *, network_deny: bool) -> list[str]:
    if not network_deny:
        return command
    return ["/usr/bin/sandbox-exec", "-p", NETWORK_DENY_PROFILE, *command]


def _stop_process(proc, grace: float, clock=time.monotonic, sleep=time.sleep,
                  *, manage_process_group: bool = True) -> dict:
    requested_signal = None
    escalation = None
    signal_errors = []

    def safe_clock() -> float:
        try:
            return clock()
        except BaseException:
            return time.monotonic()

    def safe_sleep(seconds: float) -> None:
        try:
            sleep(seconds)
        except BaseException:
            time.sleep(max(0.0, seconds))

    def request(signum: int) -> None:
        try:
            _signal_process_group(proc, signum, manage_process_group=manage_process_group)
        except OSError as error:
            signal_errors.append({"signal": signal.Signals(signum).name,
                                  "errno": getattr(error, "errno", None)})
            if proc.poll() is None:
                try:
                    proc.send_signal(signum)
                except OSError as fallback_error:
                    signal_errors.append({"signal": signal.Signals(signum).name,
                                          "fallback_errno": getattr(fallback_error, "errno", None)})

    direct_child_running = proc.poll() is None
    group_running = manage_process_group and _process_group_exists(proc.pid)
    if direct_child_running:
        requested_signal = "SIGINT"
        _event("stop_requested", pid=proc.pid, signal=requested_signal)
        request(signal.SIGINT)
        deadline = safe_clock() + grace
        while (proc.poll() is None or (manage_process_group and _process_group_exists(proc.pid))) and safe_clock() < deadline:
            safe_sleep(min(0.05, max(0.0, deadline - safe_clock())))
        if proc.poll() is None or (manage_process_group and _process_group_exists(proc.pid)):
            escalation = "SIGTERM"
            request(signal.SIGTERM)
            deadline = safe_clock() + max(1.0, grace)
            while (proc.poll() is None or (manage_process_group and _process_group_exists(proc.pid))) and safe_clock() < deadline:
                safe_sleep(min(0.05, max(0.0, deadline - safe_clock())))
            if proc.poll() is None or (manage_process_group and _process_group_exists(proc.pid)):
                escalation = "SIGKILL"
                request(signal.SIGKILL)
    elif group_running:
        requested_signal = "SIGTERM"
        _event("stop_requested", pid=proc.pid, signal=requested_signal)
        request(signal.SIGTERM)
        deadline = safe_clock() + max(1.0, grace)
        while _process_group_exists(proc.pid) and safe_clock() < deadline:
            safe_sleep(min(0.05, max(0.0, deadline - safe_clock())))
        if _process_group_exists(proc.pid):
            escalation = "SIGKILL"
            request(signal.SIGKILL)
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError as error:
            signal_errors.append({"signal": "SIGKILL", "fallback_errno": getattr(error, "errno", None)})
    returncode = proc.wait()
    return {"requested_signal": requested_signal, "escalation": escalation,
            "returncode": returncode, "waited": True, "signal_errors": signal_errors,
            "process_group_gone": (not _process_group_exists(proc.pid)
                                   if manage_process_group else None)}


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process_group(proc, signum: int, *, manage_process_group: bool) -> None:
    if not manage_process_group:
        proc.send_signal(signum)
        return
    try:
        os.killpg(proc.pid, signum)
    except ProcessLookupError:
        if proc.poll() is None:
            proc.send_signal(signum)


def _stop_remaining_group(pgid: int, grace: float, clock=time.monotonic, sleep=time.sleep) -> dict:
    """Terminate descendants left behind after the direct child exits; they cannot be waited here."""
    if not _process_group_exists(pgid):
        return {"descendants_found": False, "signal": None, "group_gone": True}
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return {"descendants_found": False, "signal": None, "group_gone": True}
    except OSError as error:
        return {"descendants_found": True, "signal": None, "group_gone": False,
                "signal_errno": getattr(error, "errno", None)}
    deadline = clock() + max(0.0, grace)
    while _process_group_exists(pgid) and clock() < deadline:
        sleep(min(0.05, max(0.0, deadline - clock())))
    if _process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError as error:
            return {"descendants_found": True, "signal": "SIGTERM", "group_gone": False,
                    "signal_errno": getattr(error, "errno", None)}
    return {"descendants_found": True, "signal": "SIGTERM/SIGKILL",
            "group_gone": not _process_group_exists(pgid)}


def run_limited_process(
    command: list[str],
    *,
    available_memory_reader=read_available_memory,
    rss_reader=read_rss_bytes,
    popen_factory=subprocess.Popen,
    timeout: float = MAX_WALL_SECONDS,
    start_min_available: int = START_MIN_AVAILABLE,
    stop_min_available: int = STOP_MIN_AVAILABLE,
    max_rss: int = MAX_RSS,
    poll_interval: float = 0.5,
    interrupt_grace: float = INTERRUPT_GRACE_SECONDS,
    clock=time.monotonic,
    sleep=time.sleep,
    network_deny: bool = False,
) -> tuple[int, str]:
    """Preflight guardian metrics, then monitor a child with fail-closed limits."""
    global _EVENT_LOG_FAILED
    _EVENT_LOG_FAILED = False
    try:
        available = _reader_call("available_memory_reader", "preflight", available_memory_reader)
    except Exception as error:
        _event("preflight_failed", reader="available_memory_reader",
               exception=type(error).__name__, errno=getattr(error, "errno", None))
        return 2, f"memory preflight unavailable ({type(error).__name__}); child not started"
    if available < start_min_available:
        _event("preflight_failed", reader="available_memory_reader", available_bytes=available,
               required_bytes=start_min_available)
        return 2, f"available memory {available} bytes below start threshold; child not started"
    try:
        guardian_rss = _reader_call("rss_reader", "preflight", rss_reader, os.getpid())
    except Exception as error:
        _event("preflight_failed", reader="rss_reader", exception=type(error).__name__,
               errno=getattr(error, "errno", None))
        return 2, f"RSS preflight unavailable ({type(error).__name__}); child not started"
    _event("preflight_ok", available_bytes=available, guardian_rss_bytes=guardian_rss,
           required_available_bytes=start_min_available)
    if _EVENT_LOG_FAILED:
        return 2, "resource preflight telemetry unavailable; child not started"

    child_command = _child_command(command, network_deny=network_deny)
    try:
        proc = popen_factory(child_command, env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "JEPA_GUARDED_RUN": "1",
        }, start_new_session=True)
    except Exception as error:
        _event("child_start_failed", exception=type(error).__name__,
               errno=getattr(error, "errno", None), message=str(error))
        return 2, f"could not start guarded child ({type(error).__name__})"
    manage_process_group = isinstance(proc, subprocess.Popen)
    try:
        started = clock()
    except BaseException:
        _stop_process(proc, interrupt_grace, manage_process_group=manage_process_group)
        raise
    try:
        _event("child_process_started", pid=proc.pid, network_deny=network_deny,
               command=child_command)
    except BaseException:
        _stop_process(proc, interrupt_grace, time.monotonic, time.sleep,
                      manage_process_group=manage_process_group)
        raise
    if _EVENT_LOG_FAILED:
        stop = _stop_process(proc, interrupt_grace, time.monotonic, time.sleep,
                             manage_process_group=manage_process_group)
        _event("child_waited", pid=proc.pid, stop_reason="telemetry_unavailable", **stop)
        return 2, "child start telemetry unavailable; stopping"

    deadline = started + timeout
    peak_rss = 0
    min_available = available
    samples = 0
    try:
        while proc.poll() is None:
            try:
                available = _reader_call("available_memory_reader", "monitor",
                                         available_memory_reader)
                rss = _reader_call("rss_reader", "monitor", rss_reader, proc.pid)
            except Exception as error:
                stop = _stop_process(proc, interrupt_grace, clock, sleep,
                                     manage_process_group=manage_process_group)
                _event("child_waited", pid=proc.pid, duration_seconds=clock() - started,
                       samples=samples, peak_rss_bytes=peak_rss or None,
                       min_available_bytes=min_available, stop_reason="measurement_failed", **stop)
                _event("monitor_stopped", reason="resource measurement unavailable",
                       exception=type(error).__name__)
                return 2, f"resource measurement unavailable ({type(error).__name__}); stopping"
            if _EVENT_LOG_FAILED:
                stop = _stop_process(proc, interrupt_grace, clock, sleep,
                                     manage_process_group=manage_process_group)
                _event("child_waited", pid=proc.pid, duration_seconds=clock() - started,
                       samples=samples, peak_rss_bytes=peak_rss or None,
                       min_available_bytes=min_available, stop_reason="telemetry_failed", **stop)
                return 2, "resource telemetry unavailable; stopping"
            samples += 1
            peak_rss = max(peak_rss, rss)
            min_available = min(min_available, available)
            if rss >= max_rss:
                stop = _stop_process(proc, interrupt_grace, clock, sleep,
                                     manage_process_group=manage_process_group)
                _event("child_waited", pid=proc.pid, duration_seconds=clock() - started,
                       samples=samples, peak_rss_bytes=peak_rss, min_available_bytes=min_available,
                       stop_reason="rss_limit", **stop)
                return 2, f"RSS limit reached ({rss} bytes); stopping"
            if available < stop_min_available:
                stop = _stop_process(proc, interrupt_grace, clock, sleep,
                                     manage_process_group=manage_process_group)
                _event("child_waited", pid=proc.pid, duration_seconds=clock() - started,
                       samples=samples, peak_rss_bytes=peak_rss, min_available_bytes=min_available,
                       stop_reason="available_memory_limit", **stop)
                return 2, f"available memory below stop threshold ({available} bytes); stopping"
            if clock() >= deadline:
                stop = _stop_process(proc, interrupt_grace, clock, sleep,
                                     manage_process_group=manage_process_group)
                _event("child_waited", pid=proc.pid, duration_seconds=clock() - started,
                       samples=samples, peak_rss_bytes=peak_rss, min_available_bytes=min_available,
                       stop_reason="wall_clock_limit", **stop)
                return 124, f"wall-clock limit reached ({timeout:g}s); stopping"
            sleep(poll_interval)
    except BaseException:
        if proc.poll() is None or (manage_process_group and _process_group_exists(proc.pid)):
            _stop_process(proc, interrupt_grace, clock, sleep,
                          manage_process_group=manage_process_group)
        else:
            proc.wait()
        raise

    returncode = proc.wait()
    descendant_cleanup = (
        _stop_remaining_group(proc.pid, interrupt_grace, time.monotonic, time.sleep)
        if manage_process_group else
        {"descendants_found": None, "signal": None, "group_gone": None,
         "reason": "custom process factory; process group not managed"}
    )
    first_fixture_timeout = (
        returncode == -getattr(signal, "SIGALRM", 14)
        and len(command) > 1
        and command[1] == str(PROJECT / "scripts" / "run_masked_prediction_pilot.py")
    )
    _event("child_waited", pid=proc.pid, duration_seconds=clock() - started,
           samples=samples, peak_rss_bytes=peak_rss or None,
           min_available_bytes=min_available,
           stop_reason=("first_fixture_wall_clock_limit" if first_fixture_timeout else "child_exit"),
           signal=("SIGALRM" if first_fixture_timeout else None),
           requested_signal=None, escalation=None, returncode=returncode, waited=True,
           descendant_cleanup=descendant_cleanup)
    if descendant_cleanup.get("group_gone") is False:
        return 2, "child exited but its process group did not terminate cleanly"
    if first_fixture_timeout:
        return 124, "first fixture exceeded the fixed 180-second wall-clock limit"
    return int(returncode or 0), "child exited"


def _dummy_probe_source(sleep_seconds: float) -> str:
    return f"""import errno, socket, sys, time
print("DUMMY_CHILD_STARTED", flush=True)
s = socket.socket()
s.settimeout(2)
try:
    s.connect(("127.0.0.1", 1))
    print("DUMMY_NETWORK_UNEXPECTEDLY_CONNECTED", flush=True)
    sys.exit(3)
except OSError as error:
    name = errno.errorcode.get(error.errno, "UNKNOWN")
    print("DUMMY_NETWORK_ERRNO", error.errno, name, flush=True)
    if error.errno not in (errno.EPERM, errno.EACCES):
        sys.exit(4)
finally:
    s.close()
time.sleep({sleep_seconds!r})
print("DUMMY_CHILD_FINISHED", flush=True)
"""


def build_native_parity_command(*, python: str, checkpoint: Path, identity_record: Path,
                                report: Path, device: str, timeout: int) -> list[str]:
    """Select one immutable parity child command; never accept an arbitrary script."""
    if python != PARITY_PYTHON:
        raise ValueError("parity must use the guardian's current Python interpreter")
    if not Path(checkpoint).is_absolute() or not Path(identity_record).is_absolute():
        raise ValueError("checkpoint and identity paths must be explicit absolute paths")
    if str(report) != PARITY_REPORT:
        raise ValueError("parity report path is outside the registered fixed output")
    if device != "cpu" or timeout != MAX_WALL_SECONDS:
        raise ValueError("parity mode requires CPU and the fixed 600-second limit")
    return [
        python, str(PROJECT / "scripts" / "check_native_predictor_parity.py"),
        "--checkpoint", str(checkpoint),
        "--identity-record", str(identity_record),
        "--report", PARITY_REPORT,
    ]


def build_pilot_command(*, python: str, checkpoint: Path, identity_record: Path,
                        processor_config: Path, compatibility_loader: Path,
                        count: int, device: str, timeout: int,
                        verified_loss_exp: float) -> list[str]:
    """Return the fixed pilot argv; callers cannot select arbitrary scripts or outputs."""
    if python != PARITY_PYTHON:
        raise ValueError("pilot must use the guardian's current Python interpreter")
    if not all(Path(value).is_absolute() for value in
               (checkpoint, identity_record, processor_config, compatibility_loader)):
        raise ValueError("pilot resources must be explicit absolute paths")
    if type(count) is not int or not 4 <= count <= 8:
        raise ValueError("pilot count must be between four and eight fixed seeds")
    if device != "cpu" or timeout != MAX_WALL_SECONDS or verified_loss_exp != 1.0:
        raise ValueError("pilot requires CPU, loss_exp=1.0, and the fixed 600-second limit")
    return [
        python, str(PROJECT / "scripts" / "run_masked_prediction_pilot.py"),
        "--execute", "--count", str(count),
        "--checkpoint", str(checkpoint),
        "--identity-record", str(identity_record),
        "--processor-config", str(processor_config),
        "--compatibility-loader", str(compatibility_loader),
        "--verified-loss-exp", "1.0",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--identity-record", type=Path)
    parser.add_argument("--processor-config", type=Path)
    parser.add_argument("--compatibility-loader", type=Path)
    parser.add_argument("--verified-loss-exp", type=float)
    parser.add_argument("--fixture", type=Path, default=PROJECT / "fixtures" / "ball-gap-v2")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--timeout", type=int, default=MAX_WALL_SECONDS)
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument("--dummy-network-probe", action="store_true",
                                help="run a model-free denied-network child and intentionally test timeout cleanup")
    execution_mode.add_argument("--native-parity", action="store_true")
    execution_mode.add_argument("--pilot", action="store_true",
                                help="run the registered fixed synthetic pilot under the resource guardian")
    parser.add_argument("--pilot-count", type=int, default=4)
    parser.add_argument("--dummy-sleep-seconds", type=float, default=30.0)
    parser.add_argument("--dummy-wall-seconds", type=float, default=1.0)
    args = parser.parse_args()

    if args.dummy_network_probe:
        if args.python != PARITY_PYTHON:
            parser.error("dummy probe must use the guardian's current Python interpreter")
        if args.dummy_sleep_seconds <= args.dummy_wall_seconds or args.dummy_wall_seconds <= 0:
            parser.error("dummy sleep must exceed a positive dummy wall-clock timeout")
        command = [args.python, "-u", "-c", _dummy_probe_source(args.dummy_sleep_seconds)]
        code, message = run_limited_process(command, timeout=args.dummy_wall_seconds,
                                            network_deny=True)
        print(message, file=sys.stderr, flush=True)
        return code

    if args.native_parity:
        if args.checkpoint is None or args.identity_record is None or args.report is None:
            parser.error("native parity requires fixed checkpoint, identity, and report paths")
        try:
            child = build_native_parity_command(
                python=args.python, checkpoint=args.checkpoint,
                identity_record=args.identity_record, report=args.report,
                device=args.device, timeout=args.timeout,
            )
        except ValueError as error:
            parser.error(str(error))
        code, message = run_limited_process(
            child, timeout=MAX_WALL_SECONDS, start_min_available=START_MIN_AVAILABLE,
            stop_min_available=STOP_MIN_AVAILABLE, max_rss=MAX_RSS,
            network_deny=True,
        )
        print(message, file=sys.stderr, flush=True)
        return code

    if args.pilot:
        if args.report is not None or args.fixture != PROJECT / "fixtures" / "ball-gap-v2":
            parser.error("pilot output and fixture paths are fixed inside the project")
        if (args.checkpoint is None or args.identity_record is None
                or args.processor_config is None or args.compatibility_loader is None):
            parser.error("pilot requires checkpoint, identity, processor config, and compatibility loader paths")
        if args.verified_loss_exp is None:
            parser.error("pilot requires --verified-loss-exp from the pinned official config")
        try:
            child = build_pilot_command(
                python=args.python, count=args.pilot_count, device=args.device,
                timeout=args.timeout, verified_loss_exp=args.verified_loss_exp,
                checkpoint=args.checkpoint, identity_record=args.identity_record,
                processor_config=args.processor_config,
                compatibility_loader=args.compatibility_loader,
            )
        except ValueError as error:
            parser.error(str(error))
        code, message = run_limited_process(
            child, timeout=MAX_WALL_SECONDS, start_min_available=START_MIN_AVAILABLE,
            stop_min_available=STOP_MIN_AVAILABLE, max_rss=MAX_RSS,
            network_deny=True,
        )
        print(message, file=sys.stderr, flush=True)
        return code

    if args.checkpoint is None or args.identity_record is None or args.verified_loss_exp is None:
        parser.error("model run requires --checkpoint, --identity-record, and --verified-loss-exp")
    if args.python != PARITY_PYTHON:
        parser.error("model runs must use the guardian's current Python interpreter")
    if not args.fixture.is_dir():
        parser.error("fixture directory not found")
    if args.processor_config is None or args.compatibility_loader is None:
        parser.error("model run requires --processor-config and --compatibility-loader")
    child = [args.python, str(PROJECT / "scripts" / "run_masked_prediction_smoke.py"),
             "--execute", "--checkpoint", str(args.checkpoint),
             "--identity-record", str(args.identity_record), "--fixture", str(args.fixture),
             "--device", args.device, "--verified-loss-exp", str(args.verified_loss_exp),
             "--processor-config", str(args.processor_config),
             "--compatibility-loader", str(args.compatibility_loader)]
    code, message = run_limited_process(child, network_deny=True)
    print(message, file=sys.stderr, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
