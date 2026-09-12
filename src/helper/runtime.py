import faulthandler
from contextlib import contextmanager
from datetime import datetime, timezone
from time import monotonic


@contextmanager
def stage(name: str, timeout_seconds: float):
    """Bound a sequential pipeline stage, including calls blocked in native code.

    The process exits on timeout; completed outputs must be saved beforehand.
    Do not nest stages: faulthandler has one process-wide watchdog timer.
    """
    if timeout_seconds <= 0:
        raise ValueError("Stage timeout must be positive")
    started = monotonic()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] START {name} (timeout={timeout_seconds}s)", flush=True)
    faulthandler.dump_traceback_later(timeout_seconds, exit=True)
    outcome = "FAILED"
    try:
        yield
        outcome = "END"
    finally:
        faulthandler.cancel_dump_traceback_later()
        print(f"{outcome} {name} ({monotonic() - started:.1f}s)", flush=True)
