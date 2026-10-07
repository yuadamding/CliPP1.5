"""Time-throttled completed-work telemetry, separate from worker liveness."""

from contextlib import contextmanager
import time


class NumericalProgress:
    """Share one throttle and cumulative counters across nested selection work."""

    def __init__(self, callback=None, interval_seconds=30.0):
        self.callback = callback
        self.interval_seconds = interval_seconds
        self.started = time.perf_counter()
        self.last_emitted = -float("inf")
        self.counters = {}
        self.timings = {}
        self.fields = {}
        self._active_timers = []

    @contextmanager
    def context(self, **fields):
        previous = self.fields
        self.fields = {**previous, **fields}
        try:
            yield
        finally:
            self.fields = previous

    def add(self, **counts):
        for name, value in counts.items():
            self.counters[name] = self.counters.get(name, 0) + value

    @contextmanager
    def timer(self, name):
        start = time.perf_counter()
        active = (name + "_seconds", start)
        self._active_timers.append(active)
        try:
            yield
        finally:
            self._active_timers.remove(active)
            key = active[0]
            self.timings[key] = self.timings.get(key, 0.0) + time.perf_counter() - start

    def snapshot(self):
        now = time.perf_counter()
        timings = self.timings.copy()
        for key, start in self._active_timers:
            timings[key] = timings.get(key, 0.0) + now - start
        return {**self.counters, **timings, "elapsed_seconds": now - self.started}

    def update(self, event, *, force=False, **fields):
        now = time.perf_counter()
        if self.callback is not None and (force or now - self.last_emitted >= self.interval_seconds):
            self.callback(
                {"stage": "numerical_progress", **self.fields, **self.snapshot(), "event": event, **fields}
            )
            self.last_emitted = now
