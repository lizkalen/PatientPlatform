"""Shared bootstrap for the stimulation-authority tests.

Not a test module (leading underscore, so pytest does not collect it). It
provides three things:

  1. `ensure_src_on_path()` - import `server.*` from a checkout with no editable
     install, mirroring what `bend/src/mvdecoder/tests/test_parity.py` does.
  2. `import_authority()` / `import_servers()` - import the modules under test
     with their heavy dependencies stubbed, so the suite runs on a bare
     Python >=3.10 with no numpy / pylsl / websockets / httpx / scipy present.
     Stubs are installed only for the duration of the import and then removed
     again, so running these tests inside a larger pytest session does not leave
     a fake `numpy` in `sys.modules` for everyone else.
  3. The fakes and the tiny assertion reporter both suites share.

The stubs are deliberately full-fidelity for what the code under test touches
(class objects where annotations are evaluated at import time, a real
`ConnectionClosed` exception class, managers with the methods the servers call).
They are a substitute for the packages, not a simplification of the tests.
"""

import asyncio
import os
import sys
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")          # bend/src

_MISSING = object()

# Every labelled assertion that has passed, for the standalone summary line.
CHECKS = []


# ── path + import plumbing ───────────────────────────────────────────────────

def ensure_src_on_path():
    """Make `server.*` importable. Normal import first, then bend/src."""
    try:
        import server  # noqa: F401
    except ImportError:
        sys.path.insert(0, SRC)       # no editable install: import from source


class _StubbedModules:
    """Install stub modules for the duration of a block, then restore.

    Restoring matters: these tests may run inside a session with other tests
    that need the real numpy/websockets. Modules imported *inside* the block
    keep working afterwards because they captured the stub objects as their own
    globals at import time.
    """

    def __init__(self, modules):
        self._modules = modules
        self._saved = {}

    def __enter__(self):
        for name, mod in self._modules.items():
            self._saved[name] = sys.modules.get(name, _MISSING)
            sys.modules[name] = mod
        return self

    def __exit__(self, *exc):
        for name, old in self._saved.items():
            if old is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        return False


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


class _StubArray:
    """Stands in for `np.ndarray` where annotations are evaluated at import."""


class _StubOutlet:
    """Stands in for `pylsl.StreamOutlet` in an evaluated annotation."""


class _StubProtocol:
    """Stands in for `websockets.WebSocketServerProtocol` in an annotation."""


class StubConnectionClosed(Exception):
    """Mirrors `websockets.exceptions.ConnectionClosed`.

    Crucially a plain `Exception` subclass, exactly like the real one - that is
    the whole reason `handle_message` has to re-raise it ahead of its catch-all.
    """


class _StubManager:
    """Stands in for DecompositionManager / MovementClassifierManager."""

    active = False
    model_path = None

    def get_status(self):
        return {}

    def stop_classification(self):
        pass


def _http_stubs():
    class HTTPError(Exception):
        pass

    class AsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def put(self, *a, **k):
            raise HTTPError("stubbed httpx: no real request is ever made")

        async def delete(self, *a, **k):
            raise HTTPError("stubbed httpx: no real request is ever made")

    return {"httpx": _module("httpx", HTTPError=HTTPError, AsyncClient=AsyncClient)}


def _server_stubs():
    exceptions = _module("websockets.exceptions",
                         ConnectionClosed=StubConnectionClosed)
    server_mod = _module("websockets.server", serve=lambda *a, **k: None)
    websockets = _module("websockets", WebSocketServerProtocol=_StubProtocol,
                         exceptions=exceptions, server=server_mod)
    return {
        "numpy": _module(
            "numpy", ndarray=_StubArray, float32="float32",
            hstack=lambda *a, **k: None, array=lambda *a, **k: None,
            asarray=lambda *a, **k: None, ascontiguousarray=lambda *a, **k: None),
        "pylsl": _module("pylsl", StreamOutlet=_StubOutlet, StreamInfo=object,
                         local_clock=lambda: 0.0),
        "websockets": websockets,
        "websockets.exceptions": exceptions,
        "websockets.server": server_mod,
        "dsp": _module("dsp"),
        "dsp.processing": _module(
            "dsp.processing",
            design_filters=lambda *a, **k: (None, None),
            init_filter_states=lambda *a, **k: (None, None),
            apply_filters=lambda *a, **k: (None, None, None)),
        "server.decomposition_manager": _module(
            "server.decomposition_manager", DecompositionManager=_StubManager),
        "server.movement_classifier_manager": _module(
            "server.movement_classifier_manager",
            MovementClassifierManager=_StubManager),
        "server.movement_training": _module(
            "server.movement_training", MovementTrainer=object,
            trigger_bouts=lambda *a, **k: [],
            estimate_trig_mid=lambda *a, **k: 0),
        "server.devices": _module("server.devices"),
        "server.devices.simulated": _module("server.devices.simulated",
                                            SimulatedDevice=object),
    }


def import_authority():
    """Import `server.stim_authority` (needs only the httpx stub)."""
    ensure_src_on_path()
    with _StubbedModules(_http_stubs()):
        import server.stim_authority as mod
    return mod


def import_servers():
    """Import both WebSocket server modules with the full stub set."""
    ensure_src_on_path()
    stubs = _http_stubs()
    stubs.update(_server_stubs())
    with _StubbedModules(stubs):
        from server.websocket_server import RippleWebSocketServer
        from server.simulated_websocket_server import SimulatedWebSocketServer
    return RippleWebSocketServer, SimulatedWebSocketServer


# ── fakes ────────────────────────────────────────────────────────────────────

class Sock:
    """A fake WebSocket: records what was sent, or raises on demand."""

    def __init__(self, name="sock"):
        self.name = name
        self.sent = []
        self.raise_on_send = None

    async def send(self, payload):
        if self.raise_on_send is not None:
            raise self.raise_on_send
        self.sent.append(payload)

    def __repr__(self):
        return f"<Sock {self.name}>"


class FakeClient:
    """Stands in for StimulationClient: no HTTP, scriptable outcomes.

    `start_gate` / `stop_gate` are asyncio.Events used to park a call *inside*
    the HTTP round trip while it holds the authority's lock. That is what makes
    the concurrency tests deterministic instead of timing-dependent.
    """

    def __init__(self):
        self.base_url = "http://127.0.0.1:11051"
        self.start_ok = True
        self.stop_ok = True
        self.starts = []            # one dict per start_train call
        self.stops = 0
        self.start_gate = None
        self.stop_gate = None

    async def start_train(self, channels, stimulator_type=None, port=None,
                          controller_url=None, metadata=None):
        if self.start_gate is not None:
            await self.start_gate.wait()
        self.starts.append({"channels": channels, "controller_url": controller_url,
                            "stimulator_type": stimulator_type, "port": port,
                            "metadata": metadata})
        return ({"status": "success", "response": {}} if self.start_ok
                else {"status": "error", "message": "start-boom"})

    async def stop(self, controller_url=None):
        if self.stop_gate is not None:
            await self.stop_gate.wait()
        self.stops += 1
        return ({"status": "success", "response": {}} if self.stop_ok
                else {"status": "error", "message": "stop-boom"})


def channel(**overrides):
    """A schema-valid stimulation channel, with fields overridden."""
    ch = {"id": 1, "amplitude": 8.0, "pulse_width": 200, "frequency": 50.0,
          "is_biphasic": True}
    ch.update(overrides)
    return ch


def start_request(**channel_overrides):
    """A valid `stimulate_start` payload with one channel."""
    return {"command": "stimulate_start", "channels": [channel(**channel_overrides)]}


class Recorder:
    """Collects broadcast messages; pass `.publish` as the authority's broadcast."""

    def __init__(self):
        self.messages = []

    async def publish(self, message):
        self.messages.append(message)

    def clear(self):
        self.messages.clear()

    @property
    def last(self):
        return self.messages[-1]

    def __len__(self):
        return len(self.messages)


# ── assertion reporting ──────────────────────────────────────────────────────

def ok(condition, label):
    """One labelled check.

    A plain `assert`, so pytest fails the test normally and reports `label`.
    The standalone runner catches it per test function and keeps going, so one
    broken scenario does not hide the rest.
    """
    assert condition, label
    print(f"  [ok] {label}")
    CHECKS.append(label)


def arun(scenario):
    """Run one async scenario in its own event loop.

    A fresh loop per test keeps the authority's watchdog/retry tasks from
    leaking between tests, and the trailing check enforces that each scenario
    cleans up after itself - an authority that exits a test with a live
    watchdog is a bug in the code under test, not just untidy.
    """
    async def wrapper():
        await scenario()
        await asyncio.sleep(0.05)
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        ok(not pending, f"scenario left no stray asyncio tasks ({len(pending)})")

    asyncio.run(wrapper())


def run_suite(title, tests):
    """Standalone runner. Returns a process exit code."""
    print(f"=== {title} ===")
    failures = []
    for test in tests:
        print(f"\n-- {test.__name__}")
        try:
            test()
        except AssertionError as exc:
            print(f"  [FAIL] {exc}")
            failures.append((test.__name__, str(exc) or "assertion failed"))
        except Exception as exc:                                  # noqa: BLE001
            traceback.print_exc()
            failures.append((test.__name__, f"{type(exc).__name__}: {exc}"))

    print(f"\n{len(CHECKS)} assertions passed across {len(tests)} test functions; "
          f"{len(failures)} failed")
    if failures:
        for name, message in failures:
            print(f"  FAILED {name}: {message}")
        return 1
    print("OK")
    return 0
