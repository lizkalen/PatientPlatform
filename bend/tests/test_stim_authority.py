"""Behavioural tests for the stimulation authority's state machine.

Layer under test: `server.stim_authority.StimAuthority` - the backend component
that decides whether electrical stimulation may start, who owns a running train,
and every path that must stop one (client command, commanding-client disconnect,
last-client disconnect, acquisition-device loss, dead-man watchdog, process
teardown), plus the retry loop that refuses to report a train as stopped until
the hardware confirms it.

These tests exist because this layer stops electrical stimulation on a human.
Before it existed, stimulation was entirely browser-owned: every train was
started with an infinite duration and the only off-switch in the system was a
WebSocket message from a browser tab (architecture audit S1-S8). A regression
here is not a broken feature, it is a train that keeps running.

No hardware, no network, no event loop of its own: the HTTP client is replaced
by a scriptable fake and the WebSocket clients by fake sockets, so the whole
suite runs on a bare Python >=3.10 with no third-party packages installed.

Run:  py -3 bend/tests/test_stim_authority.py
      (or `pytest bend/tests/test_stim_authority.py`)
"""

import asyncio
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _stim_stubs import (                                       # noqa: E402
    FakeClient, Recorder, Sock, arun, channel, import_authority, ok,
    run_suite, start_request,
)

sa = import_authority()
StimAuthority = sa.StimAuthority
StimParameterError = sa.StimParameterError
validate_start_request = sa.validate_start_request


def _authority(client=None, recorder=None, **kwargs):
    """An authority wired to a fake client and a broadcast recorder."""
    client = client if client is not None else FakeClient()
    recorder = recorder if recorder is not None else Recorder()
    kwargs.setdefault("max_train_seconds", 30)
    return StimAuthority(client=client, broadcast=recorder.publish, **kwargs), \
        client, recorder


def _rejects(payload, label):
    try:
        validate_start_request(payload)
        ok(False, f"rejects {label}")
    except StimParameterError:
        ok(True, f"rejects {label}")


# ── parameter validation: reject, never clamp ────────────────────────────────

def test_parameter_validation_rejects_out_of_range():
    """Every ceiling from StimLimits, plus the shapes a malformed payload takes.

    Rejection, never clamping: silently delivering a different current than the
    operator asked for is its own hazard.
    """
    validate_start_request(start_request())
    ok(True, "a schema-valid request passes")

    for label, payload in [
        ("amplitude 131 mA (over the 130 mA ceiling)", start_request(amplitude=131)),
        ("amplitude -1 mA", start_request(amplitude=-1)),
        ("amplitude NaN", start_request(amplitude=float("nan"))),
        ("amplitude +inf", start_request(amplitude=float("inf"))),
        ("amplitude True (bool is not a number)", start_request(amplitude=True)),
        ("amplitude '8' (string)", start_request(amplitude="8")),
        ("pulse_width 5 us (under the 10 us floor)", start_request(pulse_width=5)),
        ("pulse_width 1001 us", start_request(pulse_width=1001)),
        ("frequency 0 Hz (under the 0.1 Hz floor)", start_request(frequency=0)),
        ("frequency 2001 Hz", start_request(frequency=2001)),
        ("channel id 0", start_request(id=0)),
        ("channel id 9", start_request(id=9)),
        ("channel id 1.5 (not a whole number)", start_request(id=1.5)),
        ("is_biphasic 'yes' (not a bool)", start_request(is_biphasic="yes")),
    ]:
        _rejects(payload, label)

    for label, payload in [
        ("an empty channel list", {"channels": []}),
        ("a missing channel list", {}),
        ("channels that is not a list", {"channels": channel()}),
        ("a channel that is not an object", {"channels": ["1"]}),
        ("a channel missing 'amplitude'",
         {"channels": [{"id": 1, "pulse_width": 200, "frequency": 50}]}),
        ("a duplicated channel id", {"channels": [channel(id=1), channel(id=1)]}),
        ("9 channels (over the 8-channel cap)",
         {"channels": [channel(id=i) for i in range(1, 10)]}),
        ("stimulator_type that is not a string",
         dict(start_request(), stimulator_type=3)),
        ("metadata that is not an object", dict(start_request(), metadata="x")),
    ]:
        _rejects(payload, label)


def test_rejected_start_never_touches_hardware():
    """An invalid request must not reach the stimulator, and must be answered."""
    async def scenario():
        authority, client, recorder = _authority()
        sock = Sock("requester")

        message = await authority.handle_start(start_request(amplitude=999), sock)

        ok(not client.starts and client.stops == 0,
           "a rejected start issues no HTTP call at all")
        ok(message["status"] == "error" and message["reason"] == "rejected",
           "the reply carries status=error, reason=rejected")
        ok(message["active"] is False,
           "active reports the unchanged belief, not the attempt")
        ok(len(sock.sent) == 1 and len(recorder) == 0,
           "the rejection is replied to the requester, not broadcast")

    arun(scenario)


def test_client_controller_url_is_ignored():
    """Audit S7: a client could otherwise redirect stimulation to any host."""
    async def scenario():
        authority, client, recorder = _authority()
        sock = Sock("requester")

        await authority.handle_start(
            dict(start_request(), controller_url="http://evil.example:9"), sock)

        ok(client.starts[-1]["controller_url"] is None,
           "the client-supplied controller_url is dropped before the HTTP call")
        ok(authority.active is True, "the train is believed active after a good start")
        ok(recorder.last == {"type": "stimulation_status", "active": True,
                             "status": "ok", "reason": "requested"},
           "the start broadcast has the documented shape")

        await authority.handle_stop(sock)

    arun(scenario)


# ── the dead-man watchdog ────────────────────────────────────────────────────

def test_watchdog_does_not_fire_after_requested_stop():
    """The deadline must be disarmed by a normal stop, or it stops the NEXT train."""
    async def scenario():
        authority, client, _ = _authority(max_train_seconds=0.3)
        sock = Sock("requester")
        await authority.handle_start(start_request(), sock)

        message = await authority.handle_stop(sock)
        ok(message["active"] is False and message["status"] == "ok"
           and message["reason"] == "requested", "the stop reply has the right shape")

        stops_before = client.stops
        await asyncio.sleep(0.5)                    # well past the 0.3 s deadline
        ok(client.stops == stops_before,
           "no watchdog stop is issued after a requested stop")
        ok(authority._watchdog_task is None, "the watchdog handle is cleared")

    arun(scenario)


def test_watchdog_fires_on_stranded_train():
    """Nothing stopped the train, so the deadline must."""
    async def scenario():
        authority, client, recorder = _authority(max_train_seconds=0.3)
        await authority.handle_start(start_request(), Sock("gone"))

        stops_before = client.stops
        await asyncio.sleep(0.5)

        ok(client.stops == stops_before + 1, "the watchdog issued the stop")
        ok(authority.active is False, "the train is no longer believed active")
        ok(recorder.last["reason"] == "watchdog"
           and recorder.last["active"] is False, "reason=watchdog is broadcast")

    arun(scenario)


# ── a stop that fails at the hardware ────────────────────────────────────────

def test_failed_stop_retries_until_it_lands():
    """Audit S6: a failed stop used to be broadcast as "stimulation off"."""
    async def scenario():
        authority, client, recorder = _authority(
            stop_retry_interval_s=0.05, stop_retry_limit=10)
        sock = Sock("requester")
        await authority.handle_start(start_request(), sock)

        client.stop_ok = False
        message = await authority.handle_stop(sock)
        ok(message["active"] is True and message["status"] == "error",
           "a failed stop reports active=True, status=error")
        ok(authority.active is True,
           "the authority still believes a train is running")

        client.stop_ok = True
        await asyncio.sleep(0.2)
        ok(authority.active is False, "the retry loop landed the stop")
        ok(recorder.last["active"] is False and recorder.last["status"] == "ok",
           "success is re-broadcast once it lands")

    arun(scenario)


def test_retry_cap_goes_loud_instead_of_quiet():
    """An unreachable controller must end in an alarm, not silence."""
    async def scenario():
        authority, client, recorder = _authority(
            stop_retry_interval_s=0.02, stop_retry_limit=3)
        sock = Sock("requester")
        await authority.handle_start(start_request(), sock)

        client.stop_ok = False
        await authority.handle_stop(sock)
        await asyncio.sleep(0.3)

        ok(authority.active is True,
           "the train is still believed active after the retries are exhausted")
        ok("power it down" in (recorder.last.get("message") or ""),
           "the final broadcast tells the operator to power the stimulator down")

        client.stop_ok = True
        await authority.handle_stop(sock)

    arun(scenario)


def test_new_start_cancels_pending_stop_retry():
    """A retry belongs to the old train; left running it would kill the new one."""
    async def scenario():
        authority, client, _ = _authority(
            stop_retry_interval_s=0.05, stop_retry_limit=10)
        sock = Sock("requester")
        await authority.handle_start(start_request(), sock)

        client.stop_ok = False
        await authority.handle_stop(sock)           # arms the retry loop
        client.stop_ok = True
        await authority.handle_start(start_request(), sock)   # a NEW train

        stops_before = client.stops
        await asyncio.sleep(0.2)
        ok(client.stops == stops_before,
           "the stale retry did not stop the newly started train")
        ok(authority.active is True, "the new train is still believed active")

        await authority.handle_stop(sock)

    arun(scenario)


# ── ownership and disconnects ────────────────────────────────────────────────

def test_stop_on_owner_disconnect():
    """A train is bound to the socket that started it (audit S2/S8.2)."""
    async def scenario():
        authority, client, _ = _authority()
        owner, other = Sock("owner"), Sock("other")
        await authority.handle_start(start_request(), owner)

        stops_before = client.stops
        await authority.on_client_disconnect(other, remaining_clients=1)
        ok(client.stops == stops_before and authority.active,
           "a non-owner leaving while others remain does not stop the train")

        await authority.on_client_disconnect(owner, remaining_clients=1)
        ok(client.stops == stops_before + 1 and not authority.active,
           "the commanding client leaving stops the train")

    arun(scenario)


def test_stop_when_last_client_disconnects():
    """With nobody connected, no `stimulate_stop` can ever arrive."""
    async def scenario():
        authority, client, _ = _authority()
        owner, other = Sock("owner"), Sock("other")
        await authority.handle_start(start_request(), owner)

        stops_before = client.stops
        await authority.on_client_disconnect(other, remaining_clients=0)
        ok(client.stops == stops_before + 1 and not authority.active,
           "the last client leaving stops the train regardless of ownership")

    arun(scenario)


def test_stop_on_device_loss():
    """Audit S3: no device means no decisions, so the closed loop never closes."""
    async def scenario():
        authority, _, recorder = _authority()
        await authority.handle_start(start_request(), Sock("owner"))

        await authority.on_device_lost()
        ok(not authority.active and recorder.last["reason"] == "device_lost",
           "losing the acquisition device stops the train")
        ok(await authority.on_device_lost() is None,
           "a second device-loss while idle is a no-op")

    arun(scenario)


def test_shutdown_stops_unconditionally():
    """The controller outlives this process; the last word must be "stop"."""
    async def scenario():
        authority, client, _ = _authority()
        await authority.handle_start(start_request(), Sock("owner"))

        message = await authority.shutdown()
        ok(message["reason"] == "shutdown" and message["active"] is False,
           "teardown stops the train and says so")

        stops_before = client.stops
        await authority.shutdown()
        ok(client.stops == stops_before + 1,
           "shutdown sends the DELETE even when nothing is believed active")

    arun(scenario)


# ── a start that fails at the hardware ───────────────────────────────────────

def test_failed_start_sends_defensive_stop():
    """A failed PUT does not prove the controller armed nothing."""
    async def scenario():
        authority, client, _ = _authority()
        sock = Sock("requester")
        client.start_ok = False

        stops_before = client.stops
        message = await authority.handle_start(start_request(), sock)

        ok(message["active"] is False and message["status"] == "error"
           and message["reason"] == "requested",
           "a failed start reports status=error, active=False")
        ok(client.stops == stops_before + 1, "a defensive stop is fired")
        ok(authority._watchdog_task is None,
           "no watchdog is armed by a start that never succeeded")

    arun(scenario)


def test_failed_start_leaves_a_running_train_alone():
    """Stopping here would silently end therapy the operator still wants."""
    async def scenario():
        authority, client, _ = _authority()
        sock = Sock("requester")
        await authority.handle_start(start_request(), sock)

        client.start_ok = False
        stops_before = client.stops
        message = await authority.handle_start(start_request(), sock)

        ok(client.stops == stops_before,
           "no defensive stop is fired while a train is already active")
        ok(message["active"] is True and message["status"] == "error",
           "the running train is still reported as active")

        client.start_ok = True
        await authority.handle_stop(sock)

    arun(scenario)


TESTS = [
    test_parameter_validation_rejects_out_of_range,
    test_rejected_start_never_touches_hardware,
    test_client_controller_url_is_ignored,
    test_watchdog_does_not_fire_after_requested_stop,
    test_watchdog_fires_on_stranded_train,
    test_failed_stop_retries_until_it_lands,
    test_retry_cap_goes_loud_instead_of_quiet,
    test_new_start_cancels_pending_stop_retry,
    test_stop_on_owner_disconnect,
    test_stop_when_last_client_disconnects,
    test_stop_on_device_loss,
    test_shutdown_stops_unconditionally,
    test_failed_start_sends_defensive_stop,
    test_failed_start_leaves_a_running_train_alone,
]


if __name__ == "__main__":
    sys.exit(run_suite("stimulation authority: core state machine", TESTS))
