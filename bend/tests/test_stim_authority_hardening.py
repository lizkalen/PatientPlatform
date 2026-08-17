"""Regression tests for the stimulation authority's hardening fixes.

Companion to `test_stim_authority.py`, which covers the baseline contract. This
file pins the defects found reviewing that contract - every test here is a bug
that shipped, was caught, and must not come back:

  * a start that fails AND whose defensive stop also fails must not be recorded
    as "off": that would be a possibly-running train with no watchdog and no
    retry, which is the exact silent failure the authority exists to prevent;
  * the stop-on-disconnect decision must be taken under the lock, or a departing
    owner can kill a train another client started microseconds earlier;
  * two stop paths racing (watchdog vs manual, device-loss vs disconnect) must
    publish exactly one status, so the true reason is not overwritten;
  * the stop-success bookkeeping must live in one place, so the retry loop and
    the command path cannot drift apart;
  * the remembered rejected-controller_url set must be bounded;
  * `handle_message` must re-raise `ConnectionClosed` ahead of its catch-all -
    it is a plain `Exception`, so swallowing it would log a dropped socket as an
    alarming traceback AND bypass the disconnect path (which now stops stim).

These tests exist because this layer stops electrical stimulation on a human.

Concurrency is made deterministic, not timed: the fake HTTP client can park a
call inside the request while it holds the authority's lock, so the interleaving
under test happens on every run. The ownership-race test carries a negative
control that runs the same interleaving against a replica of the pre-fix logic
and asserts it *does* misbehave - a race test that cannot fail is not a test.

Run:  py -3 bend/tests/test_stim_authority_hardening.py
      (or `pytest bend/tests/test_stim_authority_hardening.py`)
"""

import asyncio
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _stim_stubs import (                                       # noqa: E402
    FakeClient, Recorder, Sock, StubConnectionClosed, arun, import_authority,
    import_servers, ok, run_suite, start_request,
)

sa = import_authority()
StimAuthority = sa.StimAuthority
MAX_REMEMBERED_REJECTED_URLS = sa.MAX_REMEMBERED_REJECTED_URLS


def _authority(client=None, recorder=None, **kwargs):
    client = client if client is not None else FakeClient()
    recorder = recorder if recorder is not None else Recorder()
    kwargs.setdefault("max_train_seconds", 30)
    return StimAuthority(client=client, broadcast=recorder.publish, **kwargs), \
        client, recorder


# ── a start whose defensive stop also fails ──────────────────────────────────

def test_start_failure_with_failed_defensive_stop_is_treated_as_running():
    """Both calls failed, so we know nothing. Assume the worse of the two."""
    async def scenario():
        authority, client, recorder = _authority(
            stop_retry_interval_s=0.05, stop_retry_limit=10)
        sock = Sock("requester")
        client.start_ok = False
        client.stop_ok = False

        message = await authority.handle_start(start_request(), sock)

        ok(authority.active is True,
           "active stays True when both the PUT and the DELETE fail")
        ok(message["active"] is True and message["status"] == "error",
           "the broadcast says active=True/status=error, not the S6 'off' lie")
        ok(message["reason"] == "requested",
           "the reason stays inside the fixed frontend enum")
        ok("stop also failed" in message["message"],
           "the message names the double failure")
        ok(authority._retry_task is not None and not authority._retry_task.done(),
           "a stop-retry loop is armed for the suspected train")
        ok(authority._watchdog_task is None,
           "no watchdog is armed for a train that was never confirmed started")
        ok(authority._owner.socket is sock,
           "the suspected train is bound to the requester, so a disconnect "
           "gets a shot at it too")

        client.stop_ok = True
        await asyncio.sleep(0.2)
        ok(authority.active is False, "the retry lands the stop and clears the belief")
        ok(recorder.last["active"] is False and recorder.last["status"] == "ok",
           "recovery is re-broadcast")

    arun(scenario)


def test_start_failure_with_successful_defensive_stop_stays_quiet():
    """Control for the test above: the ordinary failed start is unchanged."""
    async def scenario():
        authority, client, _ = _authority()
        client.start_ok = False
        client.stop_ok = True

        message = await authority.handle_start(start_request(), Sock("requester"))

        ok(authority.active is False and message["active"] is False,
           "a clean failure still reports active=False")
        ok(authority._retry_task is None,
           "no retry is armed when the defensive stop lands")

    arun(scenario)


# ── the ownership race ───────────────────────────────────────────────────────

async def _legacy_on_client_disconnect(authority, websocket, remaining_clients):
    """Replica of the pre-fix logic: ownership snapshotted BEFORE the lock.

    Kept verbatim so the negative control below proves the race test can fail.
    """
    owns = authority._owner.socket is websocket
    if not authority._active:
        return None
    if not owns and remaining_clients > 0:
        return None
    async with authority._lock:
        if not authority._active:
            return None
        return await authority._stop_locked("client_disconnect")


async def _interleave_disconnect_with_start(authority, client, disconnect_call,
                                            sock_a, sock_b):
    """Park B's start inside its HTTP PUT, then race A's disconnect against it.

    Returns the number of DELETEs issued while the two were interleaved.
    """
    client.start_gate = asyncio.Event()
    start_b = asyncio.create_task(authority.handle_start(start_request(), sock_b))
    await asyncio.sleep(0.02)              # B now holds the lock, mid-request
    stops_before = client.stops
    disconnect_a = asyncio.create_task(disconnect_call(authority, sock_a, 1))
    await asyncio.sleep(0.02)              # A is queued behind B on the lock
    client.start_gate.set()                # let B's start complete
    await start_b
    result = await disconnect_a
    client.start_gate = None
    return client.stops - stops_before, result


def test_owner_disconnect_does_not_kill_another_clients_new_train():
    """The whole ownership decision must be evaluated under the lock."""
    async def scenario():
        authority, client, _ = _authority()
        sock_a, sock_b = Sock("A"), Sock("B")
        await authority.handle_start(start_request(), sock_a)
        ok(authority._owner.socket is sock_a, "A owns the train")

        stops, result = await _interleave_disconnect_with_start(
            authority, client,
            lambda auth, s, n: auth.on_client_disconnect(s, remaining_clients=n),
            sock_a, sock_b)

        ok(authority.active is True,
           "B's brand-new train survives A's disconnect")
        ok(authority._owner.socket is sock_b, "ownership transferred to B")
        ok(stops == 0, "no DELETE was fired off a stale ownership snapshot")
        ok(result is None, "the disconnect handler declined to act")

        await authority.handle_stop(sock_b)

    arun(scenario)


def test_pre_lock_ownership_snapshot_would_have_killed_it():
    """Negative control: the same interleaving against the old logic."""
    async def scenario():
        authority, client, _ = _authority()
        sock_a, sock_b = Sock("A"), Sock("B")
        await authority.handle_start(start_request(), sock_a)

        stops, _ = await _interleave_disconnect_with_start(
            authority, client, _legacy_on_client_disconnect, sock_a, sock_b)

        ok(stops == 1 and not authority.active,
           "the pre-fix logic DID kill B's train, so the race test has teeth")

    arun(scenario)


def test_owner_disconnect_still_stops_its_own_train():
    """Control: with no interleaving, the disconnect stop is unchanged."""
    async def scenario():
        authority, client, _ = _authority()
        owner = Sock("owner")
        await authority.handle_start(start_request(), owner)

        stops_before = client.stops
        result = await authority.on_client_disconnect(owner, remaining_clients=1)

        ok(client.stops == stops_before + 1 and not authority.active,
           "a real owner disconnect still stops the train")
        ok(result["reason"] == "client_disconnect", "reason=client_disconnect")

    arun(scenario)


# ── racing stop paths ────────────────────────────────────────────────────────

def test_watchdog_and_manual_stop_race_publishes_once():
    """The loser must not overwrite the true reason with its own."""
    async def scenario():
        authority, client, recorder = _authority(max_train_seconds=0.05)
        sock = Sock("requester")
        await authority.handle_start(start_request(), sock)
        recorder.clear()

        client.stop_gate = asyncio.Event()      # park the watchdog's DELETE
        await asyncio.sleep(0.12)               # watchdog fired, holds the lock
        stops_before = client.stops
        manual = asyncio.create_task(authority.handle_stop(sock))
        await asyncio.sleep(0.02)               # manual stop queued on the lock
        client.stop_gate.set()
        await asyncio.sleep(0.05)
        message = await manual
        client.stop_gate = None

        ok(len(recorder) == 1, f"exactly one broadcast (got {len(recorder)})")
        ok(recorder.messages[0]["reason"] == "watchdog",
           "the surviving broadcast is the true one")
        ok(message["status"] == "ok" and message["active"] is False,
           "the manual stop still gets a well-formed idempotent ack")
        ok(client.stops == stops_before + 2,
           "an operator stop is deliberately forced through to the hardware "
           "even when we already believe we are idle")

    arun(scenario)


def test_two_autonomous_stops_race_publishes_once():
    """Autonomous losers short-circuit entirely: no DELETE, no broadcast."""
    async def scenario():
        authority, client, recorder = _authority()
        sock = Sock("owner")
        await authority.handle_start(start_request(), sock)
        recorder.clear()

        client.stop_gate = asyncio.Event()
        device_lost = asyncio.create_task(authority.on_device_lost())
        await asyncio.sleep(0.02)               # device-loss holds the lock
        stops_before = client.stops
        disconnect = asyncio.create_task(
            authority.on_client_disconnect(sock, remaining_clients=0))
        await asyncio.sleep(0.02)
        client.stop_gate.set()
        await device_lost
        await disconnect
        client.stop_gate = None

        ok(len(recorder) == 1 and recorder.messages[0]["reason"] == "device_lost",
           "the autonomous loser publishes nothing")
        ok(client.stops == stops_before + 1,
           "the autonomous loser fires no redundant DELETE")

    arun(scenario)


def test_retry_success_uses_the_shared_bookkeeping():
    """One helper writes stop-success state, so no path can drift from it."""
    async def scenario():
        authority, client, recorder = _authority(
            stop_retry_interval_s=0.05, stop_retry_limit=10)
        sock = Sock("requester")
        await authority.handle_start(start_request(), sock)

        client.stop_ok = False
        await authority.handle_stop(sock)
        ok(authority._retry_task is not None, "a retry is armed after a failed stop")

        client.stop_ok = True
        await asyncio.sleep(0.2)
        ok(authority.active is False and authority._owner.socket is None,
           "the retry path clears both active and owner")
        ok(authority._watchdog_task is None and authority._retry_task is None,
           "the retry path disarms the watchdog and clears its own handle")
        ok(recorder.last["status"] == "ok"
           and "retry attempt" in recorder.last["message"],
           "the retry success is re-broadcast with its note")

    arun(scenario)


# ── bounded diagnostics ──────────────────────────────────────────────────────

def test_rejected_controller_url_memory_is_capped():
    """A client sending a fresh URL per message must not grow the set forever."""
    async def scenario():
        authority, _, _ = _authority()

        for i in range(MAX_REMEMBERED_REJECTED_URLS + 40):
            authority._note_url_override(f"http://evil{i}.example")

        ok(len(authority._rejected_urls) == MAX_REMEMBERED_REJECTED_URLS,
           f"the remembered set is capped at {MAX_REMEMBERED_REJECTED_URLS} "
           f"(got {len(authority._rejected_urls)})")
        ok(authority._url_cap_logged is True,
           "the cap notice is latched, so it is printed exactly once")

    arun(scenario)


# ── handle_message must not swallow a dropped socket ─────────────────────────

def test_connection_closed_propagates_out_of_handle_message():
    """Audit D3 follow-up, checked against the real server modules.

    `ConnectionClosed` is a plain `Exception`, so the catch-all added for D3
    would otherwise treat a dropped socket as a command bug: an alarming
    traceback, a reply written to a socket that is gone, and - worse - the
    dedicated disconnect path (which now stops stimulation) never sees it.
    """
    ok(issubclass(StubConnectionClosed, Exception),
       "ConnectionClosed is a plain Exception, so clause order matters")

    RippleWebSocketServer, SimulatedWebSocketServer = import_servers()

    async def scenario(output):
        for label, cls, kwargs in [
            ("live", RippleWebSocketServer, {}),
            ("sim", SimulatedWebSocketServer, {"npz_path": "unused.npz"}),
        ]:
            # The constructors makedirs() their output folder, so give them a
            # throwaway one rather than littering the checkout.
            server = cls(output_folder=output, **kwargs)

            dropped = Sock("dropped")
            dropped.raise_on_send = StubConnectionClosed("peer went away")
            raised = None
            try:
                await server.handle_message('{"command": "get_status"}', dropped)
            except StubConnectionClosed:
                raised = "ConnectionClosed"
            except Exception as exc:                              # noqa: BLE001
                raised = type(exc).__name__
            ok(raised == "ConnectionClosed",
               f"[{label}] ConnectionClosed propagates out of handle_message "
               f"(got {raised!r})")

            # ...while a genuine command bug is still contained and reported.
            live = Sock("live")
            escaped = None
            try:
                await server.handle_message(
                    '{"command": "monitor_trigger", "channel": "not-a-number"}',
                    live)
            except Exception as exc:                              # noqa: BLE001
                escaped = type(exc).__name__
            ok(escaped is None,
               f"[{label}] a bad command does not escape and close the socket")
            ok(any("monitor_trigger" in sent and "error" in sent
                   for sent in live.sent),
               f"[{label}] the requester is told which command failed")

    with tempfile.TemporaryDirectory(prefix="stim_authority_test_") as tmp:
        arun(lambda: scenario(tmp))


TESTS = [
    test_start_failure_with_failed_defensive_stop_is_treated_as_running,
    test_start_failure_with_successful_defensive_stop_stays_quiet,
    test_owner_disconnect_does_not_kill_another_clients_new_train,
    test_pre_lock_ownership_snapshot_would_have_killed_it,
    test_owner_disconnect_still_stops_its_own_train,
    test_watchdog_and_manual_stop_race_publishes_once,
    test_two_autonomous_stops_race_publishes_once,
    test_retry_success_uses_the_shared_bookkeeping,
    test_rejected_controller_url_memory_is_capped,
    test_connection_closed_propagates_out_of_handle_message,
]


if __name__ == "__main__":
    sys.exit(run_suite("stimulation authority: hardening regressions", TESTS))
