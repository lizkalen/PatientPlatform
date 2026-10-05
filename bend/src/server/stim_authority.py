"""
Stimulation Authority
=====================

Server-side ownership, validation and lifecycle for electrical stimulation.

Why this module exists (architecture audit, section S1-S8): stimulation used to
be 100% browser-owned. Every train was started with ``duration_sec = -1``
(infinite) and the only off-switch in the whole system was a WebSocket message
from a browser tab. A crashed tab runs no JavaScript, so nothing stopped the
train on disconnect, on device loss, or on backend shutdown. The controller
process (``samolator``) outlives this backend, so "stimulate forever" really
does mean forever.

Safety invariants have to live in the last process that survives. This class is
that process's policy layer. It WRAPS :class:`~server.stimulation_client.StimulationClient`
(which stays a dumb HTTP client) and adds:

    * parameter validation against :class:`StimLimits` - **reject, never clamp**,
      because silently delivering a different current than the operator asked
      for is its own hazard;
    * ownership - a train is bound to the WebSocket that started it, and is
      stopped when that socket goes away, or when the last client goes away;
    * a dead-man deadline armed on every start, independent of decisions,
      messages, sockets and the acquisition loop;
    * a truthful ``active`` flag that reflects what the SERVER believes, plus a
      retry loop so a stop that failed at the hardware is not quietly reported
      as "stimulation off" (audit S6);
    * one stop path for device loss and process teardown.

Console vs logging: the CLI entrypoints never call ``logging.basicConfig``, so
``logger.info`` output is dropped on the floor. The operator's only feedback
channel during a session is the backend console window, so operator-relevant
events are ``print``ed with a ``[StimAuthority]`` prefix, matching the style of
the servers this is wired into.
"""

import asyncio
import json
import math
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from server.stimulation_client import StimulationClient


# ─────────────────────────────────────────────────────────────────────────────
# Parameter ceilings
#
# !!!  THESE NUMBERS WERE INHERITED FROM THE FRONTEND FORM SCHEMA AND HAVE   !!!
# !!!  NOT BEEN CLINICALLY REVIEWED. THEY ARE NOT A THERAPY SPECIFICATION.   !!!
#
# They are copied verbatim from `fend/src/config/sequenceSchema.js:165-169`
# (STIM_CHANNEL_FIELDS), where they exist only as HTML `min`/`max` attributes.
# That file's own comments state the values "are NOT clamped on read" and that
# "min/max are advisory only" - i.e. before this module they constrained
# nothing at all, on either side of the WebSocket. Making them the backend's
# hard limits is strictly safer than the previous state (no limit whatsoever),
# but it is NOT the same as having correct limits:
#
#   * 130 mA is a device-level ceiling, not a patient-level one. A safe
#     amplitude depends on electrode size, placement, phase charge and the
#     individual patient.
#   * 2000 Hz frequency and 1000 us pulse width are likewise the widest values
#     the form allowed, not values anyone signed off on for a human.
#   * There is no cross-parameter check here (no charge-per-phase, no duty
#     cycle, no total-current-across-channels budget).
#
# ACTION REQUIRED: have these reviewed and re-derived by the clinical lead
# responsible for the protocol, per stimulator model, and record the rationale
# next to each value. Until then they are a backstop against typos and rogue
# messages, not a guarantee of patient safety.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StimLimits:
    """Hard bounds for one stimulation channel. Out-of-range values are REJECTED.

    Single source of truth for the backend. Bounds are inclusive.
    """

    amplitude_min_ma: float = 0.0
    amplitude_max_ma: float = 130.0
    pulse_width_min_us: float = 10.0
    pulse_width_max_us: float = 1000.0
    frequency_min_hz: float = 0.1
    frequency_max_hz: float = 2000.0
    # Channel ids as the controller numbers them (schema: id min 1, max 8).
    channel_id_min: int = 1
    channel_id_max: int = 8
    # A request addressing more channels than the device has is a bug, not a
    # therapy: the RehaMove3 exposes 8. Also caps the size of a hostile payload.
    max_channels: int = 8


DEFAULT_STIM_LIMITS = StimLimits()

# Dead-man deadline for a single train, in seconds. Armed at every start,
# independent of the client, the decision stream and the acquisition loop.
#
# !!!  30 s IS A BACKSTOP, NOT A THERAPY PARAMETER.  !!!
# It exists so that a train can never outlive the failure that stranded it -
# nothing more. It was chosen as "longer than any single MOVE phase the current
# frontend produces, short enough that an unattended train is measured in
# seconds". If a legitimate protocol needs a longer continuous train, raise it
# with `--stim-max-seconds` AFTER the clinical lead has agreed a value; do not
# treat the default as an endorsed maximum stimulation duration.
DEFAULT_MAX_TRAIN_SECONDS = 30.0

# A stop that fails at the hardware is retried on this cadence until it lands.
# Capped so a permanently unreachable controller cannot leave a task spinning
# for the life of the process - after the cap we go loud instead of quiet.
DEFAULT_STOP_RETRY_INTERVAL_S = 2.0
DEFAULT_STOP_RETRY_LIMIT = 15          # 15 x 2 s = 30 s of retries

# Teardown is bounded differently: the event loop is about to close, so we
# cannot retry for 30 s, but one failed DELETE is not enough to give up on.
SHUTDOWN_STOP_ATTEMPTS = 3
SHUTDOWN_STOP_INTERVAL_S = 1.0

# Distinct client-supplied controller_url values are remembered so each is
# warned about once instead of once per command. Capped: a client sending a
# fresh URL every message would otherwise grow the set without bound, and the
# warning is a diagnostic, not a security control - the URL is ignored either
# way, whether or not we bothered to name it.
MAX_REMEMBERED_REJECTED_URLS = 32

# Reasons carried by `stimulation_status`. Kept here so both servers and any
# future test agree on the vocabulary.
REASON_REQUESTED = "requested"
REASON_REJECTED = "rejected"
REASON_WATCHDOG = "watchdog"
REASON_CLIENT_DISCONNECT = "client_disconnect"
REASON_DEVICE_LOST = "device_lost"
REASON_SHUTDOWN = "shutdown"


class StimParameterError(ValueError):
    """A `stimulate_start` payload violated StimLimits or was malformed.

    Raised before anything touches the hardware; the request is refused whole.
    """


def _as_finite_number(value: Any, what: str) -> float:
    """Coerce to float, rejecting bools, non-numerics, NaN and +/-inf.

    `isinstance(True, int)` is True in Python, so bools are excluded explicitly:
    `amplitude: true` must not be read as 1 mA.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StimParameterError(
            f"{what} must be a number, got {type(value).__name__} ({value!r})"
        )
    out = float(value)
    if not math.isfinite(out):
        raise StimParameterError(f"{what} must be finite, got {out!r}")
    return out


def _in_range(value: float, low: float, high: float, what: str, unit: str) -> float:
    if not (low <= value <= high):
        raise StimParameterError(
            f"{what} {value:g} {unit} is outside the permitted range "
            f"{low:g}-{high:g} {unit}"
        )
    return value


def validate_channels(channels: Any, limits: StimLimits = DEFAULT_STIM_LIMITS) -> list:
    """Validate the channel list of a `stimulate_start`.

    Returns the channel list to forward to the controller. Raises
    :class:`StimParameterError` on the first violation - the whole request is
    refused, and nothing is clamped or silently corrected.

    Unknown keys are forwarded verbatim: the controller's channel schema is
    owned by `samolator`, not by us, and dropping a key it needs would break
    stimulation in a way that looks like a hardware fault. `duration_sec` is
    the one exception and is not a hole: `StimulationClient.start_train`
    overrides it to -1 unconditionally.
    """
    if not isinstance(channels, list):
        raise StimParameterError(
            f"'channels' must be a list, got {type(channels).__name__}"
        )
    if not channels:
        raise StimParameterError("'channels' must not be empty")
    if len(channels) > limits.max_channels:
        raise StimParameterError(
            f"{len(channels)} channels requested, at most {limits.max_channels} allowed"
        )

    seen_ids = set()
    for i, ch in enumerate(channels):
        if not isinstance(ch, dict):
            raise StimParameterError(
                f"channel[{i}] must be an object, got {type(ch).__name__}"
            )

        for key in ("id", "amplitude", "pulse_width", "frequency"):
            if key not in ch:
                raise StimParameterError(f"channel[{i}] is missing '{key}'")

        # Channel id: an integer electrode index. Accept 1.0 as 1, reject 1.5.
        raw_id = _as_finite_number(ch["id"], f"channel[{i}].id")
        if raw_id != int(raw_id):
            raise StimParameterError(f"channel[{i}].id must be a whole number, got {raw_id!r}")
        ch_id = int(raw_id)
        _in_range(ch_id, limits.channel_id_min, limits.channel_id_max,
                  f"channel[{i}].id", "")
        if ch_id in seen_ids:
            # Two entries for one electrode is a config bug, and whether the
            # controller sums or overwrites them is undefined from here.
            raise StimParameterError(f"channel id {ch_id} appears more than once")
        seen_ids.add(ch_id)

        _in_range(_as_finite_number(ch["amplitude"], f"channel[{i}].amplitude"),
                  limits.amplitude_min_ma, limits.amplitude_max_ma,
                  f"channel[{i}].amplitude", "mA")
        _in_range(_as_finite_number(ch["pulse_width"], f"channel[{i}].pulse_width"),
                  limits.pulse_width_min_us, limits.pulse_width_max_us,
                  f"channel[{i}].pulse_width", "us")
        _in_range(_as_finite_number(ch["frequency"], f"channel[{i}].frequency"),
                  limits.frequency_min_hz, limits.frequency_max_hz,
                  f"channel[{i}].frequency", "Hz")

        if "is_biphasic" in ch and not isinstance(ch["is_biphasic"], bool):
            raise StimParameterError(
                f"channel[{i}].is_biphasic must be true or false, "
                f"got {ch['is_biphasic']!r}"
            )

    return channels


def validate_start_request(data: dict, limits: StimLimits = DEFAULT_STIM_LIMITS) -> dict:
    """Validate a whole `stimulate_start` command.

    Returns the sanitised keyword arguments for
    :meth:`StimulationClient.start_train`. Note that `controller_url` is
    deliberately absent: the server's configured URL is the only one used
    (audit S7 - a client could otherwise redirect stimulation to any host).
    """
    if not isinstance(data, dict):
        raise StimParameterError("stimulate_start payload must be an object")

    channels = validate_channels(data.get("channels"), limits)

    stimulator_type = data.get("stimulator_type")
    if stimulator_type is not None and not isinstance(stimulator_type, str):
        raise StimParameterError("'stimulator_type' must be a string or omitted")
    # Deliberately NOT an allowlist: the set of stimulator backends lives in
    # samolator, and hard-coding it here would break the next device added
    # there. The value only selects a driver; it cannot exceed a dose.

    port = data.get("port")
    if port is not None and not isinstance(port, str):
        raise StimParameterError("'port' must be a string or omitted")

    metadata = data.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise StimParameterError("'metadata' must be an object or omitted")

    return {
        "channels": channels,
        "stimulator_type": stimulator_type,
        "port": port,
        "metadata": metadata,
    }


@dataclass
class _Ownership:
    """Who started the train we believe is running, and when."""
    socket: Optional[Any] = None
    client_id: Optional[int] = None


class StimAuthority:
    """Owns stimulation state for one server instance.

    The server no longer talks to :class:`StimulationClient` directly; every
    path (command, disconnect, device loss, watchdog, teardown) goes through
    here so that `active` has exactly one writer.

    Args:
        client: the HTTP client to drive. Built with the server-configured
            controller URL; created here if not supplied.
        broadcast: ``async (dict) -> None`` used to publish `stimulation_status`
            to every connected client. Typically ``server.broadcast``.
        limits: parameter ceilings (see :class:`StimLimits`).
        max_train_seconds: dead-man deadline per train (``--stim-max-seconds``).
    """

    def __init__(
        self,
        client: Optional[StimulationClient] = None,
        broadcast: Optional[Callable[[dict], Awaitable[None]]] = None,
        limits: StimLimits = DEFAULT_STIM_LIMITS,
        max_train_seconds: float = DEFAULT_MAX_TRAIN_SECONDS,
        stop_retry_interval_s: float = DEFAULT_STOP_RETRY_INTERVAL_S,
        stop_retry_limit: int = DEFAULT_STOP_RETRY_LIMIT,
    ):
        self.client = client if client is not None else StimulationClient()
        self.limits = limits
        self.max_train_seconds = float(max_train_seconds)
        self.stop_retry_interval_s = float(stop_retry_interval_s)
        self.stop_retry_limit = int(stop_retry_limit)
        self._broadcast_fn = broadcast

        # The server's belief about the hardware. THE only copy: StimulationClient
        # also carries an `active` attribute, which was dead state that looked
        # like an interlock and was never read (audit S2). It is still written by
        # the client on each call; nothing reads it, and nothing should - this
        # flag is the one that survives a failed stop.
        self._active = False
        self._owner = _Ownership()

        # Serialises start/stop so a watchdog fire, a disconnect and a command
        # cannot interleave two HTTP calls against one train.
        self._lock = asyncio.Lock()
        self._watchdog_task: Optional[asyncio.Task] = None
        self._retry_task: Optional[asyncio.Task] = None
        # controller_url values a client tried to override, so the warning is
        # logged once per distinct value instead of once per command. Bounded by
        # MAX_REMEMBERED_REJECTED_URLS; _url_cap_logged makes the "no longer
        # naming them" notice appear exactly once.
        self._rejected_urls = set()
        self._url_cap_logged = False

    # ── state ────────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        """What the SERVER believes about the hardware right now.

        Stays True after a stop that failed, until a retry lands.
        """
        return self._active

    @property
    def controller_url(self) -> str:
        return self.client.base_url

    def set_broadcast(self, broadcast: Callable[[dict], Awaitable[None]]):
        """Late-bind the broadcast function (constructor injection preferred)."""
        self._broadcast_fn = broadcast

    # ── message plumbing ─────────────────────────────────────────────────────

    @staticmethod
    def status_message(active: bool, status: str, reason: Optional[str] = None,
                       message: Optional[str] = None) -> dict:
        """Build a `stimulation_status` message.

        Contract:
            {type: "stimulation_status",
             active: bool,                 # the server's actual belief
             status: "ok" | "error",
             reason?: "requested" | "rejected" | "watchdog" | "client_disconnect"
                      | "device_lost" | "shutdown",
             message?: str}

        `active` is built FIRST and never splatted over (the old call sites did
        ``{"active": False, **result}``, so a failed stop reported the train as
        off - audit S6).
        """
        msg = {"type": "stimulation_status", "active": bool(active), "status": status}
        if reason is not None:
            msg["reason"] = reason
        if message:
            msg["message"] = message
        return msg

    async def _publish(self, msg: dict):
        """Broadcast a status message; never let a socket problem break a stop."""
        if self._broadcast_fn is None:
            return
        try:
            await self._broadcast_fn(msg)
        except Exception as exc:
            print(f"[StimAuthority] could not broadcast stimulation_status: {exc}")

    @staticmethod
    async def _reply(msg: dict, websocket) -> None:
        """Answer the requesting socket directly (used for rejections)."""
        if websocket is None:
            return
        try:
            await websocket.send(json.dumps(msg))
        except Exception:
            # The requester vanished mid-command; nothing to do, and this must
            # never propagate into the command handler.
            pass

    # ── commands ─────────────────────────────────────────────────────────────

    async def handle_start(self, data: dict, websocket=None) -> dict:
        """Handle a `stimulate_start` command. Returns the status message sent.

        Validation happens before the lock and before any HTTP call: a rejected
        request must not touch the hardware, and must not be able to delay a
        legitimate stop by holding the lock.
        """
        self._note_url_override(data.get("controller_url") if isinstance(data, dict) else None)

        try:
            request = validate_start_request(data, self.limits)
        except StimParameterError as exc:
            # `active` is the CURRENT belief, unchanged - a rejection is not a
            # state transition, so it is answered to the requester only.
            msg = self.status_message(self._active, "error", REASON_REJECTED, str(exc))
            print(f"[StimAuthority] stimulate_start REJECTED: {exc}")
            await self._reply(msg, websocket)
            return msg

        async with self._lock:
            return await self._start_locked(request, websocket)

    async def _start_locked(self, request: dict, websocket) -> dict:
        was_active = self._active

        result = await self.client.start_train(
            channels=request["channels"],
            stimulator_type=request["stimulator_type"],
            port=request["port"],
            # S7: the server's configured URL is authoritative. Any
            # client-supplied controller_url was dropped during validation.
            controller_url=None,
            metadata=request["metadata"],
        )

        if result.get("status") == "success":
            self._active = True
            self._owner = _Ownership(socket=websocket,
                                     client_id=id(websocket) if websocket is not None else None)
            # A pending stop-retry belongs to the previous train; letting it run
            # would kill the one we just started.
            self._cancel_retry()
            self._arm_watchdog()
            n_ch = len(request["channels"])
            print(f"[StimAuthority] train started ({n_ch} channel(s)), owner="
                  f"{self._owner.client_id}, deadline {self.max_train_seconds:g}s")
            msg = self.status_message(True, "ok", REASON_REQUESTED)
            await self._publish(msg)
            return msg

        message = result.get("message") or "could not start stimulation"
        if not was_active:
            # The PUT failed, but a failure response does not prove the
            # controller armed nothing - it may have started and then errored.
            # One idempotent DELETE costs nothing and closes that window. Only
            # safe when we believed nothing was running: if a train IS active,
            # stopping it here would silently end therapy the operator wants.
            print("[StimAuthority] start failed; sending a defensive stop")
            stop_result = await self.client.stop()
            if stop_result.get("status") != "success":
                # Both calls failed, so we know nothing about the hardware: the
                # controller may have armed the channels and then refused to
                # disarm them. Assume the worse of the two possibilities rather
                # than report "off" and walk away - an unwatched train we cannot
                # see is precisely the S6 failure this class exists to prevent.
                # Same treatment as a failed stop: believe it is running, keep
                # asking, stay loud.
                self._active = True
                # Bind it to the requester so stop-on-disconnect gets a shot at
                # it too; this socket's command is what may have created it.
                self._owner = _Ownership(
                    socket=websocket,
                    client_id=id(websocket) if websocket is not None else None)
                message = (f"{message}; the follow-up stop also failed "
                           f"({stop_result.get('message')}) - a train may be "
                           f"running and is being retried")
                print(f"[StimAuthority] *** START FAILED AND THE DEFENSIVE STOP "
                      f"ALSO FAILED: {message} — assuming a train is ACTIVE; "
                      f"retrying every {self.stop_retry_interval_s:g}s ***")
                # Reason stays "requested": this is the outcome of a client
                # command, and the reason vocabulary is a fixed frontend
                # contract - a new value would be dropped by the UI.
                self._schedule_retry(REASON_REQUESTED)
        else:
            print("[StimAuthority] start failed while a train was already active; "
                  "the previous train is left running and its deadline stands")

        # No watchdog re-arm on failure: if a train was already running, its
        # original deadline is still the one that must fire, and a train we only
        # SUSPECT exists is chased by the retry loop, not by a deadline.
        # `self._active`, not `was_active`: the branch above may have just
        # changed our belief, and reporting the stale one would re-create the
        # exact "failed stop broadcast as off" bug (audit S6).
        msg = self.status_message(self._active, "error", REASON_REQUESTED, message)
        await self._publish(msg)
        return msg

    async def handle_stop(self, websocket=None) -> dict:
        """Handle a `stimulate_stop` command. Returns the status message sent.

        Ownership is NOT enforced on stop: any client must be able to stop any
        train. An emergency stop that checks credentials first is not one, and
        for the same reason it is `force`d through to the hardware even when we
        believe nothing is running.
        """
        async with self._lock:
            return await self._stop_locked(REASON_REQUESTED, force=True)

    # ── autonomous stops ─────────────────────────────────────────────────────

    async def on_client_disconnect(self, websocket, remaining_clients: int) -> Optional[dict]:
        """Called from `handle_client`'s finally, after the socket is discarded.

        Stops the train when the departing socket owns it, and also when the
        last client leaves regardless of ownership - with nobody connected there
        is no one left who could ever send `stimulate_stop` (audit S2).

        The WHOLE decision is taken inside the lock, against ownership as it is
        at that moment. Reading `_owner` before acquiring is a real race: client
        B's `stimulate_start` can be holding the lock across its HTTP PUT while
        A's disconnect waits behind it, and a stale `owns=True` snapshot would
        then stop B's brand-new train under A's name - silently ending another
        client's therapy and mislabelling the broadcast reason.

        `remaining_clients` is inherently a snapshot (the authority cannot see
        the server's client set), but it can only go stale in the safe
        direction: it was taken after this socket was discarded, so a client
        that connects in the meantime makes us stop when we need not have,
        never the reverse.
        """
        async with self._lock:
            owns = self._owner.socket is websocket
            if not self._active:
                if owns:
                    self._owner = _Ownership()
                return None
            if not owns and remaining_clients > 0:
                return None

            why = ("the commanding client disconnected" if owns
                   else "the last client disconnected")
            print(f"[StimAuthority] stopping stimulation: {why}")
            return await self._stop_locked(REASON_CLIENT_DISCONNECT)

    async def on_device_lost(self) -> Optional[dict]:
        """Stop on acquisition loss (audit S3).

        The closed loop only closes the gate when a NEW decision says so, so a
        dead device means the gate never closes and the train runs forever.
        """
        if not self._active:
            return None
        print("[StimAuthority] stopping stimulation: acquisition device lost")
        async with self._lock:
            if not self._active:
                return None
            return await self._stop_locked(REASON_DEVICE_LOST)

    async def shutdown(self) -> dict:
        """Last instruction before the process exits (audit S2 / S8.4).

        Unconditional: even when we believe nothing is running, an idempotent
        DELETE is the right thing to leave a controller process that outlives
        this backend. Retries are bounded because the event loop is closing.
        """
        self._disarm_watchdog()
        self._cancel_retry()

        # Try to take the lock, but do not let a command stuck in an HTTP
        # timeout delay the last stop of the session.
        got_lock = False
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=2.0)
            got_lock = True
        except asyncio.TimeoutError:
            print("[StimAuthority] shutdown: a stimulation command is still in "
                  "flight; sending the stop anyway")

        try:
            for attempt in range(1, SHUTDOWN_STOP_ATTEMPTS + 1):
                result = await self.client.stop()
                if result.get("status") == "success":
                    # Same shared bookkeeping as every other stop path.
                    return await self._record_stop_success(REASON_SHUTDOWN)
                print(f"[StimAuthority] shutdown stop attempt {attempt}/"
                      f"{SHUTDOWN_STOP_ATTEMPTS} failed: {result.get('message')}")
                if attempt < SHUTDOWN_STOP_ATTEMPTS:
                    await asyncio.sleep(SHUTDOWN_STOP_INTERVAL_S)
        finally:
            if got_lock:
                self._lock.release()

        print("[StimAuthority] *** COULD NOT STOP THE STIMULATOR ON SHUTDOWN. "
              "IF A TRAIN WAS RUNNING IT IS STILL RUNNING — POWER DOWN THE "
              "STIMULATOR MANUALLY. ***")
        msg = self.status_message(
            self._active, "error", REASON_SHUTDOWN,
            "backend shut down without confirming the stimulator stopped")
        await self._publish(msg)
        return msg

    # ── the one stop path ────────────────────────────────────────────────────

    async def _record_stop_success(self, reason: str,
                                   note: Optional[str] = None) -> dict:
        """Bookkeeping for a DELETE that landed. Caller holds `self._lock`.

        The ONE place stop-success state is written, so the command path, the
        retry loop and the shutdown path cannot drift apart. Both cancellations
        here are re-entrant by design: `_disarm_watchdog` is called from the
        watchdog's own fire path and `_cancel_retry` from inside the retry loop,
        which is exactly what their `is current_task()` guards are for.
        """
        self._active = False
        self._owner = _Ownership()
        self._disarm_watchdog()
        self._cancel_retry()
        print(f"[StimAuthority] stimulation stopped (reason={reason})"
              + (f": {note}" if note else ""))
        msg = self.status_message(False, "ok", reason, note)
        await self._publish(msg)
        return msg

    async def _stop_locked(self, reason: str, force: bool = False) -> dict:
        """Issue the stop and update state. Caller holds `self._lock`.

        `force` is set by the operator-facing `stimulate_stop` command only. An
        emergency stop must reach the hardware even when we believe nothing is
        running, because that belief can be wrong - a second backend instance
        against the same controller is reachable (audit S7), and the DELETE is
        idempotent and free. Autonomous stops (watchdog, disconnect, device
        loss) short-circuit instead: when two of them race for this lock, the
        loser must not fire a redundant DELETE and, far worse, publish a second
        differently-reasoned "stopped" that overwrites the true one.

        Either way, a call that does not change our belief publishes nothing. It
        still returns a well-formed idempotent ack, which is safe to unicast.
        """
        if not self._active and not force:
            return self.status_message(False, "ok", reason)

        was_active = self._active
        self._disarm_watchdog()
        self._cancel_retry()

        result = await self.client.stop()
        if result.get("status") == "success":
            if not was_active:
                return self.status_message(False, "ok", reason)
            return await self._record_stop_success(reason)

        message = result.get("message") or "could not stop stimulation"
        if was_active:
            # We asked and it did not land, so the train is still running as far
            # as we know. Say so (audit S6: this used to be broadcast as
            # "active": false) and keep asking.
            print(f"[StimAuthority] *** STOP FAILED (reason={reason}): {message} — "
                  f"a train is believed ACTIVE; retrying every "
                  f"{self.stop_retry_interval_s:g}s ***")
            self._schedule_retry(reason)
        else:
            # A forced stop while idle: nothing was running, so a failed no-op
            # DELETE must not invent a train or start a retry storm. Still
            # reported - the controller is unreachable and the operator, who
            # just pressed stop, should know that.
            print(f"[StimAuthority] stop failed while idle (reason={reason}): {message}")

        msg = self.status_message(self._active, "error", reason, message)
        await self._publish(msg)
        return msg

    # ── stop retry ───────────────────────────────────────────────────────────

    def _schedule_retry(self, reason: str):
        if self._retry_task is not None and not self._retry_task.done():
            return                      # one retry loop is enough
        self._retry_task = asyncio.create_task(self._retry_stop_loop(reason))

    def _cancel_retry(self):
        task = self._retry_task
        self._retry_task = None
        # `task is current_task()` guards the case where the retry loop itself
        # reaches a code path that clears the retry: cancelling yourself here
        # would raise CancelledError at the next await.
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _retry_stop_loop(self, reason: str):
        """Keep asking the controller to stop until it does, or we run out.

        Runs detached from any socket: the client that started the train may be
        long gone by the time this lands.
        """
        for attempt in range(1, self.stop_retry_limit + 1):
            try:
                await asyncio.sleep(self.stop_retry_interval_s)
            except asyncio.CancelledError:
                return
            async with self._lock:
                if not self._active:
                    return              # somebody else got it stopped
                print(f"[StimAuthority] stop retry {attempt}/{self.stop_retry_limit} "
                      f"(reason={reason})")
                result = await self.client.stop()
                if result.get("status") == "success":
                    # Shared bookkeeping: this loop must not keep its own copy
                    # of "what stopping means" and drift from the command path.
                    await self._record_stop_success(
                        reason, f"stopped after {attempt} retry attempt(s)")
                    return
                print(f"[StimAuthority] stop retry {attempt} failed: "
                      f"{result.get('message')}")

        print("[StimAuthority] *** GAVE UP AFTER "
              f"{self.stop_retry_limit} STOP ATTEMPTS (reason={reason}). "
              "THE STIMULATOR MAY STILL BE RUNNING — POWER IT DOWN MANUALLY. ***")
        await self._publish(self.status_message(
            self._active, "error", reason,
            f"stop failed {self.stop_retry_limit} times; the stimulator may still "
            f"be running — power it down manually"))

    # ── dead-man watchdog ────────────────────────────────────────────────────

    def _arm_watchdog(self):
        """(Re)arm the hard deadline. Called on every successful start."""
        self._disarm_watchdog()
        self._watchdog_task = asyncio.create_task(
            self._watchdog_loop(self.max_train_seconds))

    def _disarm_watchdog(self):
        """Cancel the deadline. Called on every stop, BEFORE the HTTP call, so a
        requested stop can never be followed by a spurious watchdog fire."""
        task = self._watchdog_task
        self._watchdog_task = None
        # When the watchdog's own fire path calls this, cancelling would abort
        # the very stop it is trying to perform.
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _watchdog_loop(self, deadline_s: float):
        try:
            await asyncio.sleep(deadline_s)
        except asyncio.CancelledError:
            return                      # normal stop got there first
        print(f"[StimAuthority] *** WATCHDOG FIRED: no stop within {deadline_s:g}s "
              f"of the train starting — forcing stimulation off ***")
        async with self._lock:
            if not self._active:
                return
            await self._stop_locked(REASON_WATCHDOG)

    # ── misc ─────────────────────────────────────────────────────────────────

    def _note_url_override(self, url: Optional[str]):
        """Warn (once per value) that a client-supplied controller_url is ignored.

        Audit S7: `controller_url` used to be taken per-command straight from
        the browser, so any tab could redirect stimulation to an arbitrary host.
        It is now configuration (`--stim-controller-url`), not a message field.
        """
        if not url or url in self._rejected_urls:
            return
        if len(self._rejected_urls) >= MAX_REMEMBERED_REJECTED_URLS:
            # Stop growing the set. The URL is still ignored - only the
            # per-value diagnostic stops, and only after saying so once.
            if not self._url_cap_logged:
                self._url_cap_logged = True
                print(f"[StimAuthority] more than {MAX_REMEMBERED_REJECTED_URLS} "
                      f"distinct client-supplied controller_url values seen; "
                      f"further ones are still ignored but will no longer be "
                      f"reported individually")
            return
        self._rejected_urls.add(url)
        if url.rstrip("/") != self.client.base_url:
            print(f"[StimAuthority] ignoring client-supplied controller_url "
                  f"{url!r}; using the server's configured {self.client.base_url!r}")
