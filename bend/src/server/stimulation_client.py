"""
Stimulation Client
==================

Thin async client that forwards stimulation start/stop requests from the
PatientGUI websocket server to the external stimulator controller (samolator).

It uses the controller's *existing* HTTP API - no changes are needed on the
stimulator side:

    PUT    {base_url}/api/control/stimulate   -> start a train
    DELETE {base_url}/api/control/stop        -> stop any active stimulation

The intended flow (driven by the frontend SequencePlayer) is:
    - on entering a "move" phase  -> start_train(...) with duration_sec = -1
      (an infinite train that runs until explicitly stopped)
    - on leaving the "move" phase -> stop()

Stimulation parameters (channels, amplitude, pulse width, frequency, ...) are
defined in the PatientGUI frontend and passed straight through to the
controller. Failures are swallowed and reported, never raised, so a missing or
unreachable stimulator never interrupts the EMG recording session.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_CONTROLLER_URL = "http://127.0.0.1:11051"
DEFAULT_STIMULATOR_TYPE = "science_mode3"


class StimulationClient:
    """Forwards start/stop commands to the stimulator controller over HTTP."""

    def __init__(
        self,
        base_url: str = DEFAULT_CONTROLLER_URL,
        stimulator_type: str = DEFAULT_STIMULATOR_TYPE,
        port: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.stimulator_type = stimulator_type
        self.port = port
        self.timeout = timeout
        # Whether we believe a stimulation we started is currently running.
        self.active = False

    def _url(self, override: Optional[str], path: str) -> str:
        base = (override or self.base_url).rstrip("/")
        return f"{base}{path}"

    async def start_train(
        self,
        channels: list,
        stimulator_type: Optional[str] = None,
        port: Optional[str] = None,
        controller_url: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Start an infinite train of pulses; runs until stop() is called.

        `channels` is a list of channel dicts as defined in the frontend, e.g.
            {"id": 1, "amplitude": 8.0, "pulse_width": 200,
             "frequency": 50.0, "is_biphasic": true}
        `duration_sec` is forced to -1 (infinite) so the movement phase, not a
        fixed timer, controls how long stimulation lasts.
        """
        if not channels:
            msg = "stimulate_start ignored: no channels provided"
            logger.warning(msg)
            return {"status": "error", "message": msg}

        body = {
            "mode": "train",
            "stimulator_type": stimulator_type or self.stimulator_type,
            "port": port if port is not None else self.port,
            # Force infinite duration; stop() ends it when the movement ends.
            "channels": [{**ch, "duration_sec": -1} for ch in channels],
            "metadata": metadata or {},
        }

        url = self._url(controller_url, "/api/control/stimulate")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.put(url, json=body)
                resp.raise_for_status()
                self.active = True
                logger.info("Stimulation train started (%d channel(s))", len(channels))
                return {"status": "success", "response": resp.json()}
        except httpx.HTTPError as exc:
            logger.error("Failed to start stimulation at %s: %s", url, exc)
            return {"status": "error", "message": f"Could not start stimulation: {exc}"}

    async def stop(self, controller_url: Optional[str] = None) -> dict:
        """Stop any active stimulation. Safe to call when nothing is running."""
        url = self._url(controller_url, "/api/control/stop")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.delete(url)
                resp.raise_for_status()
                self.active = False
                logger.info("Stimulation stopped")
                return {"status": "success", "response": resp.json()}
        except httpx.HTTPError as exc:
            logger.error("Failed to stop stimulation at %s: %s", url, exc)
            return {"status": "error", "message": f"Could not stop stimulation: {exc}"}
