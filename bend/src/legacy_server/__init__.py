"""Superseded two-process acquisition architecture. Retained for reference only.

The original design split acquisition from serving across two processes:

    RippleDevice --> RippleStream (stream.py) --> LSL outlet        [process 1, cli.py]
                                                      |
    LSL inlet --> EMGWebSocketServer --> WebSocket --> frontend     [process 2]
                  (emg_websocket_server.py)

``simulated_stream.py`` / ``simulate_cli.py`` are the same publisher sourced from a
``.npz`` file instead of hardware.

This was replaced in February 2026 by ``server/websocket_server.py``, which does
all of it in a single process: it talks to the device directly, filters, buffers,
records, serves the WebSocket, *and* still publishes its own LSL outlet (see
``enable_lsl`` / the ``--no-lsl`` flag). No capability was lost in the collapse —
only the process boundary went away.

Nothing here is reachable from any current entrypoint. The device drivers it depends
on still live in the active package, so these modules import from ``server``.

Imports are deliberately not re-exported here: ``stream`` pulls in ``xipppy`` and
``pylsl`` at module level, and this package should cost nothing to have on disk.
"""