"""Live acquisition server: devices, WebSocket servers and session managers.

Entrypoints, launched by the ``.bat`` files at the repository root:

    server_cli.py               Ripple hardware  (start.bat)
    quattrocento_server_cli.py  OTBioLab+ / Quattrocento  (start_quattrocento.bat)
    simulate_server_cli.py      .npz playback  (start_sim.bat)

Each starts a WebSocket server that connects to its device, filters, records, serves
the frontend, and publishes an LSL outlet.

Nothing is re-exported here on purpose. The modules are imported as siblings by the
entrypoints, and eager imports would pull in ``xipppy``, ``pylsl`` and ``websockets``
merely to touch the package.
"""