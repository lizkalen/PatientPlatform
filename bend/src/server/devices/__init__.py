"""Acquisition device drivers.

Three interchangeable sources, selected by which ``*_server_cli`` entrypoint is run:

    ripple.py        RippleDevice        Ripple Trellis over ``xipppy``
    quattrocento.py  QuattrocentoDevice  OTBioLab+ TCP socket (127.0.0.1:31000)
    simulated.py     SimulatedDevice     playback from a ``.npz`` recording

They share no base class and are duck-typed: the WebSocket servers call the same
handful of methods on whichever one they are given. If that implicit interface grows,
this is where a ``Protocol`` describing it belongs.
"""
