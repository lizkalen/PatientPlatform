"""Signal processing shared by the live server and the legacy CLI pipeline.

``processing`` — filter design and stateful filtering, chunk-wise decomposition
(``process_chunk``), and firing-rate estimation both chunk-wise
(``calculate_firing_rates``) and whole-recording (``firing_rate_sliding_window``).

``loading`` — pretrained decomposition models (MU filters, whitening matrix,
centroids, normalisation factors).

This package exists because these two modules are the only part of the old ``online``
package that both ``server`` and ``legacy_pipeline`` depend on. Keeping them here is
what lets the legacy code be deleted later without touching the live server.
"""
