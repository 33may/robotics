"""sources — data providers behind stable protocols.

Panels depend on a source *protocol* (e.g. CameraSource), never on Isaac or a socket, so
the concrete backend (test / Isaac / real robot) is swapped with zero panel change. Each
source must be non-blocking (latest-wins) so it never stalls the UI thread.
"""
