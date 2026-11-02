#!/usr/bin/env python3
"""Mock backend for the demo dashboard, without the window.

Run this alongside `npm run dev -w @porthole/demo` for frontend work with hot
reload. Nothing about the frontend knows this is synthetic — it is a Python
process publishing on the real socket, which is the entire point.

    p python/examples/demo_publisher.py

Meshes are not served here: Vite's dev server owns the origin in this mode, so
mount them there if you need them. `demo_app.py` serves them in one process.
"""

from __future__ import annotations

from demo_backend import handle_demo_commands, start_demo_backend

from porthole import PortholeBus


def main() -> None:
    bus = PortholeBus(app="demo").start()
    start_demo_backend(bus)

    print(f"publishing on ws://127.0.0.1:{bus.port} — ctrl-c to stop", flush=True)
    try:
        handle_demo_commands(bus, echo=True)
    except KeyboardInterrupt:
        pass
    finally:
        bus.stop()


if __name__ == "__main__":
    main()
