#!/usr/bin/env python3
"""End-to-end: commands over a real websocket drive the real Supervisor.
Run: p inspection/tests/test_mock_bus.py"""
import json
import tempfile
import time
from pathlib import Path

from porthole import PortholeBus
from inspection.tests.test_machine import wait_for
from inspection.ui.publisher import InspectionPublisher


def test_ws_command_reaches_previewing():
    from websockets.sync.client import connect
    from inspection.ui.mock import start_mock

    bus = PortholeBus(app="inspection", port=8899).start()
    pub = InspectionPublisher(bus)
    sup = start_mock(bus, pub, Path(tempfile.mkdtemp()) / "run")
    with connect("ws://127.0.0.1:8899") as ws:
        ws.send(json.dumps({"cmd": "view/request", "target": "survey"}))
        wait_for(lambda: sup.phase in ("planning", "previewing"), timeout=30,
                 msg="command never reached the supervisor")
        wait_for(lambda: sup.phase == "previewing", timeout=60, msg="previewing")
        ws.send(json.dumps({"cmd": "run/exit"}))
        wait_for(lambda: sup.phase == "done", timeout=15, msg="exit")
    bus.stop()
    print("OK test_mock_bus")


if __name__ == "__main__":
    test_ws_command_reaches_previewing()
